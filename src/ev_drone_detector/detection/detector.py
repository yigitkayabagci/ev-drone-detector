"""DroneDetector — end-to-end initial drone detection pipeline.

Takes raw event data, runs SPGNet segmentation, and outputs bounding boxes
for the detected drone(s). Designed for initial detection to be handed off
to a downstream tracking algorithm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ev_drone_detector.data.event_repr import sparse_to_device, voxelize_events
from ev_drone_detector.data.preprocessing import EventPreprocessor
from ev_drone_detector.detection.clustering import segmentation_to_detections
from ev_drone_detector.models.spgnet import SPGNet
from ev_drone_detector.utils.config import Config, load_config
from ev_drone_detector.utils.eval import _bbox_iou


class DroneDetector:
    """End-to-end drone detector using SPGNet.

    Usage:
        detector = DroneDetector.from_config("configs/default.yaml")
        detector.load_weights("checkpoints/best_model.pt")
        detections = detector.detect(features, coords)

    Args:
        model: SPGNet model instance.
        config: Configuration object.
        device: Torch device.
    """

    def __init__(
        self,
        model: SPGNet,
        config: Config,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.config = config

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        device: torch.device | None = None,
    ) -> DroneDetector:
        """Create a DroneDetector from a config file."""
        cfg = load_config(config_path)
        model = SPGNet(
            input_channel=cfg.model.input_channel,
            width=cfg.model.width,
            spatial_shape=cfg.sensor.spatial_shape,
            dilations=cfg.model.dilations,
        )
        return cls(model, cfg, device)

    def load_weights(self, path: str | Path) -> None:
        """Load model weights from a checkpoint file."""
        path = Path(path)
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        # Handle wrapped state dicts
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def detect(
        self,
        features: torch.Tensor | np.ndarray,
        coords: torch.Tensor | np.ndarray,
    ) -> list[dict]:
        """Detect drones in a single event stream.

        Args:
            features: (N, 4) event features [x_norm, y_norm, t_norm, polarity].
            coords: (N, 3) integer voxel coordinates [x, y, t].

        Returns:
            List of detection dicts with 'bbox', 'score', 'num_events', 'center'.
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).long()

        cfg = self.config
        spatial_shape = cfg.sensor.spatial_shape

        # Voxelize
        voxel_tensor, p2v_map = voxelize_events(
            features, coords, batch_idx=0, spatial_shape=spatial_shape
        )

        # Move to device
        voxel_tensor = sparse_to_device(voxel_tensor, self.device)
        p2v_map = p2v_map.to(self.device)

        # Forward pass
        predictions, _ = self.model(voxel_tensor)

        # Convert to bounding boxes
        det_cfg = cfg.detection
        detections = segmentation_to_detections(
            predictions=predictions,
            coords=coords,
            p2v_map=p2v_map.cpu(),
            threshold=det_cfg.seg_threshold,
            eps=det_cfg.cluster_eps,
            min_samples=det_cfg.cluster_min_samples,
            min_cluster_size=det_cfg.min_cluster_size,
            bbox_padding=det_cfg.bbox_padding,
            max_detections=det_cfg.max_detections,
            image_size=tuple(cfg.sensor.resolution),
        )

        return detections

    @torch.no_grad()
    def detect_from_npz(self, npz_path: str | Path) -> list[dict]:
        """Detect drones from an EV-UAV .npz file.

        Args:
            npz_path: Path to .npz file with 'evs_norm' and 'ev_loc' arrays.

        Returns:
            List of detection dicts.
        """
        data = np.load(str(npz_path), allow_pickle=True)
        features = torch.from_numpy(data["evs_norm"][:, 0:4].astype(np.float32))
        coords = torch.from_numpy(data["ev_loc"].astype(np.int64))
        return self.detect(features, coords)

    @torch.no_grad()
    def detect_stream(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
        preprocessor: EventPreprocessor,
        nms_iou: float = 0.5,
    ) -> list[dict]:
        """Run tiled sliding-window inference on a raw event stream.

        This is the canonical entry point for real-life event data from any
        source — Prophesee EVK4 .hdf5, DAVIS .aedat, live cameras, etc.
        Caller supplies four parallel arrays plus a preprocessor configured
        for the source sensor; this method handles tiling, polarity / time
        normalization, per-tile inference, mapping bboxes back to source
        coordinates, and NMS across overlapping tiles.

        Args:
            x: (N,) int pixel x coords in source resolution.
            y: (N,) int pixel y coords in source resolution.
            t: (N,) int64 timestamps (microseconds, any reference frame).
            p: (N,) polarity ({0, 1} or {-1, +1} — auto-detected).
            preprocessor: Preconfigured `EventPreprocessor`.
            nms_iou: IoU threshold above which two detections in the same
                time window are deduplicated (higher-score one kept).

        Returns:
            List of detection dicts whose 'bbox' / 'center' are in
            SOURCE-image pixel coords. Each detection also carries
            't_window_us': (t_start, t_end) for the window it was found in.
        """
        all_dets: list[dict] = []
        for tw in preprocessor(x, y, t, p):
            local_dets = self.detect(tw.features, tw.coords)
            x0, y0 = tw.tile_origin
            for d in local_dets:
                b = d["bbox"]
                d["bbox"] = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
                cx, cy = d["center"]
                d["center"] = [cx + x0, cy + y0]
                d["t_window_us"] = tw.t_window_us
                all_dets.append(d)
        return _nms_detections(all_dets, iou_threshold=nms_iou)


def _nms_detections(
    detections: list[dict], iou_threshold: float = 0.5,
) -> list[dict]:
    """Greedy NMS that only competes detections inside the same time window.

    Detections in different time windows describe different temporal events,
    so they should never suppress each other regardless of spatial overlap.
    """
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept: list[dict] = []
    for d in sorted_dets:
        suppressed = False
        for k in kept:
            if d.get("t_window_us") != k.get("t_window_us"):
                continue
            if _bbox_iou(d["bbox"], k["bbox"]) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(d)
    return kept
