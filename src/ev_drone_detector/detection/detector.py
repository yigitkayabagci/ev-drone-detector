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
from ev_drone_detector.detection.clustering import segmentation_to_detections
from ev_drone_detector.models.spgnet import SPGNet
from ev_drone_detector.utils.config import Config, load_config


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
