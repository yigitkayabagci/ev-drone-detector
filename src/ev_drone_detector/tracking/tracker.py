"""Sliding-window drone tracker built on top of DroneDetector.

SPGNet is trained on 8-second event windows; this module slides that
window across a longer recording and links per-window detections into
tracks via IoU + Hungarian assignment. The detector stays unchanged —
tracking is a pure post-processing layer.

Two input modes:
  - track_npz_sequence(paths)  → one .npz file per window (EV-UAV layout)
  - track_event_stream(...)    → raw event stream with configurable stride
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ev_drone_detector.tracking.association import match_by_iou

if TYPE_CHECKING:
    from ev_drone_detector.detection.detector import DroneDetector


@dataclass
class Track:
    """State for a single tracked drone."""

    track_id: int
    bbox: list[int]
    score: float
    last_window: int
    age: int = 0
    hits: int = 1
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "score": float(self.score),
            "age": self.age,
            "hits": self.hits,
            "last_window": self.last_window,
        }


class SlidingWindowTracker:
    """Tracks drones across consecutive 8-second windows.

    The tracker is detector-agnostic at the `update()` level — it only
    consumes per-window detection dicts (as produced by `DroneDetector`).
    Convenience methods drive the detector for common input layouts.

    Args:
        detector: Initialized `DroneDetector`.
        iou_threshold: Min IoU for a detection to inherit an existing track ID.
        max_age: Windows a track may stay unmatched before deletion.
        min_hits: Min total hits before a track shows up in `active_tracks`
            (suppresses one-frame flickers).
    """

    def __init__(
        self,
        detector: "DroneDetector",
        iou_threshold: float = 0.3,
        max_age: int = 3,
        min_hits: int = 1,
        extrapolate: bool = True,
    ):
        self.detector = detector
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.extrapolate = extrapolate

        self.tracks: list[Track] = []
        self._next_id = 1
        self._window_idx = 0

    @property
    def _extrapolation_target(self) -> float | None:
        """Voxel-t at which to predict each window's bbox.

        Returns the last in-range voxel-t of the SPGNet training window
        (`whole_t - 1`) so a sliding-window track reports the drone's
        position at the *end* of each window instead of its 8-second trail.
        """
        if not self.extrapolate:
            return None
        return float(self.detector.config.sensor.whole_t - 1)

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1
        self._window_idx = 0

    def update(self, detections: list[dict]) -> list[dict]:
        """Advance the tracker by one window.

        Args:
            detections: Output of `DroneDetector.detect(...)` for this window
                — each dict has at least 'bbox' and 'score'.

        Returns:
            Active tracks at this window (those matched this step and past
            the `min_hits` warmup), each as a Track.to_dict() payload.
        """
        det_boxes = [d["bbox"] for d in detections]
        track_boxes = [t.bbox for t in self.tracks]

        matches, unmatched_tr, unmatched_det = match_by_iou(
            track_boxes, det_boxes, iou_threshold=self.iou_threshold
        )

        for t_idx, d_idx in matches:
            tr = self.tracks[t_idx]
            det = detections[d_idx]
            tr.bbox = list(det["bbox"])
            tr.score = float(det["score"])
            tr.last_window = self._window_idx
            tr.age = 0
            tr.hits += 1
            tr.history.append({
                "window": self._window_idx,
                "bbox": list(det["bbox"]),
                "score": float(det["score"]),
            })

        for t_idx in unmatched_tr:
            self.tracks[t_idx].age += 1

        for d_idx in unmatched_det:
            det = detections[d_idx]
            tr = Track(
                track_id=self._next_id,
                bbox=list(det["bbox"]),
                score=float(det["score"]),
                last_window=self._window_idx,
            )
            tr.history.append({
                "window": self._window_idx,
                "bbox": list(det["bbox"]),
                "score": float(det["score"]),
            })
            self.tracks.append(tr)
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        active = [
            t.to_dict()
            for t in self.tracks
            if t.age == 0 and t.hits >= self.min_hits
        ]
        self._window_idx += 1
        return active

    @torch.no_grad()
    def track_npz_sequence(
        self,
        npz_paths: list[str | Path],
        stride: int = 1,
    ) -> list[dict]:
        """Run detector + tracker over a sorted list of .npz windows.

        Each `.npz` is treated as one 8-second window (the EV-UAV format).
        `stride` skips every Nth file (1 = process every file).

        Returns:
            One dict per processed window with: window, file, detections,
            active_tracks.
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.reset()
        target_t = self._extrapolation_target
        results = []
        selected = list(npz_paths)[::stride]
        for i, path in enumerate(selected):
            detections = self.detector.detect_from_npz(path, extrapolate_to_t=target_t)
            active = self.update(detections)
            results.append({
                "window": i,
                "file": str(path),
                "detections": detections,
                "active_tracks": active,
            })
        return results

    @torch.no_grad()
    def track_event_stream(
        self,
        features: np.ndarray | torch.Tensor,
        coords: np.ndarray | torch.Tensor,
        times_us: np.ndarray | torch.Tensor,
        window_us: int = 8_000_000,
        stride_us: int = 1_000_000,
    ) -> list[dict]:
        """Slide an 8-second window across a long event stream.

        Args:
            features: (N, 4) features [x_norm, y_norm, t_norm, polarity].
                The t_norm column is recomputed per window — its original
                values are ignored.
            coords:   (N, 3) integer voxel coords [x, y, t]. The t column is
                also recomputed per window from `times_us`.
            times_us: (N,) absolute event timestamps in microseconds. Drives
                window slicing and time-normalization.
            window_us: Length of each window in microseconds. Default 8e6
                matches the SPGNet training horizon.
            stride_us: How far the window advances each step. stride_us=window_us
                gives no overlap; smaller values give overlapped tracking.

        Returns:
            One dict per produced window with: window, t_start_us, t_end_us,
            num_events, detections, active_tracks.
        """
        if window_us <= 0 or stride_us <= 0:
            raise ValueError("window_us and stride_us must be positive")

        feats = _to_numpy(features).astype(np.float32, copy=False)
        crds = _to_numpy(coords).astype(np.int64, copy=False)
        ts = _to_numpy(times_us).astype(np.int64, copy=False)

        if feats.shape[0] != crds.shape[0] or feats.shape[0] != ts.shape[0]:
            raise ValueError(
                f"features ({feats.shape[0]}), coords ({crds.shape[0]}) and "
                f"times_us ({ts.shape[0]}) must have matching row counts"
            )
        if feats.shape[0] == 0:
            return []

        whole_t = int(self.detector.config.sensor.whole_t)
        # Microseconds per voxel-t tick — keeps the voxel grid consistent
        # with how training samples were normalized.
        voxel_step_us = window_us / float(whole_t)

        self.reset()
        target_t = self._extrapolation_target
        results: list[dict] = []

        t_start = int(ts.min())
        t_end_global = int(ts.max())
        window_idx = 0
        while t_start <= t_end_global:
            t_stop = t_start + window_us
            mask = (ts >= t_start) & (ts < t_stop)
            n_in_window = int(mask.sum())

            if n_in_window == 0:
                results.append({
                    "window": window_idx,
                    "t_start_us": int(t_start),
                    "t_end_us": int(t_stop),
                    "num_events": 0,
                    "detections": [],
                    "active_tracks": self.update([]),
                })
                t_start += stride_us
                window_idx += 1
                continue

            win_feats = feats[mask].copy()
            win_coords = crds[mask].copy()
            win_t = ts[mask] - t_start

            win_feats[:, 2] = win_t.astype(np.float32) / float(window_us)
            win_coords[:, 2] = np.clip(
                (win_t / voxel_step_us).astype(np.int64), 0, whole_t - 1
            )

            detections = self.detector.detect(
                torch.from_numpy(win_feats),
                torch.from_numpy(win_coords),
                extrapolate_to_t=target_t,
            )
            active = self.update(detections)
            results.append({
                "window": window_idx,
                "t_start_us": int(t_start),
                "t_end_us": int(t_stop),
                "num_events": n_in_window,
                "detections": detections,
                "active_tracks": active,
            })

            t_start += stride_us
            window_idx += 1

        return results


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
