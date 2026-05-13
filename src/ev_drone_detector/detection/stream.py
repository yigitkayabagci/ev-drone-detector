"""Stream-level helpers for raw-event inference and visualization.

These functions form the reusable core behind ``scripts/detect_stream.py``:

  - ``load_stream``        — auto-detect .hdf5 / .npz and return (x, y, t, p)
  - ``render_window_frame``— build one (H, W, 3) frame for a time window
  - ``iter_window_frames`` — generate (window_idx, frame) for every window
                             a detector emitted detections in
  - ``BBOX_COLORS``        — BGR colors for predicted boxes
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ev_drone_detector.data.fred import load_fred_events
from ev_drone_detector.data.preprocessing import EventPreprocessor
from ev_drone_detector.utils.viz import draw_detections, events_to_frame


HDF5_SUFFIXES = {".h5", ".hdf5"}
NPZ_SUFFIXES = {".npz"}

BBOX_COLORS: dict[str, tuple[int, int, int]] = {
    "red":    (0, 0, 255),
    "green":  (0, 255, 0),
    "blue":   (255, 0, 0),
    "yellow": (0, 255, 255),
    "white":  (255, 255, 255),
}


def load_npz_stream(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read x, y, t, p arrays from a generic raw-event .npz file."""
    path = Path(path)
    data = np.load(str(path), allow_pickle=True)
    missing = [k for k in ("x", "y", "t", "p") if k not in data.files]
    if missing:
        raise KeyError(
            f"{path}: missing required keys {missing}. "
            f"Available: {list(data.files)}"
        )
    return data["x"], data["y"], data["t"], data["p"]


def load_stream(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Auto-detect file format and return (x, y, t, p) arrays.

    Supports FRED HDF5 (``.h5``/``.hdf5``) and a generic ``.npz`` containing
    ``x``, ``y``, ``t``, ``p`` keys. For other formats, write a thin loader
    that returns the same four arrays and use that directly.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in HDF5_SUFFIXES:
        return load_fred_events(path)
    if suffix in NPZ_SUFFIXES:
        return load_npz_stream(path)
    raise ValueError(
        f"Unsupported input extension {suffix!r}. "
        f"Supported: {sorted(HDF5_SUFFIXES | NPZ_SUFFIXES)}."
    )


def render_window_frame(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    t_window_us: tuple[int, int],
    pred_detections: list[dict],
    gt_bboxes: list[list[int]],
    resolution: tuple[int, int],
    bbox_color: tuple[int, int, int] = (0, 0, 255),
    gt_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Build one (H, W, 3) BGR frame for a time window with overlaid bboxes.

    Args:
        x, y, p, t: Parallel event arrays. ``t`` must already be in the
            same reference frame as ``t_window_us`` (typically zeroed at
            the stream start).
        t_window_us: (t_start, t_end) of the window to render, in the
            same time units as ``t``.
        pred_detections: Model output; each dict needs ``bbox`` and
            ``score``. Drawn in ``bbox_color``.
        gt_bboxes: List of GT boxes ``[x1, y1, x2, y2]`` in source coords.
            Drawn in ``gt_color``. Pass an empty list to omit.
        resolution: (W, H) of the source sensor for the frame canvas.
        bbox_color, gt_color: BGR colors.
    """
    t = np.asarray(t)
    t0, t1 = t_window_us
    mask = (t >= t0) & (t < t1)
    frame = events_to_frame(
        np.asarray(x)[mask], np.asarray(y)[mask],
        polarity=np.asarray(p)[mask], resolution=resolution,
    )
    if gt_bboxes:
        frame = draw_detections(
            frame,
            [{"bbox": list(b), "score": 1.0} for b in gt_bboxes],
            color=gt_color, thickness=1,
        )
    if pred_detections:
        frame = draw_detections(
            frame, pred_detections, color=bbox_color, thickness=2,
        )
    return frame


def iter_window_frames(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    detections: list[dict],
    preprocessor: EventPreprocessor,
    gt_bboxes_by_window: dict[tuple[int, int], list[list[int]]] | None = None,
    resolution: tuple[int, int] = (1280, 720),
    bbox_color: tuple[int, int, int] = (0, 0, 255),
    gt_color: tuple[int, int, int] = (255, 255, 255),
) -> Iterator[tuple[int, tuple[int, int], np.ndarray]]:
    """Yield one (window_idx, t_window_us, frame) per temporal window.

    Stream time is internally rezeroed to ``t.min()`` so that windows
    align with the preprocessor's ``target_t_us`` step.
    """
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    t = np.asarray(t, dtype=np.int64)
    p = np.asarray(p)

    if len(t) == 0:
        return
    t_rel = t - int(t.min())
    t_max = int(t_rel.max())
    n_windows = max(1, t_max // preprocessor.target_t_us + 1)

    dets_by_window: dict[tuple[int, int], list[dict]] = {}
    for d in detections:
        dets_by_window.setdefault(tuple(d["t_window_us"]), []).append(d)

    gt_by_window = gt_bboxes_by_window or {}

    for w in range(n_windows):
        t_start = w * preprocessor.target_t_us
        t_end = t_start + preprocessor.target_t_us
        frame = render_window_frame(
            x, y, p, t_rel,
            t_window_us=(t_start, t_end),
            pred_detections=dets_by_window.get((t_start, t_end), []),
            gt_bboxes=gt_by_window.get((t_start, t_end), []),
            resolution=resolution,
            bbox_color=bbox_color,
            gt_color=gt_color,
        )
        yield w, (t_start, t_end), frame
