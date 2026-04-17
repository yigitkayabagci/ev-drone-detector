"""Render an MP4 that shows drone motion through time.

The default scripts/detect.py video collapses each .npz's 8-second event
window into a single frame (you see a long "motion trail" rather than
the drone actually flying). This script slices each .npz along the
temporal axis into short sub-windows (default 33.3 ms = 30 FPS) and
writes one video frame per sub-window, so the drone's real motion is
visible in real time.

Optional: overlay bounding boxes from a detections.json produced by
scripts/detect.py. The bbox for an .npz is drawn on every sub-frame of
that .npz (it spans the drone's full 8 s trajectory — watching the
drone fly through the box is the visual proof the detector is right).

Usage:
    # Just motion, no bbox
    uv run --no-sync python frame_detector/make_motion_video.py \\
        --input data/test \\
        --output motion.mp4 \\
        --window_ms 33.3 --fps 30

    # With bbox overlay from scripts/detect.py output
    uv run --no-sync python frame_detector/make_motion_video.py \\
        --input data/test \\
        --output motion_with_boxes.mp4 \\
        --window_ms 33.3 --fps 30 \\
        --detections results/detections.json

    # Single .npz, slower playback (10 FPS)
    uv run --no-sync python frame_detector/make_motion_video.py \\
        --input data/test/test_000.npz \\
        --output motion_000.mp4 \\
        --window_ms 33.3 --fps 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def accumulate_polarity_rgb(
    x: np.ndarray, y: np.ndarray, polarity: np.ndarray, resolution: tuple[int, int],
) -> np.ndarray:
    """Gray background, positive events blue, negative red. H x W x 3 uint8."""
    W, H = resolution
    frame = np.full((H, W, 3), 128, dtype=np.uint8)
    if x.size == 0:
        return frame
    xi = np.clip(x.astype(np.int64), 0, W - 1)
    yi = np.clip(y.astype(np.int64), 0, H - 1)
    pos = polarity > 0
    neg = ~pos
    frame[yi[pos], xi[pos]] = [255, 200, 200]
    frame[yi[neg], xi[neg]] = [200, 200, 255]
    return frame


def draw_bboxes(frame: np.ndarray, bboxes: list[dict],
                color: tuple[int, int, int] = (0, 255, 0),
                thickness: int = 2) -> np.ndarray:
    import cv2
    out = frame.copy()
    for det in bboxes:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        score = det.get("score")
        if score is not None:
            cv2.putText(out, f"{score:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Event-motion MP4 with optional bbox overlay")
    p.add_argument("--input", required=True, help=".npz file or directory")
    p.add_argument("--output", required=True, help="Output .mp4 path")
    p.add_argument("--window_ms", type=float, default=33.3,
                   help="Sub-window length in ms (default 33.3 = 30 FPS real-time)")
    p.add_argument("--fps", type=float, default=30.0, help="Video playback frame rate")
    p.add_argument("--resolution", nargs=2, type=int, default=[346, 260],
                   metavar=("W", "H"))
    p.add_argument("--detections", type=str, default=None,
                   help="detections.json from scripts/detect.py for bbox overlay")
    p.add_argument("--max_files", type=int, default=None,
                   help="Limit number of .npz files processed (for quick tests)")
    return p.parse_args()


def load_detections(path: str | None) -> dict[str, list[dict]]:
    if path is None:
        return {}
    with open(path) as f:
        data = json.load(f)
    return {Path(k).name: v for k, v in data.items()}


def iter_subframes(ev_loc: np.ndarray, polarity: np.ndarray, window_ms: float):
    """Yield (x_slice, y_slice, p_slice) per temporal sub-window."""
    t = ev_loc[:, 2].astype(np.float64)
    if t.size == 0:
        return
    t_min = float(t.min())
    t_max = float(t.max())
    if t_max <= t_min:
        yield ev_loc[:, 0], ev_loc[:, 1], polarity
        return
    n_bins = max(int(np.ceil((t_max - t_min) / window_ms)), 1)
    edges = t_min + np.arange(n_bins + 1) * window_ms
    bin_idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_bins - 1)
    for i in range(n_bins):
        mask = bin_idx == i
        yield ev_loc[mask, 0], ev_loc[mask, 1], polarity[mask]


def main() -> None:
    args = parse_args()
    import cv2

    W, H = int(args.resolution[0]), int(args.resolution[1])
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        files = [in_path]
    else:
        files = sorted(in_path.glob("*.npz"))
        if not files:
            files = sorted(in_path.rglob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found under {in_path}")
    if args.max_files is not None:
        files = files[: args.max_files]

    det_by_name = load_detections(args.detections)
    overlay = bool(det_by_name)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")

    total_frames = 0
    try:
        for f in files:
            data = np.load(str(f), allow_pickle=True)
            ev_loc = data["ev_loc"]
            polarity = data["evs_norm"][:, 3]
            bboxes = det_by_name.get(f.name, []) if overlay else []

            n_sub = 0
            for x, y, p in iter_subframes(ev_loc, polarity, args.window_ms):
                frame = accumulate_polarity_rgb(x, y, p, (W, H))
                if bboxes:
                    frame = draw_bboxes(frame, bboxes)
                writer.write(frame)
                n_sub += 1
            total_frames += n_sub
            print(f"{f.name}: {n_sub} sub-frames"
                  + (f" (with {len(bboxes)} bbox)" if bboxes else ""))
    finally:
        writer.release()

    duration = total_frames / max(args.fps, 1e-6)
    print(f"\nDone. {len(files)} file(s) -> {out_path}")
    print(f"Total frames: {total_frames}  |  Duration: {duration:.1f} s "
          f"@ {args.fps} fps")


if __name__ == "__main__":
    main()
