"""Animate the drone's event trail growing frame-by-frame from a single .npz.

Each video frame shows the CUMULATIVE events from t=0 up to the current
sub-window end, so the drone's motion trail "draws itself" progressively
instead of being pre-drawn (as in scripts/detect.py's single PNG).

Pair with --detections to also show the 8-second bounding box as a
static overlay while the trail grows inside it.

Usage:
    uv run --no-sync python frame_detector/draw_trail_animation.py \\
        --input data/test/test_000.npz \\
        --output test_000_drawing.mp4 \\
        --window_ms 33.3 --fps 30 \\
        --detections results/detections.json --show_box
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Progressive trail animation for one .npz")
    p.add_argument("--input", required=True, help="Single .npz file")
    p.add_argument("--output", required=True, help="Output .mp4 path")
    p.add_argument("--window_ms", type=float, default=33.3,
                   help="How much time each video frame adds (ms)")
    p.add_argument("--fps", type=float, default=30.0, help="Playback fps")
    p.add_argument("--resolution", nargs=2, type=int, default=[346, 260],
                   metavar=("W", "H"))
    p.add_argument("--detections", type=str, default=None,
                   help="detections.json from scripts/detect.py")
    p.add_argument("--show_box", action="store_true",
                   help="Draw the static 8-second bbox from --detections")
    p.add_argument("--fade_old", action="store_true",
                   help="Dim older events slightly so the head of the trail is brighter")
    return p.parse_args()


def load_bboxes(path, target_name):
    if path is None:
        return []
    with open(path) as f:
        data = json.load(f)
    for k, v in data.items():
        if Path(k).name == target_name:
            return v
    return []


def main():
    args = parse_args()
    import cv2

    W, H = int(args.resolution[0]), int(args.resolution[1])
    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Not a file: {in_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(str(in_path), allow_pickle=True)
    ev_loc = data["ev_loc"]
    polarity = data["evs_norm"][:, 3]

    x = np.clip(ev_loc[:, 0].astype(np.int64), 0, W - 1)
    y = np.clip(ev_loc[:, 1].astype(np.int64), 0, H - 1)
    t = ev_loc[:, 2].astype(np.float64)

    # Time-sort (EV-UAV usually already is, but guarantee it)
    order = np.argsort(t, kind="stable")
    x, y, t, polarity = x[order], y[order], t[order], polarity[order]

    t_min = float(t.min()) if t.size else 0.0
    t_max = float(t.max()) if t.size else 0.0
    duration_ms = t_max - t_min
    n_bins = max(int(np.ceil(duration_ms / args.window_ms)), 1)

    bboxes = load_bboxes(args.detections, in_path.name)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")

    # If --fade_old, we render incrementally and let the accumulated frame
    # decay slightly each step. Otherwise we render the cumulative set each
    # step so older events keep their full color.
    try:
        if args.fade_old:
            frame = np.full((H, W, 3), 128, dtype=np.uint8)
            cursor = 0
            for i in range(n_bins):
                t_end = t_min + (i + 1) * args.window_ms
                end_idx = int(np.searchsorted(t, t_end, side="right"))
                # Fade everything a bit (toward gray 128)
                frame = (
                    frame.astype(np.int16) * 0.97 + 128 * 0.03
                ).clip(0, 255).astype(np.uint8)
                # Stamp the newly-arrived events bright
                nx = x[cursor:end_idx]
                ny = y[cursor:end_idx]
                np_ = polarity[cursor:end_idx]
                pos = np_ > 0
                neg = ~pos
                frame[ny[pos], nx[pos]] = [255, 200, 200]
                frame[ny[neg], nx[neg]] = [200, 200, 255]
                cursor = end_idx

                out = frame.copy()
                if args.show_box:
                    for det in bboxes:
                        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                writer.write(out)
        else:
            # Cumulative: redraw full set up to t_end each frame
            for i in range(n_bins):
                t_end = t_min + (i + 1) * args.window_ms
                end_idx = int(np.searchsorted(t, t_end, side="right"))
                frame = np.full((H, W, 3), 128, dtype=np.uint8)
                if end_idx > 0:
                    sx = x[:end_idx]
                    sy = y[:end_idx]
                    sp = polarity[:end_idx]
                    pos = sp > 0
                    neg = ~pos
                    frame[sy[pos], sx[pos]] = [255, 200, 200]
                    frame[sy[neg], sx[neg]] = [200, 200, 255]
                if args.show_box:
                    for det in bboxes:
                        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                writer.write(frame)
    finally:
        writer.release()

    print(f"{in_path.name}: {n_bins} frames -> {out_path}")
    print(f"Duration: {n_bins / args.fps:.1f} s @ {args.fps} fps "
          f"(covers {duration_ms/1000:.1f} s of real event time)")


if __name__ == "__main__":
    main()
