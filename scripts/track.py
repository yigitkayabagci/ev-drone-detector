"""Inference script for sliding-window drone tracking.

Runs SPGNet over a sequence of 8-second event windows and links per-window
detections into tracks (one ID per drone) via IoU + Hungarian matching.

Usage:
    # Sequence mode — one .npz per 8s window
    uv run python scripts/track.py --checkpoint ckpt.pt --input data/test_seq/

    # Stream mode — single .npz with absolute event timestamps
    uv run python scripts/track.py --checkpoint ckpt.pt --input data/long.npz --mode stream \\
        --window-us 8000000 --stride-us 4000000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ev_drone_detector.detection.detector import DroneDetector
from ev_drone_detector.tracking.tracker import SlidingWindowTracker
from ev_drone_detector.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliding-window drone tracking")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True,
                        help="Directory of .npz files (sequence mode) or single .npz (stream mode)")
    parser.add_argument("--mode", choices=["sequence", "stream"], default="sequence")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--max-age", type=int, default=None)
    parser.add_argument("--min-hits", type=int, default=None)
    # Sequence mode
    parser.add_argument("--npz-stride", type=int, default=None,
                        help="(sequence) process every Nth .npz file")
    # Stream mode
    parser.add_argument("--window-us", type=int, default=None,
                        help="(stream) window length in microseconds")
    parser.add_argument("--stride-us", type=int, default=None,
                        help="(stream) window stride in microseconds")
    parser.add_argument("--times-key", type=str, default="times_us",
                        help="(stream) key in .npz with absolute event timestamps in microseconds")
    return parser.parse_args()


def _resolve(arg_value, cfg_value):
    return cfg_value if arg_value is None else arg_value


def main() -> None:
    args = parse_args()

    import torch
    device = torch.device(args.device) if args.device else None

    cfg = load_config(args.config)
    detector = DroneDetector.from_config(args.config, device=device)
    detector.load_weights(args.checkpoint)
    print(f"Loaded model from {args.checkpoint}")

    tcfg = cfg.tracking
    tracker = SlidingWindowTracker(
        detector,
        iou_threshold=_resolve(args.iou_threshold, tcfg.iou_threshold),
        max_age=_resolve(args.max_age, tcfg.max_age),
        min_hits=_resolve(args.min_hits, tcfg.min_hits),
    )

    input_path = Path(args.input)

    if args.mode == "sequence":
        if input_path.is_file():
            npz_files = [input_path]
        else:
            npz_files = sorted(input_path.glob("*.npz"))
            if not npz_files:
                npz_files = sorted(input_path.rglob("*.npz"))
        if not npz_files:
            raise SystemExit(f"No .npz files found under {input_path}")

        print(f"Sequence mode: {len(npz_files)} window(s)")
        results = tracker.track_npz_sequence(
            npz_files, stride=_resolve(args.npz_stride, tcfg.npz_stride)
        )
    else:
        if not input_path.is_file():
            raise SystemExit(f"Stream mode requires a single .npz; got {input_path}")
        data = np.load(str(input_path), allow_pickle=True)
        if args.times_key not in data.files:
            raise SystemExit(
                f"Stream mode needs absolute timestamps in .npz key "
                f"'{args.times_key}'. Available keys: {list(data.files)}"
            )
        features = data["evs_norm"][:, 0:4].astype(np.float32)
        coords = data["ev_loc"].astype(np.int64)
        times_us = data[args.times_key].astype(np.int64)

        window_us = _resolve(args.window_us, tcfg.window_us)
        stride_us = _resolve(args.stride_us, tcfg.stride_us)
        n_windows_est = max(
            1, 1 + (int(times_us.max()) - int(times_us.min())) // stride_us
        )
        print(f"Stream mode: ~{n_windows_est} window(s), "
              f"window={window_us}us, stride={stride_us}us")

        results = tracker.track_event_stream(
            features, coords, times_us, window_us=window_us, stride_us=stride_us
        )

    # Per-window summary to stdout
    track_first_seen: dict[int, int] = {}
    for win in results:
        active = win["active_tracks"]
        ids = sorted(t["track_id"] for t in active)
        for t in active:
            track_first_seen.setdefault(t["track_id"], win["window"])
        print(f"  window {win['window']:>4}: "
              f"{len(win['detections'])} det, {len(active)} active "
              f"track_ids={ids}")

    print(f"\nTotal unique tracks: {len(track_first_seen)}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        print(f"Saved results to {out_path}")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
