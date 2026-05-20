"""Inference script for sliding-window drone tracking.

Runs SPGNet over a sequence of 8-second event windows and links per-window
detections into tracks (one ID per drone) via IoU + Hungarian matching.
Optionally renders per-window PNGs or a video with colored bboxes (one
color per track ID) and a fading trajectory trail.

Usage:
    # Sequence mode — one .npz per 8s window
    uv run python scripts/track.py --checkpoint ckpt.pt --input data/test_seq/

    # Stream mode — single .npz with absolute event timestamps
    uv run python scripts/track.py --checkpoint ckpt.pt --input data/long.npz --mode stream \\
        --window-us 8000000 --stride-us 1000000

    # With video output and 20-window trails
    uv run python scripts/track.py --checkpoint ckpt.pt --input data/test_seq/ \\
        --video tracks.mp4 --trail 20 --fps 10
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
from ev_drone_detector.utils.viz import TrailBuffer, draw_tracks, events_to_frame


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
    parser.add_argument("--extrapolate", dest="extrapolate", action="store_true", default=None,
                        help="Fit linear motion per cluster and emit bbox at window end")
    parser.add_argument("--no-extrapolate", dest="extrapolate", action="store_false",
                        help="Emit raw trail bbox (no motion fit)")
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
    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Save per-window PNG visualizations to --vis-dir")
    parser.add_argument("--vis-dir", type=str, default="visualizations",
                        help="Directory for per-window PNGs (with --visualize)")
    parser.add_argument("--video", type=str, default=None,
                        help="Write a video of all windows to this path (e.g. tracks.mp4)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Frames per second for --video (default 10)")
    parser.add_argument("--trail", type=int, default=20,
                        help="Number of past windows to draw as trajectory trail (0 = off)")
    parser.add_argument("--show-velocity", action="store_true",
                        help="Draw extrapolated velocity arrow on each track")
    return parser.parse_args()


def _resolve(arg_value, cfg_value):
    return cfg_value if arg_value is None else arg_value


def _frame_for_sequence(npz_path: str, resolution: tuple[int, int]) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    coords = data["ev_loc"]
    pol = data["evs_norm"][:, 3]
    return events_to_frame(
        coords[:, 0].astype(float),
        coords[:, 1].astype(float),
        polarity=pol,
        resolution=resolution,
    )


def _frame_for_stream(
    times_us: np.ndarray,
    coords: np.ndarray,
    features: np.ndarray,
    t_start: int,
    t_end: int,
    resolution: tuple[int, int],
) -> np.ndarray:
    mask = (times_us >= t_start) & (times_us < t_end)
    if not mask.any():
        H, W = resolution[1], resolution[0]
        return np.full((H, W, 3), 128, dtype=np.uint8)
    return events_to_frame(
        coords[mask, 0].astype(float),
        coords[mask, 1].astype(float),
        polarity=features[mask, 3],
        resolution=resolution,
    )


def _open_video(video_path: str, fps: float, resolution: tuple[int, int]):
    import cv2
    W, H = resolution
    out_path = Path(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (int(W), int(H)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")
    return writer


def _save_png(frame: np.ndarray, path: Path) -> None:
    try:
        import cv2
        cv2.imwrite(str(path), frame)
    except ImportError:
        import matplotlib.pyplot as plt
        plt.imsave(str(path), frame)


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
        extrapolate=_resolve(args.extrapolate, tcfg.extrapolate),
    )

    resolution = tuple(cfg.sensor.resolution)
    input_path = Path(args.input)

    # ----- run tracker --------------------------------------------------------

    stream_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

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
        stream_data = (features, coords, times_us)

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

    # ----- per-window stdout summary -----------------------------------------

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

    # ----- visualization ------------------------------------------------------

    needs_render = args.visualize or args.video is not None
    if needs_render:
        trail_len = max(0, int(args.trail))
        trail_buf = TrailBuffer(max_length=trail_len) if trail_len > 0 else None
        vis_dir = Path(args.vis_dir) if args.visualize else None
        if vis_dir is not None:
            vis_dir.mkdir(parents=True, exist_ok=True)

        writer = _open_video(args.video, args.fps, resolution) if args.video else None
        try:
            for win in results:
                if args.mode == "sequence":
                    frame = _frame_for_sequence(win["file"], resolution)
                else:
                    features, coords, times_us = stream_data  # type: ignore[misc]
                    frame = _frame_for_stream(
                        times_us, coords, features,
                        int(win["t_start_us"]), int(win["t_end_us"]),
                        resolution,
                    )

                if trail_buf is not None:
                    trail_buf.update(win["active_tracks"])

                frame = draw_tracks(
                    frame,
                    win["active_tracks"],
                    trails=trail_buf.trails if trail_buf is not None else None,
                    show_velocity=args.show_velocity,
                )

                if vis_dir is not None:
                    _save_png(frame, vis_dir / f"window_{win['window']:04d}.png")
                if writer is not None:
                    writer.write(frame)
        finally:
            if writer is not None:
                writer.release()
                print(f"Saved video: {args.video}")
        if vis_dir is not None:
            print(f"Saved per-window PNGs to: {vis_dir}")

    # ----- JSON output --------------------------------------------------------

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        print(f"Saved results to {args.output}")


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
