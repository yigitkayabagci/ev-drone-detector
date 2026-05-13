"""End-to-end drone detection on raw event streams.

Input sources supported out of the box:

  - FRED HDF5 (Prophesee EVK4 HD)        ``--input path/to/events.hdf5``
  - Generic .npz with x, y, t, p arrays  ``--input path/to/stream.npz``

For other sources (live Prophesee, .aedat, .dat, Prophesee .raw EVT3...)
write a 5-10 line loader that returns the same ``(x, y, t, p)`` arrays and
feed them to ``DroneDetector.detect_stream`` exactly as ``run()`` below does.

Output modes (combinable):
  --output  out.json   per-window detection list (bboxes in source coords)
  --visualize          one PNG per time-window with event frame + bboxes
  --video out.mp4      MP4 of all time-windows back-to-back (1 frame/window)

Usage:
    uv run python scripts/detect_stream.py \\
        --checkpoint checkpoints/best_iou.pt \\
        --input fred/test/sequence_xx/events.hdf5 \\
        --fred-annotations fred/test/sequence_xx/coordinates.txt \\
        --source-resolution 1280 720 \\
        --tile-overlap 0.25 \\
        --video out.mp4 --visualize --vis-dir vis/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ev_drone_detector.data.fred import (
    FRED_DEFAULT_RESOLUTION,
    annotations_in_window,
    load_fred_annotations,
)
from ev_drone_detector.data.preprocessing import EventPreprocessor
from ev_drone_detector.detection.stream import (
    BBOX_COLORS,
    HDF5_SUFFIXES,
    iter_window_frames,
    load_stream,
)
from ev_drone_detector.utils.config import load_config

# DroneDetector pulls in spconv (CUDA-only); imported lazily inside run()
# so that --help works in CPU-only environments without spconv installed.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end drone detection on raw event streams",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--input", type=str, required=True,
        help="Path to .hdf5 (FRED) or .npz with x/y/t/p arrays",
    )
    p.add_argument(
        "--source-resolution", nargs=2, type=int, default=None,
        metavar=("W", "H"),
        help="Source sensor resolution. Defaults to FRED's 1280x720 for "
             ".hdf5, otherwise the config's sensor.resolution.",
    )
    p.add_argument(
        "--fred-annotations", type=str, default=None,
        help="Optional FRED coordinates.txt; ground-truth bboxes are "
             "overlaid in white on each window's frame for comparison.",
    )
    p.add_argument(
        "--tile-overlap", type=float, default=0.0,
        help="Spatial tile overlap fraction (0..0.9)",
    )
    p.add_argument(
        "--max-events-per-tile", type=int, default=None,
        help="Cap events per tile (random subsample)",
    )
    p.add_argument(
        "--polarity-mode", choices=["auto", "01", "pm1"], default="auto",
    )
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument(
        "--bbox-color", choices=sorted(BBOX_COLORS.keys()), default="red",
        help="Color for predicted boxes",
    )
    p.add_argument(
        "--visualize", action="store_true",
        help="Save one PNG per time-window with bboxes overlaid",
    )
    p.add_argument(
        "--vis-dir", type=str, default="visualizations",
        help="Output dir for --visualize",
    )
    p.add_argument(
        "--video", type=str, default=None,
        help="If set, write an MP4 with one frame per time-window",
    )
    p.add_argument(
        "--video-fps", type=float, default=2.0,
        help="Frames per second for --video output",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Optional JSON output path for the detection list",
    )
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def run(args: argparse.Namespace) -> list[dict]:
    """Programmatic entry point; returns the merged detection list."""
    import torch
    from ev_drone_detector.detection.detector import DroneDetector

    device = torch.device(args.device) if args.device else None

    cfg = load_config(args.config)
    detector = DroneDetector.from_config(args.config, device=device)
    detector.load_weights(args.checkpoint)
    print(f"Loaded model from {args.checkpoint}")

    input_path = Path(args.input)
    is_hdf5 = input_path.suffix.lower() in HDF5_SUFFIXES

    if args.source_resolution:
        source_resolution = tuple(args.source_resolution)
    elif is_hdf5:
        source_resolution = FRED_DEFAULT_RESOLUTION
    else:
        source_resolution = tuple(cfg.sensor.resolution)

    preprocessor = EventPreprocessor.from_config(
        cfg,
        source_resolution=source_resolution,
        tile_overlap=args.tile_overlap,
        polarity_mode=args.polarity_mode,
        max_events_per_tile=args.max_events_per_tile,
    )
    grid = preprocessor.tile_grid()
    print(
        f"Preprocessor: source={source_resolution} "
        f"target=({preprocessor.tgt_W},{preprocessor.tgt_H}) "
        f"t_window={preprocessor.target_t_us / 1e6:.2f}s "
        f"tiles={len(grid)} overlap={args.tile_overlap}"
    )

    x, y, t, p = load_stream(input_path)
    print(f"Loaded {len(x)} events from {input_path}")

    gt_bboxes_by_window: dict[tuple[int, int], list[list[int]]] = {}
    if args.fred_annotations:
        anns = load_fred_annotations(args.fred_annotations)
        print(f"Loaded {len(anns)} ground-truth annotations")
        t_arr = np.asarray(t, dtype=np.int64)
        if len(t_arr):
            t0_global = int(t_arr.min())
            t_span = int(t_arr.max()) - t0_global
            n_windows = max(1, t_span // preprocessor.target_t_us + 1)
            for w in range(n_windows):
                w_start = w * preprocessor.target_t_us
                w_end = w_start + preprocessor.target_t_us
                slc = annotations_in_window(
                    anns, t0_global + w_start, t0_global + w_end,
                )
                if slc:
                    gt_bboxes_by_window[(w_start, w_end)] = [
                        list(a.bbox) for a in slc
                    ]

    detections = detector.detect_stream(
        x, y, t, p, preprocessor, nms_iou=args.nms_iou,
    )
    print(f"\nDetections: {len(detections)} total across all time windows")
    for i, d in enumerate(detections):
        t0, t1 = d["t_window_us"]
        print(
            f"  {i}: bbox={d['bbox']} score={d['score']:.3f} "
            f"events={d['num_events']} t=[{t0/1e6:.2f}s, {t1/1e6:.2f}s]"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(detections, fh, indent=2)
        print(f"Saved detections to {out_path}")

    if args.visualize or args.video is not None:
        _emit_frames(
            x, y, t, p,
            detections=detections,
            preprocessor=preprocessor,
            gt_bboxes_by_window=gt_bboxes_by_window,
            source_resolution=source_resolution,
            args=args,
        )

    return detections


def _emit_frames(
    x, y, t, p, *,
    detections: list[dict],
    preprocessor: EventPreprocessor,
    gt_bboxes_by_window: dict[tuple[int, int], list[list[int]]],
    source_resolution: tuple[int, int],
    args: argparse.Namespace,
) -> None:
    """Stream per-window frames to disk / video using ``iter_window_frames``."""
    writer = None
    if args.video:
        import cv2
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        W, H = source_resolution
        writer = cv2.VideoWriter(
            str(video_path), fourcc, float(args.video_fps), (int(W), int(H)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer at {video_path}")
        print(f"Writing video to {video_path} @ {args.video_fps} fps")

    vis_dir = Path(args.vis_dir) if args.visualize else None
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    bbox_color = BBOX_COLORS[args.bbox_color]
    input_stem = Path(args.input).stem

    try:
        for w_idx, _, frame in iter_window_frames(
            x, y, t, p,
            detections=detections,
            preprocessor=preprocessor,
            gt_bboxes_by_window=gt_bboxes_by_window,
            resolution=source_resolution,
            bbox_color=bbox_color,
        ):
            if vis_dir is not None:
                out_file = vis_dir / f"{input_stem}_w{w_idx:04d}.png"
                _write_png(frame, out_file)
                print(f"  Saved {out_file}")
            if writer is not None:
                writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
            print(f"Finished video: {args.video}")


def _write_png(frame: np.ndarray, path: Path) -> None:
    try:
        import cv2
        cv2.imwrite(str(path), frame)
    except ImportError:
        import matplotlib.pyplot as plt
        plt.imsave(str(path), frame)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
