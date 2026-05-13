"""Run drone detection on a raw event stream.

This is the canonical entry point for real-life event data from any source —
Prophesee EVK4 .hdf5 / .raw, DAVIS .aedat, FRED HDF5 sequences, a live
event camera, anything that gives you four parallel arrays:

    x  (N,) int   pixel x in source resolution
    y  (N,) int   pixel y in source resolution
    t  (N,) int64 timestamps in microseconds
    p  (N,) int   polarity ({0, 1} or {-1, +1} — auto-detected)

This script ships with one generic loader (a raw .npz with keys
``x``, ``y``, ``t``, ``p``). For sources that need a special reader
(FRED HDF5, Prophesee .raw, etc.) write the 5–10 lines of loader code that
fills those four arrays, then pass them to ``DroneDetector.detect_stream``
exactly as this script does.

Usage:
    uv run python scripts/detect_stream.py \\
        --checkpoint checkpoints/best_iou.pt \\
        --input my_recording.npz \\
        --source-resolution 1280 720 \\
        --tile-overlap 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ev_drone_detector.data.preprocessing import EventPreprocessor
from ev_drone_detector.detection.detector import DroneDetector
from ev_drone_detector.utils.config import load_config


def load_npz_stream(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read (x, y, t, p) arrays from a generic raw-event .npz file.

    Required keys: ``x``, ``y``, ``t``, ``p``. Reuse this function as a
    template when writing readers for other formats.
    """
    data = np.load(str(path), allow_pickle=True)
    for k in ("x", "y", "t", "p"):
        if k not in data:
            raise KeyError(
                f"{path}: required key {k!r} not found "
                f"(available: {list(data.files)})"
            )
    return data["x"], data["y"], data["t"], data["p"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect drones in a raw event stream")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--input", type=str, required=True,
        help="Path to .npz containing x, y, t, p arrays",
    )
    p.add_argument(
        "--source-resolution", nargs=2, type=int, required=True,
        metavar=("W", "H"),
        help="Source sensor resolution (e.g. 1280 720 for Prophesee EVK4)",
    )
    p.add_argument(
        "--tile-overlap", type=float, default=0.0,
        help="Fraction overlap between adjacent tiles (0..0.9)",
    )
    p.add_argument(
        "--max-events-per-tile", type=int, default=None,
        help="Cap events per tile (random subsample). None = no cap.",
    )
    p.add_argument(
        "--polarity-mode", choices=["auto", "01", "pm1"], default="auto",
    )
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument("--output", type=str, default=None, help="Output JSON path")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    device = torch.device(args.device) if args.device else None

    cfg = load_config(args.config)
    detector = DroneDetector.from_config(args.config, device=device)
    detector.load_weights(args.checkpoint)
    print(f"Loaded model from {args.checkpoint}")

    preprocessor = EventPreprocessor.from_config(
        cfg,
        source_resolution=tuple(args.source_resolution),
        tile_overlap=args.tile_overlap,
        polarity_mode=args.polarity_mode,
        max_events_per_tile=args.max_events_per_tile,
    )
    grid = preprocessor.tile_grid()
    print(
        f"Preprocessor: source={tuple(args.source_resolution)} "
        f"target=({preprocessor.tgt_W},{preprocessor.tgt_H}) "
        f"t_window={preprocessor.target_t_us/1e6:.2f}s "
        f"tiles={len(grid)} overlap={args.tile_overlap}"
    )

    x, y, t, p = load_npz_stream(Path(args.input))
    print(f"Loaded {len(x)} events from {args.input}")

    detections = detector.detect_stream(
        x, y, t, p, preprocessor, nms_iou=args.nms_iou,
    )

    print(f"\nFound {len(detections)} detection(s) in source coords:")
    for i, d in enumerate(detections):
        bbox = d["bbox"]
        t0, t1 = d["t_window_us"]
        print(
            f"  {i}: bbox={bbox} score={d['score']:.3f} "
            f"events={d['num_events']} "
            f"t=[{t0/1e6:.2f}s, {t1/1e6:.2f}s]"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(detections, f, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
