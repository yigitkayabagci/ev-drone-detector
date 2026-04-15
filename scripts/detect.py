"""Inference script for drone detection.

Usage:
    uv run python scripts/detect.py --checkpoint checkpoints/best_iou.pt --input data/test/sample.npz
    uv run python scripts/detect.py --checkpoint checkpoints/best_iou.pt --input data/test/ --visualize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ev_drone_detector.detection.detector import DroneDetector
from ev_drone_detector.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect drones in event data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to .npz file or directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--visualize", action="store_true", help="Save visualization images")
    parser.add_argument("--vis_dir", type=str, default="visualizations")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup detector
    import torch
    device = None
    if args.device:
        device = torch.device(args.device)

    detector = DroneDetector.from_config(args.config, device=device)
    detector.load_weights(args.checkpoint)
    print(f"Loaded model from {args.checkpoint}")

    # Collect input files
    input_path = Path(args.input)
    if input_path.is_file():
        npz_files = [input_path]
    else:
        npz_files = sorted(input_path.glob("*.npz"))
        if not npz_files:
            npz_files = sorted(input_path.rglob("*.npz"))

    print(f"Processing {len(npz_files)} files...")

    all_results = {}

    for npz_path in npz_files:
        print(f"\nProcessing: {npz_path.name}")
        detections = detector.detect_from_npz(npz_path)

        print(f"  Found {len(detections)} drone(s)")
        for i, det in enumerate(detections):
            bbox = det["bbox"]
            print(
                f"  Detection {i}: bbox={bbox}, "
                f"score={det['score']:.3f}, "
                f"events={det['num_events']}"
            )

        all_results[str(npz_path)] = detections

        # Visualization
        if args.visualize:
            _visualize_detection(npz_path, detections, args.vis_dir)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def _visualize_detection(
    npz_path: Path, detections: list[dict], vis_dir: str
) -> None:
    """Save a visualization of the detection results."""
    from ev_drone_detector.utils.viz import draw_detections, events_to_frame

    data = np.load(str(npz_path), allow_pickle=True)
    evs = data["evs_norm"]
    coords = data["ev_loc"]

    frame = events_to_frame(
        coords[:, 0].astype(float),
        coords[:, 1].astype(float),
        polarity=evs[:, 3],
    )
    frame = draw_detections(frame, detections)

    vis_path = Path(vis_dir)
    vis_path.mkdir(parents=True, exist_ok=True)
    out_file = vis_path / f"{npz_path.stem}_detection.png"

    try:
        import cv2
        cv2.imwrite(str(out_file), frame)
    except ImportError:
        import matplotlib.pyplot as plt
        plt.imsave(str(out_file), frame)

    print(f"  Saved visualization: {out_file}")


if __name__ == "__main__":
    main()
