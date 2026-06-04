"""Inference script for drone detection.

Usage:
    uv run python scripts/detect.py --checkpoint checkpoints/best_iou.pt --input data/test/sample.npz
    uv run python scripts/detect.py --checkpoint checkpoints/best_iou.pt --input data/test/ --visualize
    uv run python scripts/detect.py --checkpoint checkpoints/best_iou.pt --input data/test/ --video out.mp4 --fps 25
"""

from __future__ import annotations

import os

# spconv/cumm: force the prebuilt CUDA kernels and skip the JIT fallback that
# breaks on bleeding-edge Colab runtimes. MUST be set before spconv is imported.
os.environ.setdefault("CUMM_DISABLE_JIT", "1")
os.environ.setdefault("SPCONV_DISABLE_JIT", "1")

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
    parser.add_argument("--visualize", action="store_true", help="Save per-frame PNG visualizations")
    parser.add_argument("--vis_dir", type=str, default="visualizations")
    parser.add_argument("--video", type=str, default=None,
                        help="Write a video of all frames to this path (e.g. out.mp4)")
    parser.add_argument("--fps", type=float, default=25.0, help="Frames per second for --video")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _build_frame(
    npz_path: Path,
    detections: list[dict],
    resolution: tuple[int, int] = (346, 260),
) -> np.ndarray:
    """Render an event frame with bounding boxes drawn on top."""
    from ev_drone_detector.utils.viz import draw_detections, events_to_frame

    data = np.load(str(npz_path), allow_pickle=True)
    evs = data["evs_norm"]
    coords = data["ev_loc"]

    frame = events_to_frame(
        coords[:, 0].astype(float),
        coords[:, 1].astype(float),
        polarity=evs[:, 3],
        resolution=resolution,
    )
    return draw_detections(frame, detections)


def main() -> None:
    args = parse_args()

    # Setup detector
    import torch
    device = None
    if args.device:
        device = torch.device(args.device)

    cfg = load_config(args.config)
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

    # Optional video writer (requires OpenCV)
    writer = None
    if args.video:
        import cv2
        W, H = cfg.sensor.resolution
        video_path = Path(args.video)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (int(W), int(H)))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")
        print(f"Writing video to {video_path} @ {args.fps} fps")

    all_results = {}

    try:
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

            if args.visualize or writer is not None:
                frame = _build_frame(
                    npz_path, detections, resolution=tuple(cfg.sensor.resolution)
                )

                if args.visualize:
                    vis_path = Path(args.vis_dir)
                    vis_path.mkdir(parents=True, exist_ok=True)
                    out_file = vis_path / f"{npz_path.stem}_detection.png"
                    try:
                        import cv2
                        cv2.imwrite(str(out_file), frame)
                    except ImportError:
                        import matplotlib.pyplot as plt
                        plt.imsave(str(out_file), frame)
                    print(f"  Saved visualization: {out_file}")

                if writer is not None:
                    writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
            print(f"Finished video: {args.video}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
