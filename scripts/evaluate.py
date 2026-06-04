"""Evaluate a trained SPGNet detector on the test set.

Computes detection metrics: mAP@50, mAP@50-95, Precision, Recall.

Ground-truth boxes are derived from the per-event drone labels in each .npz
(evs_norm[:, 4] == 1) by clustering them exactly like the detector clusters its
positive predictions — so GT and predicted boxes live in the same space and the
IoU comparison is fair.

Usage:
    uv run python scripts/evaluate.py --checkpoint checkpoints/best_iou.pt
    uv run python scripts/evaluate.py --checkpoint checkpoints/last.pt --device cuda:0
"""

from __future__ import annotations

import os

# spconv/cumm: use prebuilt kernels, skip JIT (bites on bleeding-edge Colab).
os.environ.setdefault("CUMM_DISABLE_JIT", "1")
os.environ.setdefault("SPCONV_DISABLE_JIT", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ev_drone_detector.detection.clustering import cluster_events_to_bbox
from ev_drone_detector.detection.detector import DroneDetector
from ev_drone_detector.utils.config import load_config
from ev_drone_detector.utils.eval import compute_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SPGNet detector on the test set")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-dir", type=str, default=None, help="Override data.test_dir")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Write metrics JSON here")
    parser.add_argument("--gt-min-cluster-size", type=int, default=None,
                        help="Min events for a GT box (default: detection.min_cluster_size)")
    return parser.parse_args()


def gt_boxes_from_labels(ev_loc, labels, det_cfg, image_size, gt_min_cluster_size):
    """Cluster the labelled drone events (label==1) into ground-truth boxes."""
    mask = labels >= 0.5
    if not mask.any():
        return []
    xy = ev_loc[mask, :2].astype(float)
    gt = cluster_events_to_bbox(
        xy,
        scores=None,
        eps=det_cfg.cluster_eps,
        min_samples=det_cfg.cluster_min_samples,
        min_cluster_size=gt_min_cluster_size,
        bbox_padding=det_cfg.bbox_padding,
        max_detections=100,
        image_size=image_size,
    )
    return [d["bbox"] for d in gt]


def main() -> None:
    args = parse_args()
    import torch

    cfg = load_config(args.config)
    if args.test_dir is not None:
        cfg.data.test_dir = args.test_dir
    gt_min = args.gt_min_cluster_size or cfg.detection.min_cluster_size
    image_size = tuple(cfg.sensor.resolution)

    device = torch.device(args.device) if args.device else None
    detector = DroneDetector.from_config(args.config, device=device)
    detector.load_weights(args.checkpoint)
    print(f"Loaded {args.checkpoint} on {detector.device}")

    test_dir = Path(cfg.data.test_dir)
    files = sorted(test_dir.glob("*.npz")) or sorted(test_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files under {test_dir.resolve()}")
    print(f"Evaluating on {len(files)} test files...")

    preds_per_image: list[dict] = []
    gts_per_image: list[list] = []

    for i, npz_path in enumerate(files, 1):
        data = np.load(str(npz_path), allow_pickle=True)
        evs = data["evs_norm"]
        ev_loc = data["ev_loc"]
        if evs.shape[1] < 5:
            raise ValueError(
                f"{npz_path.name} has no label column (evs_norm has {evs.shape[1]} cols); "
                "cannot build ground-truth boxes."
            )
        labels = evs[:, 4].astype(np.float32)

        gts_per_image.append(
            gt_boxes_from_labels(ev_loc, labels, cfg.detection, image_size, gt_min)
        )
        dets = detector.detect_from_npz(npz_path)
        preds_per_image.append({
            "boxes": [d["bbox"] for d in dets],
            "scores": [d["score"] for d in dets],
        })
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    m = compute_map(preds_per_image, gts_per_image)

    n_pred = sum(len(p["boxes"]) for p in preds_per_image)
    print("\n" + "=" * 44)
    print(f"TEST SET  ({len(files)} images, {m['num_gt']} GT boxes, {n_pred} predictions)")
    print(f"Model      : SPGNet  ({Path(args.checkpoint).name})")
    print("-" * 44)
    print(f"mAP@50     : {m['map_50']:.4f}")
    print(f"mAP@50-95  : {m['map_50_95']:.4f}")
    print(f"Precision  : {m['precision']:.4f}")
    print(f"Recall     : {m['recall']:.4f}")
    print("=" * 44)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(m, f, indent=2)
        print(f"Metrics saved to {out}")


if __name__ == "__main__":
    main()
