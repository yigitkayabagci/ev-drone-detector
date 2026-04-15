"""Evaluation metrics for event-based drone detection.

Includes:
- IoU (Intersection over Union) for segmentation
- Segmentation accuracy
- Detection metrics (precision, recall) for bounding boxes
"""

from __future__ import annotations

import numpy as np
import torch


def compute_iou(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute IoU for binary segmentation.

    Args:
        predictions: (N,) predicted probabilities.
        labels: (N,) binary ground truth.
        threshold: Binarization threshold.

    Returns:
        Mean IoU across both classes.
    """
    pred_binary = (predictions >= threshold).astype(int)
    labels_int = labels.astype(int)

    ious = []
    for cls in [0, 1]:
        pred_mask = pred_binary == cls
        label_mask = labels_int == cls
        intersection = (pred_mask & label_mask).sum()
        union = (pred_mask | label_mask).sum()
        if union > 0:
            ious.append(intersection / union)
        else:
            ious.append(1.0)

    return float(np.mean(ious))


def compute_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute segmentation accuracy."""
    pred_binary = (predictions >= threshold).astype(int)
    labels_int = labels.astype(int)
    return float((pred_binary == labels_int).mean())


def compute_detection_metrics(
    pred_bboxes: list[list[int]],
    gt_bboxes: list[list[int]],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute precision and recall for bounding box detections.

    Args:
        pred_bboxes: List of predicted [x_min, y_min, x_max, y_max].
        gt_bboxes: List of ground truth [x_min, y_min, x_max, y_max].
        iou_threshold: IoU threshold for matching.

    Returns:
        Dict with 'precision', 'recall', 'f1'.
    """
    if not gt_bboxes:
        return {
            "precision": 1.0 if not pred_bboxes else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not pred_bboxes else 0.0,
        }
    if not pred_bboxes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_gt = set()
    tp = 0

    for pred in pred_bboxes:
        best_iou = 0.0
        best_gt_idx = -1

        for i, gt in enumerate(gt_bboxes):
            if i in matched_gt:
                continue
            iou = _bbox_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    precision = tp / len(pred_bboxes) if pred_bboxes else 0.0
    recall = tp / len(gt_bboxes) if gt_bboxes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def _bbox_iou(box1: list[int], box2: list[int]) -> float:
    """Compute IoU between two bounding boxes [x_min, y_min, x_max, y_max]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0
