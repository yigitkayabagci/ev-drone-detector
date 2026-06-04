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


def _ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    """VOC all-points Average Precision: area under the precision envelope."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # Make precision monotonically decreasing (envelope) from the right.
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _ap_at_iou(all_preds: list, gts_per_image: list, iou_thr: float, npos: int) -> float | None:
    """AP at one IoU threshold. all_preds: list of (img_idx, score, box) sorted by score desc."""
    if npos == 0:
        return None
    matched = [[False] * len(g) for g in gts_per_image]
    n = len(all_preds)
    tp = np.zeros(n)
    fp = np.zeros(n)
    for i, (img, _score, box) in enumerate(all_preds):
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gts_per_image[img]):
            if matched[img][j]:
                continue
            iou = _bbox_iou(box, gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            tp[i] = 1
            matched[img][best_j] = True
        else:
            fp[i] = 1
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / npos
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return _ap_from_pr(recall, precision)


def _pr_at_iou(preds_per_image: list, gts_per_image: list, iou_thr: float = 0.5) -> tuple[float, float]:
    """Global precision/recall at one IoU threshold (greedy, score-ordered matching)."""
    tp = fp = fn = 0
    for p, gts in zip(preds_per_image, gts_per_image):
        boxes = p["boxes"]
        scores = p["scores"]
        matched = [False] * len(gts)
        order = list(np.argsort(-np.asarray(scores))) if len(scores) else []
        for k in order:
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gts):
                if matched[j]:
                    continue
                iou = _bbox_iou(boxes[k], gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_thr:
                tp += 1
                matched[best_j] = True
            else:
                fp += 1
        fn += matched.count(False)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def compute_map(
    preds_per_image: list[dict],
    gts_per_image: list[list],
    iou_thresholds: list[float] | None = None,
) -> dict:
    """COCO-style mAP for single-class (drone) detection.

    Args:
        preds_per_image: one dict per image with keys
            "boxes": list of [x_min, y_min, x_max, y_max]
            "scores": list of confidence scores (same length)
        gts_per_image: one list per image of ground-truth [x_min,y_min,x_max,y_max].
        iou_thresholds: IoU thresholds for the mAP sweep (default 0.5:0.05:0.95).

    Returns:
        dict with map_50, map_50_95, precision, recall (P/R at IoU 0.5),
        ap_per_iou, and num_gt.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95

    all_preds = []
    for img, p in enumerate(preds_per_image):
        for b, s in zip(p["boxes"], p["scores"]):
            all_preds.append((img, float(s), b))
    all_preds.sort(key=lambda t: t[1], reverse=True)

    npos = sum(len(g) for g in gts_per_image)

    ap_per_iou = {thr: _ap_at_iou(all_preds, gts_per_image, thr, npos) for thr in iou_thresholds}
    valid_aps = [v for v in ap_per_iou.values() if v is not None]
    map_50 = ap_per_iou.get(0.5)
    map_50_95 = float(np.mean(valid_aps)) if valid_aps else 0.0

    precision, recall = _pr_at_iou(preds_per_image, gts_per_image, 0.5)

    return {
        "map_50": float(map_50) if map_50 is not None else 0.0,
        "map_50_95": map_50_95,
        "precision": precision,
        "recall": recall,
        "ap_per_iou": {str(k): (None if v is None else float(v)) for k, v in ap_per_iou.items()},
        "num_gt": int(npos),
    }
