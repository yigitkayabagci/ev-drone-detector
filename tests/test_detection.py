"""Tests for drone detection pipeline."""

import pytest
import numpy as np
from ev_drone_detector.detection.clustering import (
    cluster_events_to_bbox,
    segmentation_to_detections,
)
from ev_drone_detector.utils.eval import _bbox_iou, compute_detection_metrics, compute_iou, compute_accuracy


def test_cluster_events_empty():
    """Test clustering with no events."""
    result = cluster_events_to_bbox(np.array([]).reshape(0, 2))
    assert result == []


def test_cluster_events_single_cluster():
    """Test clustering with a tight cluster of events."""
    rng = np.random.RandomState(42)
    n_events = 50
    events_xy = rng.normal(loc=[100, 100], scale=3, size=(n_events, 2))

    detections = cluster_events_to_bbox(
        events_xy, eps=10.0, min_samples=3, min_cluster_size=5,
        bbox_padding=5, image_size=(346, 260),
    )

    assert len(detections) >= 1
    det = detections[0]
    assert "bbox" in det
    assert "score" in det
    assert "center" in det

    cx, cy = det["center"]
    assert abs(cx - 100) < 20
    assert abs(cy - 100) < 20


def test_cluster_events_two_clusters():
    """Test clustering with two separated clusters."""
    rng = np.random.RandomState(42)

    c1 = rng.normal(loc=[50, 50], scale=3, size=(30, 2))
    c2 = rng.normal(loc=[200, 200], scale=3, size=(30, 2))

    events_xy = np.vstack([c1, c2])
    detections = cluster_events_to_bbox(
        events_xy, eps=10.0, min_samples=3, min_cluster_size=5
    )

    assert len(detections) == 2


def test_cluster_with_scores():
    """Test clustering with confidence scores."""
    rng = np.random.RandomState(42)
    events_xy = rng.normal(loc=[100, 100], scale=3, size=(50, 2))
    scores = rng.uniform(0.7, 1.0, size=50)

    detections = cluster_events_to_bbox(
        events_xy, scores=scores, eps=10.0, min_samples=3
    )

    assert len(detections) >= 1
    assert detections[0]["score"] > 0.5


def test_bbox_iou():
    """Test bounding box IoU computation."""
    assert _bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    iou = _bbox_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert 0.0 < iou < 1.0


def test_detection_metrics():
    """Test detection precision/recall computation."""
    metrics = compute_detection_metrics(
        pred_bboxes=[[0, 0, 10, 10]],
        gt_bboxes=[[0, 0, 10, 10]],
    )
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0

    metrics = compute_detection_metrics(
        pred_bboxes=[],
        gt_bboxes=[[0, 0, 10, 10]],
    )
    assert metrics["recall"] == 0.0

    metrics = compute_detection_metrics(
        pred_bboxes=[[0, 0, 10, 10]],
        gt_bboxes=[],
    )
    assert metrics["precision"] == 0.0


def test_eval_metrics():
    """Test segmentation evaluation metrics."""
    pred = np.array([1.0, 1.0, 0.0, 0.0])
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    assert compute_iou(pred, labels) == 1.0
    assert compute_accuracy(pred, labels) == 1.0

    pred = np.array([1.0, 1.0, 0.0, 0.0])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert compute_iou(pred, labels) == 0.0
    assert compute_accuracy(pred, labels) == 0.0


def test_max_detections_limit():
    """Test that max_detections parameter is respected."""
    rng = np.random.RandomState(42)

    clusters = []
    for i in range(5):
        cx, cy = 50 + i * 60, 50 + i * 40
        clusters.append(rng.normal(loc=[cx, cy], scale=3, size=(20, 2)))
    events_xy = np.vstack(clusters)

    detections = cluster_events_to_bbox(
        events_xy, eps=10.0, min_samples=3, min_cluster_size=5,
        max_detections=3,
    )

    assert len(detections) <= 3


def test_segmentation_to_detections_cpu():
    """segmentation_to_detections must work when predictions/p2v_map devices differ.

    Here everything is on CPU, but the implementation moves predictions to CPU
    *before* indexing, which is what makes the CUDA path (predictions on GPU,
    p2v_map on CPU) safe too.
    """
    import torch

    rng = np.random.RandomState(0)
    n = 40
    xy = rng.normal(loc=[100, 100], scale=2, size=(n, 2)).astype(np.int64)
    t = np.zeros((n, 1), dtype=np.int64)
    coords = torch.from_numpy(np.concatenate([xy, t], axis=1))  # (N, 3) = x, y, t
    p2v_map = torch.arange(n)               # each event in its own voxel
    predictions = torch.ones(n, 1)          # all positive

    dets = segmentation_to_detections(
        predictions, coords, p2v_map,
        threshold=0.5, eps=10.0, min_samples=3, min_cluster_size=5,
        image_size=(346, 260),
    )
    assert len(dets) >= 1
    assert "bbox" in dets[0] and "score" in dets[0]
    cx, cy = dets[0]["center"]
    assert abs(cx - 100) < 25 and abs(cy - 100) < 25


def test_segmentation_to_detections_all_negative():
    """No event above threshold -> empty detection list, no DBSCAN crash."""
    import torch

    coords = torch.zeros(10, 3, dtype=torch.int64)
    p2v_map = torch.arange(10)
    predictions = torch.zeros(10, 1)        # all below threshold
    dets = segmentation_to_detections(predictions, coords, p2v_map, threshold=0.5)
    assert dets == []


def test_events_to_frame_paper_scheme():
    """White bg, dark-gray background events, red events inside a bbox (BGR)."""
    from ev_drone_detector.utils.viz import events_to_frame

    xs = np.array([10, 100])  # first inside bbox, second outside
    ys = np.array([10, 100])
    frame = events_to_frame(xs, ys, resolution=(200, 200), bboxes=[[5, 5, 20, 20]])

    assert frame.shape == (200, 200, 3) and frame.dtype == np.uint8
    assert tuple(frame[0, 0]) == (255, 255, 255)      # background white
    assert tuple(frame[10, 10]) == (0, 0, 255)        # in-bbox event -> red (BGR)
    assert tuple(frame[100, 100]) == (70, 70, 70)     # outside event -> dark gray


def test_events_to_frame_no_bboxes():
    """With no detections, all events are dark gray (no red)."""
    from ev_drone_detector.utils.viz import events_to_frame

    frame = events_to_frame(np.array([5]), np.array([5]), resolution=(50, 50))
    assert tuple(frame[5, 5]) == (70, 70, 70)
    assert not (frame == (0, 0, 255)).all(axis=2).any()  # no red anywhere


def test_bbox_clamping():
    """Test that bboxes are clamped to image bounds."""
    events_xy = np.array([[0, 0], [1, 1], [2, 0], [0, 2], [1, 0]], dtype=float)

    detections = cluster_events_to_bbox(
        events_xy, eps=5.0, min_samples=2, min_cluster_size=3,
        bbox_padding=10, image_size=(346, 260),
    )

    if detections:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            assert x1 >= 0
            assert y1 >= 0
            assert x2 <= 345
            assert y2 <= 259
