"""Tests for drone detection pipeline."""

import pytest
import numpy as np
from ev_drone_detector.detection.clustering import cluster_events_to_bbox
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


def test_extrapolate_moving_cluster_to_window_end():
    """Linear motion fit recovers a drone's position at target_t.

    A drone moves from (100, 100) at t=0 to (180, 100) at t=8000 (no noise).
    Extrapolating to t=7999 must put the bbox center near (180, 100), not
    at the trail centroid (~140, 100).
    """
    rng = np.random.RandomState(0)
    n = 200
    t = np.linspace(0, 8000, n, dtype=np.float64)
    x = 100.0 + (80.0 / 8000.0) * t + rng.normal(scale=0.5, size=n)
    y = 100.0 + rng.normal(scale=0.5, size=n)
    events_xy = np.stack([x, y], axis=1)

    detections = cluster_events_to_bbox(
        events_xy,
        events_t=t,
        extrapolate_to_t=7999.0,
        eps=10.0, min_samples=3, min_cluster_size=5,
        bbox_padding=0,
    )
    assert len(detections) == 1
    det = detections[0]
    cx, cy = det["center"]
    assert abs(cx - 180) < 5, f"extrapolated cx={cx}, expected ~180"
    assert abs(cy - 100) < 5
    # Velocity should be ~ (80/8000, 0) = (0.01, 0)
    vx, vy = det["velocity"]
    assert abs(vx - 0.01) < 0.002
    assert abs(vy) < 0.01


def test_extrapolate_static_cluster_falls_back_to_centroid():
    """A cluster at one timestamp has no temporal spread → trail centroid."""
    rng = np.random.RandomState(0)
    events_xy = rng.normal(loc=[50, 50], scale=2, size=(40, 2))
    events_t = np.full(40, 1000.0)

    detections = cluster_events_to_bbox(
        events_xy,
        events_t=events_t,
        extrapolate_to_t=7999.0,
        eps=10.0, min_samples=3, min_cluster_size=5,
    )
    assert len(detections) == 1
    cx, cy = detections[0]["center"]
    assert abs(cx - 50) < 5
    assert abs(cy - 50) < 5
    # Zero velocity in the degenerate-fit branch
    assert detections[0]["velocity"] == [0.0, 0.0]


def test_extrapolation_changes_bbox_vs_trail():
    """Same cluster: trail bbox vs extrapolated bbox should differ in center."""
    rng = np.random.RandomState(1)
    n = 200
    t = np.linspace(0, 8000, n, dtype=np.float64)
    x = 50.0 + 0.02 * t + rng.normal(scale=0.3, size=n)
    y = 50.0 + rng.normal(scale=0.3, size=n)
    events_xy = np.stack([x, y], axis=1)

    trail = cluster_events_to_bbox(events_xy, eps=10.0, min_samples=3, min_cluster_size=5)
    extra = cluster_events_to_bbox(
        events_xy, events_t=t, extrapolate_to_t=7999.0,
        eps=10.0, min_samples=3, min_cluster_size=5,
    )
    assert len(trail) == 1 and len(extra) == 1
    # Extrapolated center must be further right than the trail centroid.
    assert extra[0]["center"][0] > trail[0]["center"][0] + 20
    # "velocity" key is present only on the extrapolated path.
    assert "velocity" not in trail[0]
    assert "velocity" in extra[0]
