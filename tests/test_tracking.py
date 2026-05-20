"""Tests for sliding-window tracking.

These tests cover the association layer and the SlidingWindowTracker
state machine in isolation — no SPGNet / spconv dependency, since the
tracker's `update()` consumes plain detection dicts.
"""

from __future__ import annotations

import numpy as np
import pytest

from ev_drone_detector.tracking.association import iou_matrix, match_by_iou
from ev_drone_detector.tracking.tracker import SlidingWindowTracker, Track


# ---------------------------------------------------------------- association


def test_iou_matrix_identity():
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
    m = iou_matrix(boxes, boxes)
    assert m.shape == (2, 2)
    np.testing.assert_allclose(np.diag(m), [1.0, 1.0])
    assert m[0, 1] == 0.0


def test_iou_matrix_partial_overlap():
    m = iou_matrix([[0, 0, 10, 10]], [[5, 5, 15, 15]])
    # intersection = 5*5 = 25, union = 100+100-25 = 175
    np.testing.assert_allclose(m[0, 0], 25.0 / 175.0)


def test_iou_matrix_empty():
    assert iou_matrix([], []).shape == (0, 0)
    assert iou_matrix([[0, 0, 1, 1]], []).shape == (1, 0)
    assert iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)


def test_match_by_iou_clean_assignment():
    tracks = [[0, 0, 10, 10], [100, 100, 120, 120]]
    dets = [[101, 101, 121, 121], [1, 1, 11, 11]]
    matches, ut, ud = match_by_iou(tracks, dets, iou_threshold=0.3)
    # Hungarian should pair track 0 <-> det 1 and track 1 <-> det 0
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert ut == [] and ud == []


def test_match_by_iou_threshold_rejects_low_iou():
    tracks = [[0, 0, 10, 10]]
    dets = [[100, 100, 110, 110]]
    matches, ut, ud = match_by_iou(tracks, dets, iou_threshold=0.3)
    assert matches == []
    assert ut == [0]
    assert ud == [0]


def test_match_by_iou_empty_inputs():
    matches, ut, ud = match_by_iou([], [[0, 0, 10, 10]], iou_threshold=0.3)
    assert matches == [] and ut == [] and ud == [0]

    matches, ut, ud = match_by_iou([[0, 0, 10, 10]], [], iou_threshold=0.3)
    assert matches == [] and ut == [0] and ud == []


# -------------------------------------------------------------------- tracker


def _det(bbox, score=0.9):
    return {"bbox": list(bbox), "score": float(score), "num_events": 10,
            "center": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]}


class _StubSensor:
    whole_t = 8000


class _StubCfg:
    sensor = _StubSensor()


class _StubDetector:
    """Minimal stand-in so SlidingWindowTracker can be constructed in tests."""

    config = _StubCfg()


def _tracker(**kw) -> SlidingWindowTracker:
    return SlidingWindowTracker(_StubDetector(), **kw)


def test_new_detection_gets_track_id():
    tr = _tracker()
    active = tr.update([_det([10, 10, 30, 30])])
    assert len(active) == 1
    assert active[0]["track_id"] == 1
    assert active[0]["hits"] == 1


def test_same_drone_keeps_id_across_windows():
    tr = _tracker(iou_threshold=0.3)
    tr.update([_det([10, 10, 30, 30])])
    tr.update([_det([12, 12, 32, 32])])
    active = tr.update([_det([14, 14, 34, 34])])
    assert len(active) == 1
    assert active[0]["track_id"] == 1
    assert active[0]["hits"] == 3


def test_separate_drones_get_distinct_ids():
    tr = _tracker(iou_threshold=0.3)
    active = tr.update([_det([10, 10, 30, 30]), _det([200, 200, 220, 220])])
    ids = sorted(t["track_id"] for t in active)
    assert ids == [1, 2]

    # Both move slightly — IDs must stay stable
    active2 = tr.update([_det([12, 12, 32, 32]), _det([202, 202, 222, 222])])
    ids2 = sorted(t["track_id"] for t in active2)
    assert ids2 == [1, 2]


def test_crossing_tracks_swap_via_hungarian():
    """Two tracks moving toward each other — Hungarian must pick the
    globally-optimal assignment instead of greedy first-match."""
    tr = _tracker(iou_threshold=0.1)
    tr.update([_det([0, 0, 20, 20]), _det([100, 0, 120, 20])])
    # Detections swap relative order; Hungarian should still match correctly.
    active = tr.update([_det([100, 0, 120, 20]), _det([0, 0, 20, 20])])
    by_bbox = {tuple(t["bbox"]): t["track_id"] for t in active}
    assert by_bbox[(0, 0, 20, 20)] == 1
    assert by_bbox[(100, 0, 120, 20)] == 2


def test_unmatched_track_ages_then_deletes():
    tr = _tracker(iou_threshold=0.3, max_age=2)
    tr.update([_det([10, 10, 30, 30])])           # window 0: hit
    tr.update([])                                  # window 1: age=1
    tr.update([])                                  # window 2: age=2
    # Still alive at age == max_age
    assert any(t.track_id == 1 for t in tr.tracks)
    tr.update([])                                  # window 3: age=3 > max_age → gone
    assert all(t.track_id != 1 for t in tr.tracks)


def test_min_hits_suppresses_one_window_flicker():
    tr = _tracker(min_hits=2)
    active = tr.update([_det([10, 10, 30, 30])])
    assert active == []
    active = tr.update([_det([11, 11, 31, 31])])
    assert len(active) == 1
    assert active[0]["track_id"] == 1


def test_track_history_records_per_window_state():
    tr = _tracker()
    tr.update([_det([10, 10, 30, 30], score=0.7)])
    tr.update([_det([11, 11, 31, 31], score=0.8)])
    hist = tr.tracks[0].history
    assert [h["window"] for h in hist] == [0, 1]
    assert hist[1]["score"] == pytest.approx(0.8)


def test_reset_clears_state():
    tr = _tracker()
    tr.update([_det([10, 10, 30, 30])])
    tr.update([_det([12, 12, 32, 32])])
    tr.reset()
    assert tr.tracks == []
    assert tr._window_idx == 0
    # Fresh track gets id 1 again
    active = tr.update([_det([10, 10, 30, 30])])
    assert active[0]["track_id"] == 1


def test_invalid_stride_raises():
    tr = _tracker()
    with pytest.raises(ValueError):
        tr.track_npz_sequence(["a.npz"], stride=0)


# -------------------------------------------------- track_event_stream slicing


class _SliceDetector:
    """Detector stub that records per-window event coverage and returns
    a single bbox at the median event position — exercises slicing logic
    without needing spconv / a model."""

    def __init__(self):
        self.calls: list[dict] = []
        self.config = _StubCfg()

    def detect(self, features, coords):
        f = features.numpy() if hasattr(features, "numpy") else np.asarray(features)
        c = coords.numpy() if hasattr(coords, "numpy") else np.asarray(coords)
        self.calls.append({
            "n_events": int(c.shape[0]),
            "t_feat_max": float(f[:, 2].max()) if f.shape[0] else 0.0,
            "t_coord_max": int(c[:, 2].max()) if c.shape[0] else 0,
            "xy_mean": (float(c[:, 0].mean()), float(c[:, 1].mean())) if c.shape[0] else (0.0, 0.0),
        })
        if c.shape[0] == 0:
            return []
        cx = int(np.median(c[:, 0]))
        cy = int(np.median(c[:, 1]))
        return [{"bbox": [cx - 5, cy - 5, cx + 5, cy + 5],
                 "score": 0.9, "num_events": int(c.shape[0]),
                 "center": [cx, cy]}]


def test_stream_mode_slices_windows_and_renormalizes_time():
    det = _SliceDetector()
    tracker = SlidingWindowTracker(det)

    # 16 seconds of events at constant position — 2 non-overlapping windows
    n = 800
    times_us = np.linspace(0, 16_000_000 - 1, n).astype(np.int64)
    coords = np.zeros((n, 3), dtype=np.int64)
    coords[:, 0] = 100
    coords[:, 1] = 50
    coords[:, 2] = times_us  # placeholder, will be re-normalized
    features = np.zeros((n, 4), dtype=np.float32)

    results = tracker.track_event_stream(
        features, coords, times_us,
        window_us=8_000_000, stride_us=8_000_000,
    )
    assert len(results) == 2
    # Each window must see ~half the events
    for r in results:
        assert 350 < r["num_events"] < 450
    # t_norm must be normalized into [0, 1]
    for call in det.calls:
        assert 0.0 <= call["t_feat_max"] <= 1.0
        # whole_t = 8000 → voxel t in [0, 7999]
        assert call["t_coord_max"] <= 7999
    # Same drone — should stay track 1 across both windows
    win1_ids = [t["track_id"] for t in results[1]["active_tracks"]]
    assert win1_ids == [1]


def test_stream_mode_overlapping_stride_produces_more_windows():
    det = _SliceDetector()
    tracker = SlidingWindowTracker(det)

    n = 1000
    times_us = np.linspace(0, 16_000_000 - 1, n).astype(np.int64)
    coords = np.zeros((n, 3), dtype=np.int64)
    coords[:, 0] = 100
    coords[:, 1] = 50
    features = np.zeros((n, 4), dtype=np.float32)

    no_overlap = tracker.track_event_stream(
        features, coords, times_us,
        window_us=8_000_000, stride_us=8_000_000,
    )
    overlap = tracker.track_event_stream(
        features, coords, times_us,
        window_us=8_000_000, stride_us=4_000_000,
    )
    assert len(overlap) > len(no_overlap)
