"""Tests for the stream-level visualization / inference helpers.

These tests do NOT load the SPGNet model — they exercise the pure
preprocessing + rendering portions of the pipeline that have to work
correctly for the end-to-end script to be useful. Model-level inference
is covered by tests/test_data.py / test_model.py and is skipped here
when spconv is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ev_drone_detector.data.preprocessing import EventPreprocessor
from ev_drone_detector.detection.stream import (
    BBOX_COLORS,
    iter_window_frames,
    load_npz_stream,
    load_stream,
    render_window_frame,
)


def _save_npz(path: Path, x, y, t, p):
    np.savez(str(path), x=x, y=y, t=t, p=p)


def test_load_npz_stream_roundtrip(tmp_path):
    x = np.arange(50, dtype=np.int64)
    y = np.arange(50, dtype=np.int64) + 100
    t = np.arange(50, dtype=np.int64) * 1000
    p = np.tile([0, 1], 25).astype(np.int64)
    path = tmp_path / "stream.npz"
    _save_npz(path, x, y, t, p)

    rx, ry, rt, rp = load_npz_stream(path)
    assert np.array_equal(rx, x)
    assert np.array_equal(ry, y)
    assert np.array_equal(rt, t)
    assert np.array_equal(rp, p)


def test_load_npz_stream_missing_key_raises(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(str(path), x=np.zeros(3), y=np.zeros(3), t=np.zeros(3))
    with pytest.raises(KeyError, match="missing required keys"):
        load_npz_stream(path)


def test_load_stream_autodetects_npz(tmp_path):
    path = tmp_path / "s.npz"
    _save_npz(path, np.array([1]), np.array([2]), np.array([0]), np.array([1]))
    x, y, t, p = load_stream(path)
    assert x.tolist() == [1] and y.tolist() == [2]


def test_load_stream_autodetects_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "s.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.array([10, 20], dtype=np.uint16))
        f.create_dataset("y", data=np.array([30, 40], dtype=np.uint16))
        f.create_dataset("t", data=np.array([1000, 2000], dtype=np.int64))
        f.create_dataset("p", data=np.array([0, 1], dtype=np.uint8))
    x, y, t, p = load_stream(path)
    assert x.tolist() == [10, 20]
    assert p.tolist() == [0, 1]


def test_load_stream_rejects_unknown_extension(tmp_path):
    path = tmp_path / "s.unknown"
    path.write_text("")
    with pytest.raises(ValueError, match="Unsupported input extension"):
        load_stream(path)


def test_render_window_frame_shape_and_dtype():
    x = np.array([100, 200, 300], dtype=np.int64)
    y = np.array([50, 150, 250], dtype=np.int64)
    p = np.array([1, 0, 1], dtype=np.int64)
    t = np.array([0, 1_000_000, 2_000_000], dtype=np.int64)
    detections = [{"bbox": [80, 30, 220, 170], "score": 0.9}]
    frame = render_window_frame(
        x, y, p, t,
        t_window_us=(0, 8_000_000),
        pred_detections=detections,
        gt_bboxes=[],
        resolution=(1280, 720),
    )
    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.uint8


def test_render_window_frame_filters_by_time():
    """Events outside the requested window must not appear in the canvas."""
    x = np.array([100, 200], dtype=np.int64)
    y = np.array([100, 200], dtype=np.int64)
    p = np.array([1, 1], dtype=np.int64)
    t = np.array([1_000_000, 10_000_000], dtype=np.int64)
    frame = render_window_frame(
        x, y, p, t,
        t_window_us=(0, 8_000_000),
        pred_detections=[],
        gt_bboxes=[],
        resolution=(1280, 720),
    )
    # Event 0 (t=1e6) is inside window -> non-gray pixel at (100, 100)
    # Event 1 (t=1e7) is outside window -> pixel at (200, 200) should be the gray bg
    bg = np.array([128, 128, 128], dtype=np.uint8)
    assert not np.array_equal(frame[100, 100], bg)
    assert np.array_equal(frame[200, 200], bg)


def test_render_window_frame_draws_pred_bbox():
    """The predicted bbox color must actually appear on the frame."""
    frame = render_window_frame(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        t_window_us=(0, 8_000_000),
        pred_detections=[{"bbox": [10, 10, 60, 60], "score": 1.0}],
        gt_bboxes=[],
        resolution=(200, 200),
        bbox_color=BBOX_COLORS["red"],
    )
    # The top edge of the bbox at y=10 between x=10 and x=60 should be the
    # bbox color (or the rectangle's outline color via cv2/manual fallback).
    edge_row = frame[10, 10:60, :]
    has_red = np.any(np.all(edge_row == np.array(BBOX_COLORS["red"]), axis=-1))
    assert has_red


def test_render_window_frame_draws_gt_bbox():
    frame = render_window_frame(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        t_window_us=(0, 8_000_000),
        pred_detections=[],
        gt_bboxes=[[20, 20, 80, 80]],
        resolution=(200, 200),
        gt_color=(255, 255, 255),
    )
    edge_row = frame[20, 20:80, :]
    has_white = np.any(np.all(edge_row == np.array([255, 255, 255]), axis=-1))
    assert has_white


def test_iter_window_frames_yields_one_per_window():
    """16 seconds of events with 8s windows should yield 2 frames."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.integers(0, 1280, size=n).astype(np.int64)
    y = rng.integers(0, 720, size=n).astype(np.int64)
    t = rng.integers(0, 16_000_000, size=n).astype(np.int64)
    p = rng.integers(0, 2, size=n).astype(np.int64)

    pp = EventPreprocessor(source_resolution=(1280, 720), target_t_us=8_000_000)
    frames = list(iter_window_frames(
        x, y, t, p, detections=[], preprocessor=pp, resolution=(1280, 720),
    ))
    assert len(frames) == 2
    indices = [w for w, _, _ in frames]
    assert indices == [0, 1]
    for _, twin, frame in frames:
        assert frame.shape == (720, 1280, 3)
        assert twin[1] - twin[0] == 8_000_000


def test_iter_window_frames_routes_detections_to_correct_window():
    """A detection in window 1 must only be drawn on the window-1 frame."""
    x = np.array([100, 100], dtype=np.int64)
    y = np.array([100, 100], dtype=np.int64)
    p = np.array([1, 1], dtype=np.int64)
    t = np.array([1_000_000, 10_000_000], dtype=np.int64)

    pp = EventPreprocessor(source_resolution=(200, 200), target_t_us=8_000_000)
    det_w0 = {
        "bbox": [50, 50, 150, 150], "score": 0.9, "num_events": 1,
        "center": [100, 100], "t_window_us": (0, 8_000_000),
    }
    det_w1 = {
        "bbox": [70, 70, 130, 130], "score": 0.7, "num_events": 1,
        "center": [100, 100], "t_window_us": (8_000_000, 16_000_000),
    }
    frames = list(iter_window_frames(
        x, y, t, p, detections=[det_w0, det_w1], preprocessor=pp,
        resolution=(200, 200), bbox_color=BBOX_COLORS["red"],
    ))
    assert len(frames) == 2
    # First frame: red box at y=50, between x=50..150
    f0 = frames[0][2]
    has_red_w0 = np.any(np.all(
        f0[50, 50:150, :] == np.array(BBOX_COLORS["red"]), axis=-1
    ))
    # Second frame: red box at y=70, between x=70..130
    f1 = frames[1][2]
    has_red_w1 = np.any(np.all(
        f1[70, 70:130, :] == np.array(BBOX_COLORS["red"]), axis=-1
    ))
    assert has_red_w0 and has_red_w1
    # No red at y=70 on first frame (the w1 detection should not leak)
    no_red_w0_at_w1 = not np.any(np.all(
        f0[70, 70:130, :] == np.array(BBOX_COLORS["red"]), axis=-1
    ))
    assert no_red_w0_at_w1


def test_iter_window_frames_empty_stream_yields_nothing():
    pp = EventPreprocessor(source_resolution=(1280, 720))
    out = list(iter_window_frames(
        np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        np.array([], dtype=np.int64), np.array([], dtype=np.int64),
        detections=[], preprocessor=pp, resolution=(1280, 720),
    ))
    assert out == []


def test_iter_window_frames_gt_overlay_per_window():
    """GT bboxes must be routed to their window like predictions."""
    rng = np.random.default_rng(0)
    n = 100
    x = rng.integers(0, 200, size=n).astype(np.int64)
    y = rng.integers(0, 200, size=n).astype(np.int64)
    t = rng.integers(0, 16_000_000, size=n).astype(np.int64)
    p = rng.integers(0, 2, size=n).astype(np.int64)
    pp = EventPreprocessor(source_resolution=(200, 200), target_t_us=8_000_000)

    gt_by_window = {
        (0, 8_000_000): [[10, 10, 50, 50]],
        (8_000_000, 16_000_000): [[60, 60, 100, 100]],
    }
    frames = {
        w: frame for w, _, frame in iter_window_frames(
            x, y, t, p, detections=[], preprocessor=pp,
            gt_bboxes_by_window=gt_by_window, resolution=(200, 200),
            gt_color=(255, 255, 255),
        )
    }
    assert set(frames.keys()) == {0, 1}
    # Frame 0: should have white pixels at y=10
    has_white_f0 = np.any(np.all(
        frames[0][10, 10:50, :] == np.array([255, 255, 255]), axis=-1
    ))
    has_white_f1 = np.any(np.all(
        frames[1][60, 60:100, :] == np.array([255, 255, 255]), axis=-1
    ))
    assert has_white_f0 and has_white_f1
