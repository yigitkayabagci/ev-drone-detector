"""Tests for track-aware visualization helpers.

Avoid asserting exact pixel layouts (OpenCV / matplotlib may not be present
on every CI machine); instead check geometric invariants — distinct colors
per ID, ring-buffer bounds, frame shape, etc.
"""

from __future__ import annotations

import numpy as np

from ev_drone_detector.utils.viz import (
    TrailBuffer,
    color_for_track,
    draw_tracks,
    events_to_frame,
)


def test_color_for_track_distinct_and_cycles():
    # First 8 IDs hit each palette slot exactly once.
    colors = [color_for_track(i) for i in range(1, 9)]
    assert len(set(colors)) == 8
    # IDs 9 and 17 wrap back to ID 1's color.
    assert color_for_track(9) == color_for_track(1)
    assert color_for_track(17) == color_for_track(1)


def test_trail_buffer_respects_max_length():
    buf = TrailBuffer(max_length=3)
    for i in range(5):
        buf.update([{
            "track_id": 1,
            "bbox": [i, i, i + 10, i + 10],
            "score": 0.9,
        }])
    # Only the last 3 centers survive.
    assert len(buf.trails[1]) == 3
    # Newest is at the end and matches the last update.
    last = buf.trails[1][-1]
    assert last == (4 + 5, 4 + 5)  # bbox [4,4,14,14] → center (9, 9)


def test_trail_buffer_independent_per_id():
    buf = TrailBuffer(max_length=10)
    buf.update([
        {"track_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"track_id": 2, "bbox": [100, 100, 110, 110], "score": 0.9},
    ])
    buf.update([
        {"track_id": 1, "bbox": [2, 2, 12, 12], "score": 0.9},
    ])
    assert len(buf.trails[1]) == 2
    assert len(buf.trails[2]) == 1


def test_draw_tracks_returns_same_shape_and_modifies_image():
    frame = np.full((260, 346, 3), 128, dtype=np.uint8)
    tracks = [
        {"track_id": 1, "bbox": [50, 50, 80, 80], "score": 0.9},
        {"track_id": 2, "bbox": [200, 100, 230, 130], "score": 0.7},
    ]
    out = draw_tracks(frame, tracks)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    # Some pixels must have changed (the bbox edges).
    assert not np.array_equal(out, frame)


def test_draw_tracks_distinct_id_colors_on_frame():
    """Verify the two bboxes actually paint with different (non-gray) colors."""
    frame = np.full((260, 346, 3), 128, dtype=np.uint8)
    tracks = [
        {"track_id": 1, "bbox": [50, 50, 80, 80], "score": 0.9},
        {"track_id": 2, "bbox": [200, 100, 230, 130], "score": 0.7},
    ]
    out = draw_tracks(frame, tracks, thickness=2)

    # Sample a pixel inside the top edge of each bbox; should be that
    # track's palette color, not the gray background.
    p1 = tuple(int(v) for v in out[50, 60])
    p2 = tuple(int(v) for v in out[100, 215])
    assert p1 != (128, 128, 128)
    assert p2 != (128, 128, 128)
    assert p1 != p2


def test_draw_tracks_empty_active_returns_unchanged():
    frame = np.full((260, 346, 3), 128, dtype=np.uint8)
    out = draw_tracks(frame, [])
    assert np.array_equal(out, frame)


def test_draw_tracks_only_renders_trails_for_active_ids():
    """Stale (non-active) IDs in the trail buffer must not be drawn."""
    frame = np.full((260, 346, 3), 128, dtype=np.uint8)
    trails = {
        1: [(10.0, 10.0), (20.0, 10.0), (30.0, 10.0)],  # active
        99: [(10.0, 200.0), (20.0, 200.0)],              # stale — must NOT draw
    }
    active = [{"track_id": 1, "bbox": [25, 5, 35, 15], "score": 0.9}]

    out = draw_tracks(frame, active, trails=trails)

    # Along y=200 (where the stale trail lived) the frame should still be gray.
    band = out[200, 5:25]
    assert np.all(band == 128), "stale trail must not be painted"


def test_events_to_frame_basic_shape():
    n = 20
    x = np.arange(n) * 5
    y = np.full(n, 50)
    pol = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    frame = events_to_frame(x.astype(float), y.astype(float), pol, resolution=(346, 260))
    assert frame.shape == (260, 346, 3)
    assert frame.dtype == np.uint8
