"""Tests for the generic event preprocessor (EventPreprocessor)."""

from __future__ import annotations

import numpy as np
import pytest

from ev_drone_detector.data.preprocessing import EventPreprocessor, TileWindow


def _make_events(
    n: int = 100, W: int = 346, H: int = 260, t_us: int = 8_000_000,
    polarity_01: bool = False, seed: int = 0,
):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, W, size=n).astype(np.int64)
    y = rng.integers(0, H, size=n).astype(np.int64)
    t = rng.integers(0, t_us, size=n).astype(np.int64)
    if polarity_01:
        p = rng.integers(0, 2, size=n).astype(np.int64)
    else:
        p = rng.choice([-1, 1], size=n).astype(np.int64)
    return x, y, t, p


def test_preprocessor_basic_passthrough():
    """source = target resolution -> single tile, all events present."""
    x, y, t, p = _make_events(n=500, W=346, H=260)
    pp = EventPreprocessor(source_resolution=(346, 260))
    tws = list(pp(x, y, t, p))
    assert len(tws) == 1
    tw = tws[0]
    assert isinstance(tw, TileWindow)
    assert tw.tile_origin == (0, 0)
    assert tw.features.shape == (500, 4)
    assert tw.coords.shape == (500, 3)
    assert tw.coords.dtype == np.int64
    assert tw.features.dtype == np.float32


def test_polarity_auto_remaps_01():
    """{0,1} polarity must be remapped to {-1,+1}."""
    x, y, t, p = _make_events(n=200, polarity_01=True)
    pp = EventPreprocessor(source_resolution=(346, 260))
    feats = np.concatenate([tw.features for tw in pp(x, y, t, p)], axis=0)
    assert set(np.unique(feats[:, 3]).tolist()) == {-1.0, 1.0}


def test_polarity_auto_keeps_pm1():
    """{-1,+1} polarity must pass through unchanged."""
    x, y, t, p = _make_events(n=200, polarity_01=False)
    pp = EventPreprocessor(source_resolution=(346, 260))
    feats = np.concatenate([tw.features for tw in pp(x, y, t, p)], axis=0)
    assert set(np.unique(feats[:, 3]).tolist()) == {-1.0, 1.0}


def test_polarity_explicit_01():
    """polarity_mode='01' should treat input as {0,1} unconditionally."""
    p = np.array([0, 1, 0, 1])
    x = np.array([10, 20, 30, 40])
    y = np.array([10, 20, 30, 40])
    t = np.array([0, 1000, 2000, 3000], dtype=np.int64)
    pp = EventPreprocessor(source_resolution=(346, 260), polarity_mode="01")
    feats = np.concatenate([tw.features for tw in pp(x, y, t, p)], axis=0)
    assert feats[:, 3].tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_tile_grid_covers_source():
    """Every (x, y) in source must fall inside at least one tile."""
    # Corners + center of a 1280x720 source
    x = np.array([0, 1279, 0, 1279, 640], dtype=np.int64)
    y = np.array([0, 0, 719, 719, 360], dtype=np.int64)
    t = np.array([0, 1000, 2000, 3000, 4000], dtype=np.int64)
    p = np.array([1, 1, -1, -1, 1])
    pp = EventPreprocessor(source_resolution=(1280, 720))
    tws = list(pp(x, y, t, p))
    total_events = sum(tw.features.shape[0] for tw in tws)
    assert total_events >= 5


def test_tile_grid_shape_when_source_larger():
    """1280x720 with 346x260 target should produce a multi-tile grid."""
    pp = EventPreprocessor(source_resolution=(1280, 720))
    grid = pp.tile_grid()
    assert len(grid) > 1
    xs = sorted(set(g[0] for g in grid))
    ys = sorted(set(g[1] for g in grid))
    # Last tile origin must let the tile reach the source edge.
    assert xs[-1] + pp.tgt_W >= pp.src_W
    assert ys[-1] + pp.tgt_H >= pp.src_H


def test_tile_grid_single_tile_when_smaller_or_equal():
    """If source <= target, just one tile at (0, 0)."""
    pp_eq = EventPreprocessor(source_resolution=(346, 260))
    assert pp_eq.tile_grid() == [(0, 0)]
    pp_lt = EventPreprocessor(source_resolution=(240, 180))
    assert pp_lt.tile_grid() == [(0, 0)]


def test_tile_overlap_creates_more_tiles():
    """Higher overlap should yield strictly more tiles for the same source."""
    n0 = len(EventPreprocessor(source_resolution=(1280, 720), tile_overlap=0.0).tile_grid())
    n5 = len(EventPreprocessor(source_resolution=(1280, 720), tile_overlap=0.5).tile_grid())
    assert n5 > n0


def test_temporal_windowing_splits_long_stream():
    """A 16-second stream with 8-second windows should yield 2 distinct windows."""
    x, y, t, p = _make_events(n=500, t_us=16_000_000)
    pp = EventPreprocessor(source_resolution=(346, 260), target_t_us=8_000_000)
    tws = list(pp(x, y, t, p))
    windows = {tw.t_window_us for tw in tws}
    assert len(windows) == 2
    assert (0, 8_000_000) in windows
    assert (8_000_000, 16_000_000) in windows


def test_empty_input_yields_nothing():
    pp = EventPreprocessor(source_resolution=(346, 260))
    out = list(pp(np.array([]), np.array([]), np.array([], dtype=np.int64), np.array([])))
    assert out == []


def test_mismatched_lengths_raise():
    pp = EventPreprocessor(source_resolution=(346, 260))
    with pytest.raises(ValueError):
        list(pp(np.array([1, 2]), np.array([1]), np.array([1, 2], dtype=np.int64), np.array([1, 1])))


def test_features_normalized_in_unit_range():
    """x_norm, y_norm, t_norm must lie in [0, 1]."""
    x, y, t, p = _make_events(n=1000, W=346, H=260, t_us=8_000_000)
    pp = EventPreprocessor(source_resolution=(346, 260))
    feats = np.concatenate([tw.features for tw in pp(x, y, t, p)], axis=0)
    assert feats[:, 0].min() >= 0.0 and feats[:, 0].max() <= 1.0
    assert feats[:, 1].min() >= 0.0 and feats[:, 1].max() <= 1.0
    assert feats[:, 2].min() >= 0.0 and feats[:, 2].max() <= 1.0


def test_t_voxel_within_target_bins():
    """t voxel indices must lie in [0, target_t_bins - 1]."""
    x, y, t, p = _make_events(n=1000)
    pp = EventPreprocessor(source_resolution=(346, 260), target_t_bins=8192)
    coords = np.concatenate([tw.coords for tw in pp(x, y, t, p)], axis=0)
    assert coords[:, 2].min() >= 0
    assert coords[:, 2].max() <= 8191


def test_max_events_per_tile_cap():
    x, y, t, p = _make_events(n=10000, W=346, H=260)
    pp = EventPreprocessor(source_resolution=(346, 260), max_events_per_tile=500)
    for tw in pp(x, y, t, p):
        assert tw.features.shape[0] <= 500


def test_tile_local_coords():
    """coords[:, 0] and coords[:, 1] must be within [0, tgt_W) and [0, tgt_H)."""
    x, y, t, p = _make_events(n=2000, W=1280, H=720)
    pp = EventPreprocessor(source_resolution=(1280, 720))
    for tw in pp(x, y, t, p):
        assert tw.coords[:, 0].min() >= 0
        assert tw.coords[:, 0].max() < pp.tgt_W
        assert tw.coords[:, 1].min() >= 0
        assert tw.coords[:, 1].max() < pp.tgt_H


def test_from_config_matches_default_yaml():
    """from_config should pick up resolution + t window from the YAML config."""
    from ev_drone_detector.utils.config import load_config

    cfg = load_config("configs/default.yaml")
    pp = EventPreprocessor.from_config(cfg, source_resolution=(1280, 720))
    assert pp.tgt_W == 346
    assert pp.tgt_H == 260
    assert pp.target_t_us == 8_000_000
    assert pp.target_t_bins == 8192
