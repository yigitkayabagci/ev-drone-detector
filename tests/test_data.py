"""Tests for event data representation and voxelization."""

import pytest
import numpy as np
import torch

try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False

needs_spconv = pytest.mark.skipif(not HAS_SPCONV, reason="spconv not installed")


@needs_spconv
def test_voxelize_basic():
    """Test basic voxelization with simple events."""
    from ev_drone_detector.data.event_repr import voxelize_events

    # 5 events, 4 features each
    features = torch.tensor([
        [0.1, 0.2, 0.0, 1.0],
        [0.1, 0.2, 0.0, -1.0],  # Same voxel as event 0
        [0.5, 0.5, 0.5, 1.0],
        [0.9, 0.9, 1.0, -1.0],
        [0.9, 0.9, 1.0, 1.0],  # Same voxel as event 3
    ], dtype=torch.float32)

    coords = torch.tensor([
        [10, 20, 100],
        [10, 20, 100],  # Same voxel
        [50, 50, 500],
        [90, 90, 1000],
        [90, 90, 1000],  # Same voxel
    ], dtype=torch.int64)

    spatial_shape = [100, 100, 2000]
    voxel_tensor, p2v_map = voxelize_events(features, coords, 0, spatial_shape)

    # Should have 3 unique voxels
    assert voxel_tensor.features.shape[0] == 3
    assert voxel_tensor.features.shape[1] == 4

    # p2v_map should map each event to its voxel
    assert p2v_map.shape[0] == 5
    assert p2v_map[0] == p2v_map[1]  # Same voxel
    assert p2v_map[3] == p2v_map[4]  # Same voxel


@needs_spconv
def test_voxelize_single_event():
    """Test voxelization with a single event."""
    from ev_drone_detector.data.event_repr import voxelize_events

    features = torch.tensor([[0.5, 0.5, 0.5, 1.0]], dtype=torch.float32)
    coords = torch.tensor([[50, 50, 500]], dtype=torch.int64)

    voxel_tensor, p2v_map = voxelize_events(features, coords, 0, [100, 100, 1000])

    assert voxel_tensor.features.shape[0] == 1
    assert p2v_map.shape[0] == 1
    assert torch.allclose(voxel_tensor.features[0], features[0])


@needs_spconv
def test_voxelize_averaging():
    """Test that features are averaged within same voxel."""
    from ev_drone_detector.data.event_repr import voxelize_events

    features = torch.tensor([
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ], dtype=torch.float32)
    coords = torch.tensor([[10, 10, 10], [10, 10, 10]], dtype=torch.int64)

    voxel_tensor, _ = voxelize_events(features, coords, 0, [100, 100, 100])

    assert voxel_tensor.features.shape[0] == 1
    expected = torch.tensor([0.5, 0.5, 0.5, 1.0])
    assert torch.allclose(voxel_tensor.features[0], expected)


@needs_spconv
def test_collate_events():
    """Test batch collation."""
    from ev_drone_detector.data.event_repr import collate_events

    batch = [
        {
            "features": torch.randn(100, 4),
            "coords": torch.randint(0, 50, (100, 3)),
            "labels": torch.zeros(100),
        },
        {
            "features": torch.randn(80, 4),
            "coords": torch.randint(0, 50, (80, 3)),
            "labels": torch.ones(80),
        },
    ]

    result = collate_events(batch, spatial_shape=[100, 100, 100])

    assert "voxel_tensor" in result
    assert "labels" in result
    assert "p2v_map" in result
    assert result["labels"].shape[0] == 180
    assert result["p2v_map"].shape[0] == 180


def test_synthetic_dataset():
    """Test synthetic dataset generation."""
    from ev_drone_detector.data.dataset import SyntheticEventDataset

    ds = SyntheticEventDataset(num_samples=5, num_events=1000)
    assert len(ds) == 5

    sample = ds[0]
    assert sample["features"].shape == (1000, 4)
    assert sample["coords"].shape == (1000, 3)
    assert sample["labels"].shape == (1000,)
    assert sample["labels"].sum() > 0  # Should have some drone events


def test_synthetic_dataset_reproducible():
    """Test that synthetic dataset is deterministic."""
    from ev_drone_detector.data.dataset import SyntheticEventDataset

    ds1 = SyntheticEventDataset(num_samples=3, num_events=100)
    ds2 = SyntheticEventDataset(num_samples=3, num_events=100)

    s1 = ds1[0]
    s2 = ds2[0]
    assert torch.allclose(s1["features"], s2["features"])
    assert torch.allclose(s1["labels"], s2["labels"])
