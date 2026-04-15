"""Tests for SPGNet model and components.

These tests verify tensor shapes and basic forward passes.
Full model tests require CUDA + spconv, so they're skipped on CPU-only systems.
"""

import pytest
import torch

try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
except ImportError:
    HAS_SPCONV = False

needs_spconv = pytest.mark.skipif(not HAS_SPCONV, reason="spconv not installed")


@needs_spconv
def test_se_module():
    """Test SEModule forward pass."""
    from ev_drone_detector.models.blocks import SEModule

    se = SEModule(channels=16, reduction=2)
    features = torch.randn(10, 16)
    coords = torch.randint(0, 20, (10, 4), dtype=torch.int32)
    coords[:, 0] = 0
    st = spconv.SparseConvTensor(features, coords, [20, 20, 20], batch_size=1)

    out = se(st)
    assert out.features.shape == (10, 16)


@needs_spconv
def test_gdconv():
    """Test GDConv: splits channels evenly, each group temp→temp."""
    from ev_drone_detector.models.blocks import GDConv

    # 12 channels / 4 dilations = 3 per group
    gdconv = GDConv(out_channels=12, dilations=[1, 2, 3, 4])
    features = torch.randn(20, 12)
    coords = torch.randint(0, 30, (20, 4), dtype=torch.int32)
    coords[:, 0] = 0
    st = spconv.SparseConvTensor(features, coords, [30, 30, 30], batch_size=1)

    out = gdconv(st)
    assert out.features.shape == (20, 12)


@needs_spconv
def test_gdblock():
    """Test GDBlock with ad_channels."""
    from ev_drone_detector.models.blocks import GDBlock

    # ad_channels=16 → mid = 12+16=28, must be divisible by 4 → 28/4=7 ✓
    block = GDBlock(in_channels=12, out_channels=12, dilations=[1, 2, 3, 4], ad_channels=16)
    features = torch.randn(20, 12)
    coords = torch.randint(0, 30, (20, 4), dtype=torch.int32)
    coords[:, 0] = 0
    st = spconv.SparseConvTensor(features, coords, [30, 30, 30], batch_size=1)

    out = block(st)
    assert out.features.shape[1] == 12  # Output should be out_channels


@needs_spconv
def test_gdblock_ad_channels_zero():
    """Test GDBlock with ad_channels=0 (stage 4 config)."""
    from ev_drone_detector.models.blocks import GDBlock

    block = GDBlock(in_channels=48, out_channels=48, dilations=[1, 2, 3, 4], ad_channels=0)
    features = torch.randn(15, 48)
    coords = torch.randint(0, 25, (15, 4), dtype=torch.int32)
    coords[:, 0] = 0
    st = spconv.SparseConvTensor(features, coords, [25, 25, 25], batch_size=1)

    out = block(st)
    assert out.features.shape == (15, 48)


@needs_spconv
def test_sparse_basic_block():
    """Test SparseBasicBlock residual connection."""
    from ev_drone_detector.models.blocks import SparseBasicBlock

    block = SparseBasicBlock(inplanes=16, planes=16, indice_key="test_res")
    features = torch.randn(15, 16)
    coords = torch.randint(0, 25, (15, 4), dtype=torch.int32)
    coords[:, 0] = 0
    st = spconv.SparseConvTensor(features, coords, [25, 25, 25], batch_size=1)

    out = block(st)
    assert out.features.shape == (15, 16)


@needs_spconv
def test_spgnet_forward_small():
    """Test SPGNet forward pass with a small spatial shape.

    Spatial shape must be divisible by stride cascade:
    - Spatial: divisible by 2^3 = 8
    - Temporal: divisible by 4^3 = 64
    - First dim / 2^num_pools must equal 11 for PatchAttention
    """
    from ev_drone_detector.models.spgnet import SPGNet

    # Use spatial shape that works with PatchAttention target (11,9,8)
    # 88 = 11 * 8 (3 levels of /2 + 4 pools)
    # Actually for small test, let's use: [88, 72, 512]
    # Stage 2: [44, 36, 128], pa2: log2(44/11)=2 pools
    # Stage 3: [22, 18, 32], pa3: log2(22/11)=1 pool
    # Stage 4: [11, 9, 8], pa4: log2(11/11)=0 pools (no pooling)
    spatial_shape = [88, 72, 512]
    model = SPGNet(
        input_channel=4,
        width=12,
        spatial_shape=spatial_shape,
        dilations=[1, 2, 3, 4],
    )
    model.eval()

    n_voxels = 100
    features = torch.randn(n_voxels, 4)
    coords = torch.zeros(n_voxels, 4, dtype=torch.int32)
    coords[:, 0] = 0  # batch idx
    coords[:, 1] = torch.randint(0, spatial_shape[0], (n_voxels,))
    coords[:, 2] = torch.randint(0, spatial_shape[1], (n_voxels,))
    coords[:, 3] = torch.randint(0, spatial_shape[2], (n_voxels,))

    voxel_tensor = spconv.SparseConvTensor(
        features, coords, spatial_shape, batch_size=1
    )

    with torch.no_grad():
        predictions, voxel_output = model(voxel_tensor)

    assert predictions.shape[1] == 1
    assert (predictions >= 0).all() and (predictions <= 1).all()
    assert predictions.shape[0] == voxel_output.features.shape[0]


@needs_spconv
def test_spgnet_param_count():
    """Test that model parameter count is reasonable (~4M for width=12)."""
    from ev_drone_detector.models.spgnet import SPGNet

    model = SPGNet(
        input_channel=4,
        width=12,
        spatial_shape=[88, 72, 512],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 100_000 < n_params < 50_000_000
