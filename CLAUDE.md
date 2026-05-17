# EV-Drone-Detector

## Think Hard

When working on this project, ALWAYS think deeply and carefully before writing code. This means:

1. **Understand the architecture first**: SPGNet (EV-SpSegNet) is a U-shaped sparse 3D encoder-decoder that operates on voxelized event point clouds. It segments events into target (drone) vs background/noise. Before modifying any model code, trace the full data flow from raw events → voxelization → sparse tensor → encoder → decoder → per-event prediction → clustering → bounding box.

2. **Verify tensor shapes at every step**: Event data is inherently sparse and 3D (x, y, t). The voxelization maps N events with features (x, y, t, polarity) and coordinates (x, y, t) into sparse voxels. Track shapes through the entire pipeline.

3. **Test every change**: After any code modification, run the test suite. Write a minimal test case for new functionality before or immediately after implementation.

4. **Think about edge cases**: Event camera data is asynchronous and sparse. Consider: empty voxels, single-event voxels, very large event counts (>700K), zero-drone scenarios, multi-drone scenarios.

5. **GPU memory awareness**: Sparse convolutions are memory-efficient but voxelization can create large intermediate tensors. Always consider batch size=1 as the default for large event streams.

## Project Goal

This project adapts SPGNet (EV-SpSegNet) from the EV-UAV benchmark for **initial drone detection** — finding the first bounding box of a drone in event camera data. This is NOT a tracker; it produces an initial detection that will be fed to a downstream tracking algorithm.

**Pipeline**: Raw events → Voxelization → SPGNet segmentation → Event clustering → Bounding box output

## Primary Sources

### Paper
- **Title**: "Event-based Tiny Object Detection: A Benchmark Dataset and Baseline"
- **Authors**: Nuo Chen, Chao Xiao, Yimian Dai, Shiman He, Miao Li, Wei An
- **ArXiv**: 2506.23575v1 (June 2025)
- **Key contributions**:
  - EV-UAV dataset: 147 sequences, 2.3M event-level annotations, avg target size 6.8×5.4 pixels
  - EV-SpSegNet: U-shaped sparse 3D encoder-decoder with GDSCA modules
  - STC Loss: Spatiotemporal Correlation loss leveraging motion continuity
  - DAVIS346 camera: 346×260 resolution, up to 10^6 Hz

### Repository
- **GitHub**: https://github.com/ChenYichen9527/Ev-UAV
- **Key files in original repo**:
  - `model/evspsegnet.py` — Main model architecture
  - `model/basemodel.py` — GDBlock, GDConv, SEModule, Downsample_block
  - `dataset/basedataset.py` — Voxelization and collation
  - `dataset/ev_uav.py` — Dataset loader (.npz format)
  - `utils/stcloss.py` — STC loss implementation
  - `configs/evisseg_evuav.yaml` — Training configuration

## Architecture Details (EV-SpSegNet / SPGNet)

IMPORTANT: These details are verified against the ACTUAL SOURCE CODE at
https://github.com/ChenYichen9527/Ev-UAV — not just the paper.

### Input
- Events as (x, y, t, polarity) normalized features
- Voxel coordinates as (x, y, t) integers
- Sparse 3D tensor via spconv.SparseConvTensor
- spatial_shape MUST be padded: [352, 288, 8192] (original uses [11*32, 9*32, 256*32])

### Encoder (4 stages)
- Stage 1: SubMConv3d(4→12) + GDBlock(12→12, ad_channels=16)
- Stage 2: SparseConv3d(kernel=3, stride=[2,2,4]) (12→24) + GDBlock(ad_channels=16) + PatchAttention
- Stage 3: SparseConv3d(kernel=3, stride=[2,2,4]) (24→48) + GDBlock(ad_channels=8) + PatchAttention
- Stage 4: SparseConv3d(kernel=3, stride=[2,2,4]) (48→48) + GDBlock(ad_channels=0) + PatchAttention

### Decoder (4 UR blocks)
Each UR block does:
1. SparseBasicBlock on lateral/skip features
2. Concat bottom + transformed lateral
3. GDBlock fusion
4. channel_reduction: reshape(N, C_out, C_in/C_out) → sum → reduces channels (NO learned params)
5. Residual add: fused + reduced
6. InverseConv3d upsampling

CRITICAL decoder details:
- First UR block is SELF-REFERENTIAL: uses x_conv4 as BOTH lateral AND bottom
- Last UR block uses a GDBlock instead of InverseConv (already at full resolution)
- Decoder GDBlocks all use ad_channels=16

### Output
- Linear(12→1) + Sigmoid → per-event drone probability
- Our addition: cluster positive events → fit bounding box

### GDSCA Module (verified against basemodel.py)
- Shortcut: SubMConv3d(in, out, 1) + BN (additive skip, NOT concat)
- Main: pw_conv(in, out+ad) → GDConv(out+ad, dilation=[1,2,3,4]) → SE → pw_conv(out+ad, out)
- GDConv: splits channels EVENLY, each group has temp=channels//4, conv is temp→temp
- Fusion: ADDITIVE (main + shortcut) → ReLU
- ad_channels adds extra width to main path only

### PatchAttention (verified against evspsegnet.py)
- Target spatial: (11, 9, 8) — hardcoded in original
- num_pools = log2(spatial_dim[0] / 11)
- Cascaded MaxPool3d(2,2,4) → MultiheadAttention(1 head) → InverseConv3d cascade
- Has final SubMConv3d(C, C, 1) AFTER residual

### STC Loss (verified against stcloss.py)
- Fixed SubMConv3d(1, 1, [k,k,tau]) with all-ones weights (frozen)
- w_stc = sigmoid(neighbor_sum - mean), then DETACHED
- loss = label * w_stc * (-log(p)) + (1-label) * (1-w_stc) * (-log(1-p))
- NOTE: Paper describes gamma exponent but CODE DOES NOT USE IT — we follow the code

## Key Configuration (verified against code + config)
- width: 12 (base channels)
- input_channel: 4 (x, y, t, polarity)
- spatial_shape: [352, 288, 8192] (padded from [346, 260, 8000])
- STC loss: k=3, τ=5 (no gamma in actual code)
- Downsampling: SparseConv3d(kernel=3, stride=[2,2,4], padding=1) — NOT kernel=stride!
- Training: Adam, lr=0.001, 50 epochs, StepLR(step=10, γ=0.1), batch_size=1
- BatchNorm1d: eps=1e-3, momentum=0.01

## Dependencies
- PyTorch >= 2.1.0
- spconv (sparse convolutions) — `spconv-cu120` for local, `spconv-cu118` for Colab
- numpy, scipy, scikit-learn (clustering)
- opencv-python (visualization)
- uv (package manager)

## Development Notes
- Voxelization is implemented in pure PyTorch (no custom CUDA ops required) using torch.unique + scatter
- Code must be CUDA GPU compatible and Colab-ready
- Always test with `uv run pytest tests/` after changes
- The detection pipeline outputs bounding boxes as (x_min, y_min, x_max, y_max) in pixel coordinates
- spconv tests skip on macOS (no CUDA) — verify on GPU machine or Colab
