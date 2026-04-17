# EV-Drone-Detector

Event-based drone detection using SPGNet for initial bounding box estimation from event camera data.

Based on [EV-SpSegNet](https://github.com/ChenYichen9527/Ev-UAV) and the paper "Event-based Tiny Object Detection: A Benchmark Dataset and Baseline" (Chen et al., 2025).

## Overview

This project adapts the EV-SpSegNet architecture for **initial drone detection** from event camera data. The pipeline:

1. **Event Voxelization**: Raw events (x, y, t, polarity) are voxelized into sparse 3D tensors
2. **SPGNet Segmentation**: U-shaped sparse 3D encoder-decoder with GDSCA modules segments drone events
3. **Clustering**: Positive events are clustered using DBSCAN to extract bounding boxes
4. **Output**: Initial bounding box(es) for downstream tracking

## Installation

### Local (CUDA GPU)

```bash
uv sync --extra cuda --extra dev
```

### Google Colab

```bash
pip install -e ".[colab,dev]"
```

### CPU-only (for development/testing)

```bash
uv sync --extra dev
```

## Usage

### Training

```bash
# With real EV-UAV data
uv run python scripts/train.py --config configs/default.yaml

# With synthetic data (for testing)
uv run python scripts/train.py --config configs/default.yaml --synthetic
```

### Detection

```bash
# Single file, with PNG visualization
uv run python scripts/detect.py \
    --checkpoint checkpoints/best_iou.pt \
    --input data/test/sample.npz \
    --visualize

# Whole directory, write an MP4 of all frames
uv run python scripts/detect.py \
    --checkpoint checkpoints/best_iou.pt \
    --input data/test/ \
    --video detections.mp4 --fps 25
```

### As a library

```python
from ev_drone_detector.detection.detector import DroneDetector

detector = DroneDetector.from_config("configs/default.yaml")
detector.load_weights("checkpoints/best_iou.pt")
detections = detector.detect(features, coords)

for det in detections:
    print(f"Drone at {det['bbox']}, confidence: {det['score']:.3f}")
```

## Project Structure

```
ev-drone-detector/
├── CLAUDE.md                  # Development guidelines and architecture notes
├── pyproject.toml             # Project config (uv)
├── configs/
│   └── default.yaml           # Training/inference configuration
├── src/ev_drone_detector/
│   ├── models/
│   │   ├── spgnet.py          # Main SPGNet model
│   │   ├── blocks.py          # GDBlock, SEModule, SparseBasicBlock
│   │   └── patch_attention.py # Patch Attention module
│   ├── data/
│   │   ├── event_repr.py      # Event voxelization (pure PyTorch)
│   │   └── dataset.py         # EV-UAV dataset loader + synthetic data
│   ├── detection/
│   │   ├── detector.py        # End-to-end detection pipeline
│   │   └── clustering.py      # DBSCAN clustering for bbox extraction
│   ├── losses/
│   │   └── stc_loss.py        # Spatiotemporal Correlation loss
│   └── utils/
│       ├── config.py          # YAML config system
│       ├── eval.py            # Evaluation metrics
│       └── viz.py             # Visualization utilities
├── scripts/
│   ├── train.py               # Training script
│   └── detect.py              # Inference script
└── tests/                     # Test suite
```

## References

- Paper: "Event-based Tiny Object Detection: A Benchmark Dataset and Baseline" (arXiv:2506.23575)
- Original code: https://github.com/ChenYichen9527/Ev-UAV
- Dataset: EV-UAV (147 sequences, 2.3M event-level annotations, DAVIS346 camera)
