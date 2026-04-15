"""Configuration loader for EV-Drone-Detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "spgnet"
    width: int = 12
    input_channel: int = 4
    dilations: list[int] = field(default_factory=lambda: [1, 2, 3, 4])


@dataclass
class SensorConfig:
    resolution: list[int] = field(default_factory=lambda: [346, 260])
    whole_t: int = 8000
    # Padded spatial shape for spconv — must match original [352, 288, 8192]
    spatial_shape: list[int] = field(default_factory=lambda: [352, 288, 8192])


@dataclass
class DataConfig:
    max_events: int = 700_000
    train_dir: str = "data/train"
    val_dir: str = "data/val"
    test_dir: str = "data/test"
    num_workers: int = 4


@dataclass
class SchedulerConfig:
    type: str = "step"
    step_size: int = 10
    gamma: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 1
    optimizer: str = "adam"
    lr: float = 0.001
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    seed: int = 37
    val_start_epoch: int = 40
    save_dir: str = "checkpoints"


@dataclass
class STCConfig:
    k: int = 3
    tau: int = 5


@dataclass
class LossConfig:
    stc: STCConfig = field(default_factory=STCConfig)


@dataclass
class DetectionConfig:
    seg_threshold: float = 0.5
    min_cluster_size: int = 5
    cluster_eps: float = 10.0
    cluster_min_samples: int = 3
    bbox_padding: int = 5
    max_detections: int = 10


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)


def _update_dataclass(dc: Any, d: dict) -> None:
    """Recursively update a dataclass from a dictionary."""
    for key, value in d.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _update_dataclass(current, value)
        else:
            setattr(dc, key, value)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file, falling back to defaults."""
    cfg = Config()
    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f)
            if raw:
                _update_dataclass(cfg, raw)
    return cfg
