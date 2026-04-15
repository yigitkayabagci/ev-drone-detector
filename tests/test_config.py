"""Tests for configuration system."""

import pytest
from pathlib import Path
from ev_drone_detector.utils.config import load_config, Config


def test_default_config():
    """Test that default config loads with correct values."""
    cfg = load_config()
    assert cfg.model.width == 12
    assert cfg.model.input_channel == 4
    assert cfg.sensor.resolution == [346, 260]
    assert cfg.sensor.whole_t == 8000
    assert cfg.sensor.spatial_shape == [352, 288, 8192]
    assert cfg.loss.stc.k == 3
    assert cfg.loss.stc.tau == 5
    assert cfg.training.epochs == 50
    assert cfg.detection.seg_threshold == 0.5


def test_yaml_config_load():
    """Test loading from the default YAML config file."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if config_path.exists():
        cfg = load_config(config_path)
        assert cfg.model.width == 12
        assert cfg.model.dilations == [1, 2, 3, 4]
        assert cfg.training.lr == 0.001
        assert cfg.sensor.spatial_shape == [352, 288, 8192]


def test_config_types():
    """Test that config fields have correct types."""
    cfg = Config()
    assert isinstance(cfg.model.width, int)
    assert isinstance(cfg.training.lr, float)
    assert isinstance(cfg.model.dilations, list)
    assert isinstance(cfg.sensor.spatial_shape, list)
