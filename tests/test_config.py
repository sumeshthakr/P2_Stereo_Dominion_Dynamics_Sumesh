"""
Unit tests for camera configuration module.
Author: Sumesh Thakur (sumeshthkr@gmail.com)
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    CameraConfig,
    load_config_from_json,
    create_default_config,
    save_config_to_json
)


class TestCameraConfig:
    """Tests for CameraConfig dataclass."""
    
    def test_baseline_calculation(self):
        """Test baseline property calculation."""
        config = create_default_config(baseline=0.54)
        assert abs(config.baseline - 0.54) < 1e-6
    
    def test_focal_length_property(self):
        """Test focal length extraction from camera matrix."""
        config = create_default_config(focal_length=700.0)
        assert abs(config.focal_length_left - 700.0) < 1e-6
        assert abs(config.focal_length_right - 700.0) < 1e-6
    
    def test_principal_point_property(self):
        """Test principal point extraction."""
        config = create_default_config(
            focal_length=700.0,
            principal_point=(640.0, 360.0)
        )
        cx, cy = config.principal_point_left
        assert abs(cx - 640.0) < 1e-6
        assert abs(cy - 360.0) < 1e-6


class TestConfigLoading:
    """Tests for configuration loading and saving."""
    
    def test_load_valid_config(self):
        """Test loading a valid JSON configuration."""
        config_data = {
            "camera_matrix_left": [
                [721.5, 0.0, 609.5],
                [0.0, 721.5, 172.8],
                [0.0, 0.0, 1.0]
            ],
            "camera_matrix_right": [
                [721.5, 0.0, 609.5],
                [0.0, 721.5, 172.8],
                [0.0, 0.0, 1.0]
            ],
            "dist_coeffs_left": [0.0, 0.0, 0.0, 0.0, 0.0],
            "dist_coeffs_right": [0.0, 0.0, 0.0, 0.0, 0.0],
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "T": [-0.54, 0.0, 0.0],
            "image_size": [1242, 375]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = load_config_from_json(temp_path)
            assert config.image_size == (1242, 375)
            assert config.camera_matrix_left.shape == (3, 3)
            assert abs(config.baseline - 0.54) < 1e-6
        finally:
            Path(temp_path).unlink()
    
    def test_load_missing_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config_from_json("/nonexistent/path/config.json")
    
    def test_load_invalid_config(self):
        """Test loading configuration with missing fields."""
        invalid_config = {"camera_matrix_left": [[1, 0, 0]]}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_config_from_json(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_save_and_load_config(self):
        """Test round-trip save and load."""
        original = create_default_config(
            focal_length=800.0,
            baseline=0.6,
            image_size=(1920, 1080)
        )
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_config_to_json(original, temp_path)
            loaded = load_config_from_json(temp_path)
            
            assert loaded.image_size == original.image_size
            assert abs(loaded.baseline - original.baseline) < 1e-6
            assert abs(loaded.focal_length_left - original.focal_length_left) < 1e-6
        finally:
            Path(temp_path).unlink()


class TestDefaultConfig:
    """Tests for default configuration creation."""
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        config = create_default_config()
        
        assert config.camera_matrix_left.shape == (3, 3)
        assert config.camera_matrix_right.shape == (3, 3)
        assert config.R.shape == (3, 3)
        assert config.T.shape == (3,)
        assert len(config.dist_coeffs_left) == 5
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        config = create_default_config(
            focal_length=500.0,
            principal_point=(320.0, 240.0),
            baseline=0.3,
            image_size=(640, 480)
        )
        
        assert abs(config.focal_length_left - 500.0) < 1e-6
        assert config.image_size == (640, 480)
        assert abs(config.baseline - 0.3) < 1e-6
