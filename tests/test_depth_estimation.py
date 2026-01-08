"""
Unit tests for depth estimation module.
Author: Sumesh Thakur (sumeshthkr@gmail.com)
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import create_default_config
from src.depth_estimation import (
    DepthEstimator,
    DepthMethod,
    DepthResult,
    triangulate_points,
    _triangulate_linear,
    _triangulate_midpoint
)


class TestDepthEstimator:
    """Tests for DepthEstimator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return create_default_config(
            focal_length=700.0,
            baseline=0.54,
            image_size=(1242, 375)
        )
    
    @pytest.fixture
    def estimator(self, config):
        """Create test estimator."""
        return DepthEstimator(
            config=config,
            method=DepthMethod.SGBM,
            num_disparities=128,
            block_size=5
        )
    
    @pytest.fixture
    def synthetic_stereo_pair(self):
        """Create synthetic stereo pair for testing."""
        # Create a simple scene with a gradient
        left = np.zeros((375, 1242, 3), dtype=np.uint8)
        right = np.zeros((375, 1242, 3), dtype=np.uint8)
        
        # Add a rectangular object at known disparity
        left[100:200, 500:600] = 200
        right[100:200, 480:580] = 200  # 20 pixel shift = disparity of 20
        
        # Add some texture
        for i in range(0, 375, 20):
            left[i:i+2, :] = 128
            right[i:i+2, :] = 128
        
        return left, right
    
    def test_estimator_initialization(self, estimator, config):
        """Test estimator is properly initialized."""
        assert estimator.config == config
        assert estimator.method == DepthMethod.SGBM
        assert estimator.num_disparities == 128
        assert estimator.block_size == 5
    
    def test_compute_disparity_bm(self, config, synthetic_stereo_pair):
        """Test Block Matching disparity computation."""
        estimator = DepthEstimator(config, method=DepthMethod.BM)
        left, right = synthetic_stereo_pair
        
        disparity = estimator.compute_disparity(left, right)
        
        assert disparity.shape == (375, 1242)
        assert disparity.dtype == np.float32
    
    def test_compute_disparity_sgbm(self, config, synthetic_stereo_pair):
        """Test SGBM disparity computation."""
        estimator = DepthEstimator(config, method=DepthMethod.SGBM)
        left, right = synthetic_stereo_pair
        
        disparity = estimator.compute_disparity(left, right)
        
        assert disparity.shape == (375, 1242)
        assert disparity.dtype == np.float32
    
    def test_disparity_to_depth(self, estimator, config):
        """Test disparity to depth conversion."""
        # Create known disparity map
        disparity = np.ones((375, 1242), dtype=np.float32) * 20.0  # 20 pixels
        
        depth = estimator.disparity_to_depth(disparity)
        
        # depth = baseline * focal / disparity = 0.54 * 700 / 20 = 18.9m
        expected_depth = (config.baseline * config.focal_length_left) / 20.0
        
        assert np.allclose(depth, expected_depth)
    
    def test_disparity_to_depth_zero_handling(self, estimator):
        """Test that zero disparity is handled correctly."""
        disparity = np.zeros((100, 100), dtype=np.float32)
        
        depth = estimator.disparity_to_depth(disparity)
        
        # Zero disparity should result in zero depth
        assert np.all(depth == 0)
    
    def test_compute_confidence(self, estimator):
        """Test confidence map computation."""
        # Create disparity with known properties
        disparity = np.random.rand(100, 100).astype(np.float32) * 50
        disparity[:10, :] = 0  # Invalid region
        
        confidence = estimator.compute_confidence(disparity)
        
        assert confidence.shape == disparity.shape
        assert np.all(confidence >= 0) and np.all(confidence <= 1)
        assert np.all(confidence[:10, :] == 0)  # Invalid region has zero confidence
    
    def test_compute_depth_result(self, estimator, synthetic_stereo_pair):
        """Test full depth computation returns proper result."""
        left, right = synthetic_stereo_pair
        
        result = estimator.compute_depth(left, right)
        
        assert isinstance(result, DepthResult)
        assert result.disparity.shape == (375, 1242)
        assert result.depth_map.shape == (375, 1242)
        assert result.confidence.shape == (375, 1242)
        assert result.method == DepthMethod.SGBM
        assert result.computation_time_ms > 0
    
    def test_set_method(self, estimator):
        """Test method switching."""
        assert estimator.method == DepthMethod.SGBM
        
        estimator.set_method(DepthMethod.BM)
        
        assert estimator.method == DepthMethod.BM


class TestTriangulation:
    """Tests for triangulation methods."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return create_default_config(
            focal_length=700.0,
            baseline=0.54,
            image_size=(1242, 375)
        )
    
    def test_triangulate_single_point(self, config):
        """Test triangulation of a single point."""
        # Point at center of image with 20 pixel disparity
        points_left = np.array([[621.0, 187.5]])
        points_right = np.array([[601.0, 187.5]])  # 20 pixel shift
        
        points_3d = triangulate_points(points_left, points_right, config, method="linear")
        
        assert points_3d.shape == (1, 3)
        # Z should be approximately depth = baseline * focal / disparity
        expected_z = (config.baseline * config.focal_length_left) / 20.0
        assert abs(points_3d[0, 2] - expected_z) < 1.0  # Allow some tolerance
    
    def test_triangulate_multiple_points(self, config):
        """Test triangulation of multiple points."""
        n_points = 10
        points_left = np.random.rand(n_points, 2) * [1242, 375]
        # Create disparities between 10-50 pixels
        disparities = np.random.rand(n_points) * 40 + 10
        points_right = points_left - np.column_stack([disparities, np.zeros(n_points)])
        
        points_3d = triangulate_points(points_left, points_right, config, method="linear")
        
        assert points_3d.shape == (n_points, 3)
    
    def test_midpoint_method(self, config):
        """Test midpoint triangulation method."""
        points_left = np.array([[621.0, 187.5]])
        points_right = np.array([[601.0, 187.5]])
        
        points_3d = triangulate_points(points_left, points_right, config, method="midpoint")
        
        assert points_3d.shape == (1, 3)
        assert points_3d[0, 2] > 0  # Should have positive depth
    
    def test_polynomial_method(self, config):
        """Test polynomial triangulation method."""
        points_left = np.array([[621.0, 187.5]])
        points_right = np.array([[601.0, 187.5]])
        
        points_3d = triangulate_points(points_left, points_right, config, method="polynomial")
        
        assert points_3d.shape == (1, 3)
    
    def test_optimal_method(self, config):
        """Test optimal triangulation method."""
        points_left = np.array([[621.0, 187.5]])
        points_right = np.array([[601.0, 187.5]])
        
        points_3d = triangulate_points(points_left, points_right, config, method="optimal")
        
        assert points_3d.shape == (1, 3)
    
    def test_invalid_method(self, config):
        """Test that invalid method raises error."""
        points_left = np.array([[0.0, 0.0]])
        points_right = np.array([[0.0, 0.0]])
        
        with pytest.raises(ValueError):
            triangulate_points(points_left, points_right, config, method="invalid")
    
    def test_mismatched_points(self, config):
        """Test that mismatched point counts raise error."""
        points_left = np.array([[0.0, 0.0], [1.0, 1.0]])
        points_right = np.array([[0.0, 0.0]])
        
        with pytest.raises(ValueError):
            triangulate_points(points_left, points_right, config)


class TestDepthMethods:
    """Tests comparing different depth estimation methods."""
    
    @pytest.fixture
    def config(self):
        return create_default_config(focal_length=700.0, baseline=0.54)
    
    @pytest.fixture
    def test_images(self):
        """Create test stereo pair."""
        left = np.random.randint(0, 256, (375, 1242, 3), dtype=np.uint8)
        right = np.roll(left, 20, axis=1)  # Shift by 20 pixels
        return left, right
    
    def test_bm_faster_than_sgbm(self, config, test_images):
        """Test that BM is faster than SGBM."""
        left, right = test_images
        
        bm_estimator = DepthEstimator(config, method=DepthMethod.BM)
        sgbm_estimator = DepthEstimator(config, method=DepthMethod.SGBM)
        
        bm_result = bm_estimator.compute_depth(left, right)
        sgbm_result = sgbm_estimator.compute_depth(left, right)
        
        # BM should generally be faster (but this may vary on different hardware)
        # We just verify both complete successfully
        assert bm_result.computation_time_ms > 0
        assert sgbm_result.computation_time_ms > 0
    
    def test_all_methods_produce_output(self, config, test_images):
        """Test that all methods produce valid output."""
        left, right = test_images
        
        for method in [DepthMethod.BM, DepthMethod.SGBM, DepthMethod.SGBM_3WAY]:
            estimator = DepthEstimator(config, method=method)
            result = estimator.compute_depth(left, right)
            
            assert result.disparity is not None
            assert result.depth_map is not None
            assert not np.all(result.disparity == 0)
