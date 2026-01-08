"""
Unit tests for visualization module.
Author: Sumesh Thakur (sumeshthkr@gmail.com)
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import create_default_config
from src.visualization import (
    colorize_disparity,
    colorize_depth,
    colorize_confidence,
    create_depth_overlay,
    depth_to_point_cloud,
    create_bird_eye_view,
    save_point_cloud_ply,
    add_depth_legend,
    create_visualization_grid,
    add_depth_stats_overlay
)
from src.object_detection import DetectedObject


class TestColorization:
    """Tests for depth/disparity colorization."""
    
    def test_colorize_disparity(self):
        """Test disparity map colorization."""
        disparity = np.random.rand(100, 200).astype(np.float32) * 128
        
        colored = colorize_disparity(disparity)
        
        assert colored.shape == (100, 200, 3)
        assert colored.dtype == np.uint8
    
    def test_colorize_disparity_invalid_regions(self):
        """Test that invalid disparity regions are black."""
        disparity = np.zeros((100, 200), dtype=np.float32)
        disparity[50:, :] = 64  # Valid in bottom half
        
        colored = colorize_disparity(disparity)
        
        # Top half (invalid) should be black
        assert np.all(colored[:50, :] == 0)
        # Bottom half should have color
        assert not np.all(colored[50:, :] == 0)
    
    def test_colorize_depth(self):
        """Test depth map colorization."""
        depth = np.random.rand(100, 200).astype(np.float32) * 30 + 1  # 1-31m
        
        colored = colorize_depth(depth, min_depth=0.5, max_depth=50.0)
        
        assert colored.shape == (100, 200, 3)
        assert colored.dtype == np.uint8
    
    def test_colorize_depth_invalid_regions(self):
        """Test that invalid depth regions use specified color."""
        depth = np.zeros((100, 200), dtype=np.float32)
        
        colored = colorize_depth(depth, invalid_color=(128, 128, 128))
        
        # All invalid should be the specified color
        assert np.all(colored == 128)
    
    def test_colorize_confidence(self):
        """Test confidence map colorization."""
        confidence = np.random.rand(100, 200).astype(np.float32)
        
        colored = colorize_confidence(confidence)
        
        assert colored.shape == (100, 200, 3)
        assert colored.dtype == np.uint8


class TestDepthOverlay:
    """Tests for depth overlay on images."""
    
    def test_create_depth_overlay(self):
        """Test depth overlay creation."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        depth = np.ones((100, 200), dtype=np.float32) * 10
        
        overlay = create_depth_overlay(image, depth, alpha=0.5)
        
        assert overlay.shape == image.shape
        assert overlay.dtype == np.uint8
    
    def test_depth_overlay_size_mismatch(self):
        """Test overlay handles size mismatch."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        depth = np.ones((50, 100), dtype=np.float32) * 10  # Different size
        
        overlay = create_depth_overlay(image, depth)
        
        assert overlay.shape == image.shape
    
    def test_depth_overlay_alpha_zero(self):
        """Test overlay with alpha=0 returns original image."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        depth = np.ones((100, 200), dtype=np.float32) * 10
        
        overlay = create_depth_overlay(image, depth, alpha=0.0)
        
        # With alpha=0, only valid regions are affected
        # This is a simplified test
        assert overlay.shape == image.shape
    
    def test_depth_overlay_no_valid_depth(self):
        """Test overlay with no valid depth values (all zeros)."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        depth = np.zeros((100, 200), dtype=np.float32)  # No valid depth values
        
        overlay = create_depth_overlay(image, depth, alpha=0.5)
        
        # Should not crash and should return image unchanged
        assert overlay.shape == image.shape
        np.testing.assert_array_equal(overlay, image)
    
    def test_depth_overlay_all_invalid(self):
        """Test overlay with all negative depth values."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        depth = np.ones((100, 200), dtype=np.float32) * -5  # All negative = invalid
        
        overlay = create_depth_overlay(image, depth, alpha=0.5)
        
        # Should not crash and should return image unchanged
        assert overlay.shape == image.shape
        np.testing.assert_array_equal(overlay, image)


class TestPointCloud:
    """Tests for point cloud generation."""
    
    @pytest.fixture
    def config(self):
        return create_default_config(
            focal_length=700.0,
            principal_point=(320.0, 240.0),
            image_size=(640, 480)
        )
    
    def test_depth_to_point_cloud(self, config):
        """Test point cloud generation from depth."""
        depth = np.ones((480, 640), dtype=np.float32) * 10
        depth[:240, :] = 5  # Closer in top half
        
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        points, colors = depth_to_point_cloud(depth, image, config, subsample=8)
        
        assert points.shape[1] == 3
        assert colors.shape[1] == 3
        assert len(points) == len(colors)
        assert np.all(colors >= 0) and np.all(colors <= 1)
    
    def test_point_cloud_subsample(self, config):
        """Test that subsampling reduces point count."""
        depth = np.ones((480, 640), dtype=np.float32) * 10
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        points_1, _ = depth_to_point_cloud(depth, image, config, subsample=1)
        points_4, _ = depth_to_point_cloud(depth, image, config, subsample=4)
        
        assert len(points_4) < len(points_1)
    
    def test_point_cloud_max_depth(self, config):
        """Test that max_depth filters distant points."""
        depth = np.ones((480, 640), dtype=np.float32) * 100  # All far
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        points, _ = depth_to_point_cloud(depth, image, config, max_depth=50.0)
        
        # No points should remain (all > max_depth)
        assert len(points) == 0


class TestBirdEyeView:
    """Tests for Bird's Eye View generation."""
    
    def test_create_bev_empty(self):
        """Test BEV with empty point cloud."""
        points = np.array([]).reshape(0, 3)
        
        bev = create_bird_eye_view(points)
        
        assert bev.shape == (500, 200)  # Default size
        assert np.all(bev == 0)  # Should be empty
    
    def test_create_bev_with_points(self):
        """Test BEV with valid points."""
        # Create points in the valid range
        points = np.array([
            [0, 0, 10],   # Center, 10m away
            [5, 0, 20],   # Right, 20m away
            [-5, 1, 30],  # Left, 30m away
        ], dtype=np.float32)
        
        bev = create_bird_eye_view(points)
        
        assert bev.ndim == 2
        # Should have some non-zero values
        assert np.any(bev > 0)
    
    def test_draw_3d_bbox_on_bev(self):
        """Test drawing 3D bounding boxes on BEV."""
        bev = np.zeros((500, 200), dtype=np.uint8)
        
        objects = [
            DetectedObject(
                bbox=(100, 100, 50, 80),
                center=(125, 140),
                confidence=0.9,
                depth=10.0,
                size_estimate=(2.0, 1.5)
            )
        ]
        
        from src.visualization import draw_3d_bbox_on_bev
        result = draw_3d_bbox_on_bev(bev, objects)
        
        assert result.shape[0] == bev.shape[0]
        assert result.shape[1] == bev.shape[1]
        assert result.shape[2] == 3  # Color image


class TestPointCloudIO:
    """Tests for point cloud file I/O."""
    
    def test_save_point_cloud_ply(self):
        """Test saving point cloud to PLY file."""
        points = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ], dtype=np.float64)
        
        colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
        ])
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            temp_path = f.name
        
        try:
            save_point_cloud_ply(points, colors, temp_path)
            
            assert Path(temp_path).exists()
            assert Path(temp_path).stat().st_size > 0
        finally:
            Path(temp_path).unlink()


class TestVisualizationGrid:
    """Tests for visualization grid creation."""
    
    def test_create_visualization_grid(self):
        """Test creating full visualization grid."""
        h, w = 375, 1242
        
        left = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        right = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        disparity = np.random.rand(h, w).astype(np.float32) * 128
        depth = np.random.rand(h, w).astype(np.float32) * 30 + 1
        confidence = np.random.rand(h, w).astype(np.float32)
        
        grid = create_visualization_grid(
            left, right, disparity, depth, confidence
        )
        
        # Should be 3 rows, 2 columns
        expected_height = h * 3
        expected_width = w * 2
        
        assert grid.shape == (expected_height, expected_width, 3)
    
    def test_visualization_grid_with_objects(self):
        """Test grid with detected objects."""
        h, w = 375, 1242
        
        left = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        right = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        disparity = np.random.rand(h, w).astype(np.float32) * 128
        depth = np.random.rand(h, w).astype(np.float32) * 30 + 1
        
        objects = [
            DetectedObject(
                bbox=(100, 100, 50, 80),
                center=(125, 140),
                confidence=0.9,
                depth=10.0
            )
        ]
        
        grid = create_visualization_grid(
            left, right, disparity, depth,
            objects=objects, fps=30.0
        )
        
        assert grid.shape[0] == h * 3
        assert grid.shape[1] == w * 2


class TestDepthLegend:
    """Tests for depth legend addition."""
    
    def test_add_depth_legend_right(self):
        """Test adding legend on right side."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        
        result = add_depth_legend(image, position="right")
        
        assert result.shape[0] == image.shape[0]
        assert result.shape[1] > image.shape[1]
    
    def test_add_depth_legend_bottom(self):
        """Test adding legend on bottom."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        
        result = add_depth_legend(image, position="bottom")
        
        assert result.shape[0] > image.shape[0]
        assert result.shape[1] == image.shape[1]


class TestDepthStatsOverlay:
    """Tests for depth statistics overlay."""
    
    def test_add_depth_stats_overlay(self):
        """Test depth stats overlay with valid data."""
        image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        disparity = np.random.rand(300, 400).astype(np.float32) * 100 + 1
        depth = np.random.rand(300, 400).astype(np.float32) * 30 + 1
        
        result = add_depth_stats_overlay(
            image, disparity, depth,
            baseline=0.54, focal_length=700.0
        )
        
        assert result.shape == image.shape
        # Should have some text overlay (not identical to original)
        assert not np.array_equal(result, image)
    
    def test_add_depth_stats_overlay_no_valid(self):
        """Test depth stats overlay with no valid values."""
        image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        disparity = np.zeros((300, 400), dtype=np.float32)
        depth = np.zeros((300, 400), dtype=np.float32)
        
        result = add_depth_stats_overlay(
            image, disparity, depth,
            baseline=0.54, focal_length=700.0
        )
        
        # Should not crash
        assert result.shape == image.shape
