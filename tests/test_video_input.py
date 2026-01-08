"""
Unit tests for video input module.
Author: Sumesh Thakur (sumeshthkr@gmail.com)
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import create_default_config
from src.video_input import StereoVideoSource, StereoFrame


class TestStereoFrame:
    """Tests for StereoFrame dataclass."""
    
    def test_stereo_frame_creation(self):
        """Test creating a StereoFrame."""
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        right = np.zeros((480, 640, 3), dtype=np.uint8)
        
        frame = StereoFrame(
            left=left,
            right=right,
            timestamp=1000.0,
            frame_number=1
        )
        
        assert frame.left.shape == (480, 640, 3)
        assert frame.right.shape == (480, 640, 3)
        assert frame.timestamp == 1000.0
        assert frame.frame_number == 1


class TestStereoVideoSource:
    """Tests for StereoVideoSource class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return create_default_config(
            focal_length=700.0,
            baseline=0.54,
            image_size=(1242, 375)
        )
    
    def test_video_source_initialization(self, config):
        """Test video source initialization."""
        source = StereoVideoSource(
            left_source="left.mp4",
            right_source="right.mp4",
            config=config,
            downsample_factor=2.0
        )
        
        assert source.left_source == "left.mp4"
        assert source.right_source == "right.mp4"
        assert source.downsample_factor == 2.0
        assert not source.is_side_by_side
    
    def test_side_by_side_mode(self, config):
        """Test side-by-side video mode."""
        source = StereoVideoSource(
            left_source="stereo.mp4",
            config=config
        )
        
        assert source.is_side_by_side
    
    @pytest.mark.skipif(
        not Path("/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/left.mp4").exists(),
        reason="Test video files not available"
    )
    def test_open_video_files(self, config):
        """Test opening actual video files."""
        source = StereoVideoSource(
            left_source="/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/left.mp4",
            right_source="/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/right.mp4",
            config=config
        )
        
        opened = source.open()
        assert opened
        
        frame = source.read()
        assert frame is not None
        assert isinstance(frame, StereoFrame)
        assert frame.left.shape == frame.right.shape
        
        source.close()
    
    @pytest.mark.skipif(
        not Path("/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/left.mp4").exists(),
        reason="Test video files not available"
    )
    def test_frame_generator(self, config):
        """Test frame generator."""
        source = StereoVideoSource(
            left_source="/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/left.mp4",
            right_source="/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/right.mp4",
            config=config
        )
        
        source.open()
        
        frame_count = 0
        for frame in source.frames():
            frame_count += 1
            if frame_count >= 5:
                break
        
        assert frame_count == 5
        source.close()
    
    @pytest.mark.skipif(
        not Path("/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/left.mp4").exists(),
        reason="Test video files not available"
    )
    def test_downsample_factor(self, config):
        """Test frame downsampling."""
        source_full = StereoVideoSource(
            left_source="/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/left.mp4",
            right_source="/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/right.mp4",
            config=config,
            downsample_factor=1.0
        )
        
        source_half = StereoVideoSource(
            left_source="/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/left.mp4",
            right_source="/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/right.mp4",
            config=config,
            downsample_factor=2.0
        )
        
        source_full.open()
        source_half.open()
        
        frame_full = source_full.read()
        frame_half = source_half.read()
        
        # Half resolution should have roughly half the dimensions
        assert frame_half.left.shape[0] < frame_full.left.shape[0]
        assert frame_half.left.shape[1] < frame_full.left.shape[1]
        
        source_full.close()
        source_half.close()
    
    def test_context_manager(self, config):
        """Test context manager usage."""
        # This test just checks the interface without actual files
        source = StereoVideoSource(
            left_source="nonexistent.mp4",
            right_source="nonexistent.mp4",
            config=config
        )
        
        # Context manager should not raise even if open fails
        with source:
            pass  # Open will fail silently


class TestVideoSourceProperties:
    """Tests for video source properties."""
    
    @pytest.fixture
    def config(self):
        return create_default_config()
    
    @pytest.mark.skipif(
        not Path("/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/left.mp4").exists(),
        reason="Test video files not available"
    )
    def test_fps_property(self, config):
        """Test FPS property."""
        source = StereoVideoSource(
            left_source="/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/left.mp4",
            right_source="/home/runner/work/problem2_stereo_vision/problem2_stereo_vision/right.mp4",
            config=config
        )
        
        source.open()
        
        # FPS should be a positive number
        assert source.fps > 0
        
        source.close()
