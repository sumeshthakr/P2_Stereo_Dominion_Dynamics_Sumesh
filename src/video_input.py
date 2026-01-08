"""
Video/Camera Input Module
=========================

Handles input from dual webcams or stereo video files.
Provides frame synchronization and preprocessing for stereo matching.

Author: Sumesh Thakur (sumeshthkr@gmail.com)

References:
- OpenCV VideoCapture: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
- Stereo Camera Systems: https://www.cvlibs.net/datasets/kitti/setup.php
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Generator, Union
from pathlib import Path
from dataclasses import dataclass
from .config import CameraConfig


@dataclass
class StereoFrame:
    """
    Container for a synchronized pair of stereo frames.
    
    Attributes:
        left: Left camera frame (BGR format)
        right: Right camera frame (BGR format)
        timestamp: Frame timestamp in milliseconds
        frame_number: Sequential frame number
    """
    left: np.ndarray
    right: np.ndarray
    timestamp: float
    frame_number: int


class StereoVideoSource:
    """
    Unified interface for stereo video input from files or webcams.
    
    Supports:
    - Dual webcam input (live feed)
    - Separate left/right video files
    - Side-by-side stereo video files
    
    Reference: OpenCV VideoCapture documentation
    https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html
    """
    
    def __init__(
        self,
        left_source: Union[str, int],
        right_source: Optional[Union[str, int]] = None,
        config: Optional[CameraConfig] = None,
        target_fps: Optional[float] = None,
        downsample_factor: float = 1.0
    ):
        """
        Initialize stereo video source.
        
        Args:
            left_source: Path to left video file or webcam index
            right_source: Path to right video file or webcam index (None for side-by-side video)
            config: Camera configuration for rectification
            target_fps: Target frame rate (None for original)
            downsample_factor: Factor to downsample frames (1.0 = no downsampling)
            
        Note: For real-time performance, consider using downsample_factor > 1.0
        """
        self.left_source = left_source
        self.right_source = right_source
        self.config = config
        self.target_fps = target_fps
        self.downsample_factor = downsample_factor
        
        self.cap_left: Optional[cv2.VideoCapture] = None
        self.cap_right: Optional[cv2.VideoCapture] = None
        self.is_side_by_side = right_source is None
        
        self.frame_count = 0
        self._original_fps: float = 30.0
        self._frame_width: int = 0
        self._frame_height: int = 0
        
        # Rectification maps (computed once for efficiency)
        self._rect_maps_left: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._rect_maps_right: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
    def open(self) -> bool:
        """
        Open video sources.
        
        Returns:
            True if sources opened successfully
        """
        # Open left source
        self.cap_left = cv2.VideoCapture(self.left_source)
        if not self.cap_left.isOpened():
            print(f"Error: Could not open left source: {self.left_source}")
            return False
        
        # Open right source if not side-by-side
        if not self.is_side_by_side:
            self.cap_right = cv2.VideoCapture(self.right_source)
            if not self.cap_right.isOpened():
                print(f"Error: Could not open right source: {self.right_source}")
                self.cap_left.release()
                return False
        
        # Get video properties
        self._original_fps = self.cap_left.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(self.cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self.cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Compute rectification maps if config is provided
        if self.config is not None:
            self._compute_rectification_maps()
        
        return True
    
    def _compute_rectification_maps(self) -> None:
        """
        Compute stereo rectification maps for efficient undistortion.
        
        Uses cv2.stereoRectify to compute rectification transforms that make
        epipolar lines horizontal, simplifying stereo matching.
        
        Reference: OpenCV stereoRectify documentation
        https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6
        """
        if self.config is None:
            return
        
        # Compute rectification parameters
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.config.camera_matrix_left,
            self.config.dist_coeffs_left,
            self.config.camera_matrix_right,
            self.config.dist_coeffs_right,
            self.config.image_size,
            self.config.R,
            self.config.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0  # Crop to valid pixels only
        )
        
        # Compute undistortion and rectification maps
        self._rect_maps_left = cv2.initUndistortRectifyMap(
            self.config.camera_matrix_left,
            self.config.dist_coeffs_left,
            R1, P1,
            self.config.image_size,
            cv2.CV_32FC1
        )
        
        self._rect_maps_right = cv2.initUndistortRectifyMap(
            self.config.camera_matrix_right,
            self.config.dist_coeffs_right,
            R2, P2,
            self.config.image_size,
            cv2.CV_32FC1
        )
        
        # Store Q matrix for depth computation
        self._Q = Q
    
    def read(self) -> Optional[StereoFrame]:
        """
        Read a synchronized pair of stereo frames.
        
        Returns:
            StereoFrame or None if no more frames
        """
        if self.cap_left is None:
            return None
        
        ret_left, frame_left = self.cap_left.read()
        if not ret_left:
            return None
        
        if self.is_side_by_side:
            # Split side-by-side video
            mid_x = frame_left.shape[1] // 2
            frame_right = frame_left[:, mid_x:]
            frame_left = frame_left[:, :mid_x]
        else:
            # Read from right source
            ret_right, frame_right = self.cap_right.read()
            if not ret_right:
                return None
        
        # Apply rectification if maps are available
        if self._rect_maps_left is not None:
            frame_left = cv2.remap(
                frame_left,
                self._rect_maps_left[0],
                self._rect_maps_left[1],
                cv2.INTER_LINEAR
            )
        
        if self._rect_maps_right is not None:
            frame_right = cv2.remap(
                frame_right,
                self._rect_maps_right[0],
                self._rect_maps_right[1],
                cv2.INTER_LINEAR
            )
        
        # Downsample if needed for performance
        if self.downsample_factor > 1.0:
            new_size = (
                int(frame_left.shape[1] / self.downsample_factor),
                int(frame_left.shape[0] / self.downsample_factor)
            )
            frame_left = cv2.resize(frame_left, new_size, interpolation=cv2.INTER_AREA)
            frame_right = cv2.resize(frame_right, new_size, interpolation=cv2.INTER_AREA)
        
        timestamp = self.cap_left.get(cv2.CAP_PROP_POS_MSEC)
        self.frame_count += 1
        
        return StereoFrame(
            left=frame_left,
            right=frame_right,
            timestamp=timestamp,
            frame_number=self.frame_count
        )
    
    def frames(self) -> Generator[StereoFrame, None, None]:
        """
        Generator that yields all frames from the video source.
        
        Yields:
            StereoFrame objects until video ends
        """
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame
    
    def close(self) -> None:
        """Release video captures."""
        if self.cap_left is not None:
            self.cap_left.release()
        if self.cap_right is not None:
            self.cap_right.release()
    
    @property
    def fps(self) -> float:
        """Get the original FPS of the video."""
        return self._original_fps
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get frame dimensions (width, height)."""
        return (self._frame_width, self._frame_height)
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def open_webcam_pair(
    left_index: int = 0,
    right_index: int = 1,
    config: Optional[CameraConfig] = None,
    resolution: Tuple[int, int] = (640, 480)
) -> StereoVideoSource:
    """
    Convenience function to open a pair of webcams.
    
    Args:
        left_index: Index of left webcam (default: 0)
        right_index: Index of right webcam (default: 1)
        config: Camera configuration
        resolution: Desired resolution (width, height)
        
    Returns:
        Configured StereoVideoSource
        
    Note: Ensure webcams are physically aligned horizontally for best results.
    """
    source = StereoVideoSource(
        left_source=left_index,
        right_source=right_index,
        config=config
    )
    return source


def open_video_pair(
    left_path: str,
    right_path: str,
    config: Optional[CameraConfig] = None,
    downsample_factor: float = 1.0
) -> StereoVideoSource:
    """
    Convenience function to open a pair of video files.
    
    Args:
        left_path: Path to left video file
        right_path: Path to right video file
        config: Camera configuration
        downsample_factor: Downsampling for performance
        
    Returns:
        Configured StereoVideoSource
    """
    # Validate paths
    if not Path(left_path).exists():
        raise FileNotFoundError(f"Left video not found: {left_path}")
    if not Path(right_path).exists():
        raise FileNotFoundError(f"Right video not found: {right_path}")
    
    return StereoVideoSource(
        left_source=left_path,
        right_source=right_path,
        config=config,
        downsample_factor=downsample_factor
    )
