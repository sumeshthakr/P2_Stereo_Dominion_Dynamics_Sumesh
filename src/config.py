"""
Camera Configuration Module
===========================

Handles loading and validation of camera intrinsic and extrinsic parameters.
Supports JSON configuration files or manual parameter entry.

Author: Sumesh Thakur (sumeshthkr@gmail.com)

References:
- OpenCV Camera Calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- KITTI Dataset Calibration: https://www.cvlibs.net/datasets/kitti/setup.php
"""

import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path


@dataclass
class CameraConfig:
    """
    Stores camera calibration parameters for a stereo camera pair.
    
    Attributes:
        camera_matrix_left: 3x3 intrinsic matrix for left camera [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        camera_matrix_right: 3x3 intrinsic matrix for right camera
        dist_coeffs_left: Distortion coefficients for left camera (k1, k2, p1, p2, k3)
        dist_coeffs_right: Distortion coefficients for right camera
        R: 3x3 rotation matrix between cameras
        T: 3x1 translation vector between cameras (baseline)
        image_size: Tuple of (width, height) of the images
        
    The baseline (distance between cameras) can be computed from T.
    Reference: Multiple View Geometry in Computer Vision, Hartley & Zisserman, Ch. 9
    """
    camera_matrix_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_left: np.ndarray
    dist_coeffs_right: np.ndarray
    R: np.ndarray
    T: np.ndarray
    image_size: Tuple[int, int]
    
    @property
    def baseline(self) -> float:
        """
        Compute the baseline (horizontal distance between cameras).
        For a standard stereo rig, this is typically the x-component of T.
        
        Returns:
            Baseline distance in meters (or same units as calibration)
        """
        return abs(self.T[0])
    
    @property
    def focal_length_left(self) -> float:
        """Get focal length from left camera matrix (assumes fx ≈ fy)."""
        return self.camera_matrix_left[0, 0]
    
    @property
    def focal_length_right(self) -> float:
        """Get focal length from right camera matrix."""
        return self.camera_matrix_right[0, 0]
    
    @property
    def principal_point_left(self) -> Tuple[float, float]:
        """Get principal point (cx, cy) from left camera."""
        return (self.camera_matrix_left[0, 2], self.camera_matrix_left[1, 2])
    
    @property
    def principal_point_right(self) -> Tuple[float, float]:
        """Get principal point (cx, cy) from right camera."""
        return (self.camera_matrix_right[0, 2], self.camera_matrix_right[1, 2])


def load_config_from_json(config_path: str) -> CameraConfig:
    """
    Load camera configuration from a JSON file.
    
    The JSON file should contain:
    - camera_matrix_left: 3x3 intrinsic matrix
    - camera_matrix_right: 3x3 intrinsic matrix
    - dist_coeffs_left: Distortion coefficients (5 values)
    - dist_coeffs_right: Distortion coefficients (5 values)
    - R: 3x3 rotation matrix
    - T: 3x1 translation vector
    - image_size: [width, height]
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        CameraConfig object with loaded parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
        
    Reference: Format based on KITTI dataset calibration files
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Validate required fields
    required_fields = [
        'camera_matrix_left', 'camera_matrix_right',
        'dist_coeffs_left', 'dist_coeffs_right',
        'R', 'T', 'image_size'
    ]
    
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Convert to numpy arrays
    config = CameraConfig(
        camera_matrix_left=np.array(data['camera_matrix_left'], dtype=np.float64),
        camera_matrix_right=np.array(data['camera_matrix_right'], dtype=np.float64),
        dist_coeffs_left=np.array(data['dist_coeffs_left'], dtype=np.float64),
        dist_coeffs_right=np.array(data['dist_coeffs_right'], dtype=np.float64),
        R=np.array(data['R'], dtype=np.float64),
        T=np.array(data['T'], dtype=np.float64),
        image_size=tuple(data['image_size'])
    )
    
    # Validate matrix dimensions
    if config.camera_matrix_left.shape != (3, 3):
        raise ValueError("camera_matrix_left must be 3x3")
    if config.camera_matrix_right.shape != (3, 3):
        raise ValueError("camera_matrix_right must be 3x3")
    if config.R.shape != (3, 3):
        raise ValueError("R (rotation matrix) must be 3x3")
    if config.T.shape != (3,):
        raise ValueError("T (translation vector) must have 3 elements")
    
    return config


def create_default_config(
    focal_length: float = 700.0,
    principal_point: Tuple[float, float] = (640.0, 360.0),
    baseline: float = 0.54,
    image_size: Tuple[int, int] = (1280, 720)
) -> CameraConfig:
    """
    Create a default camera configuration with reasonable parameters.
    Useful for quick testing or when calibration data is not available.
    
    Args:
        focal_length: Focal length in pixels (default: 700)
        principal_point: Principal point (cx, cy) in pixels
        baseline: Distance between cameras in meters (default: 0.54m, KITTI-like)
        image_size: Image dimensions (width, height)
        
    Returns:
        CameraConfig with default parameters
        
    Note: For accurate depth estimation, proper camera calibration is essential.
    Reference: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """
    camera_matrix = np.array([
        [focal_length, 0.0, principal_point[0]],
        [0.0, focal_length, principal_point[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros(5, dtype=np.float64)
    
    R = np.eye(3, dtype=np.float64)
    T = np.array([-baseline, 0.0, 0.0], dtype=np.float64)
    
    return CameraConfig(
        camera_matrix_left=camera_matrix.copy(),
        camera_matrix_right=camera_matrix.copy(),
        dist_coeffs_left=dist_coeffs.copy(),
        dist_coeffs_right=dist_coeffs.copy(),
        R=R,
        T=T,
        image_size=image_size
    )


def save_config_to_json(config: CameraConfig, output_path: str) -> None:
    """
    Save camera configuration to a JSON file.
    
    Args:
        config: CameraConfig object to save
        output_path: Path for the output JSON file
    """
    data = {
        'camera_matrix_left': config.camera_matrix_left.tolist(),
        'camera_matrix_right': config.camera_matrix_right.tolist(),
        'dist_coeffs_left': config.dist_coeffs_left.tolist(),
        'dist_coeffs_right': config.dist_coeffs_right.tolist(),
        'R': config.R.tolist(),
        'T': config.T.tolist(),
        'image_size': list(config.image_size)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def interactive_config_entry() -> CameraConfig:
    """
    Interactively prompt the user to enter camera parameters.
    Useful when no configuration file is available.
    
    Returns:
        CameraConfig with user-entered parameters
    """
    print("\n=== Camera Configuration Entry ===")
    print("Enter camera parameters (press Enter for defaults)\n")
    
    # Get image size
    width_str = input("Image width [1280]: ").strip()
    height_str = input("Image height [720]: ").strip()
    width = int(width_str) if width_str else 1280
    height = int(height_str) if height_str else 720
    
    # Get focal length
    focal_str = input("Focal length in pixels [700.0]: ").strip()
    focal_length = float(focal_str) if focal_str else 700.0
    
    # Get principal point
    cx_str = input(f"Principal point cx [{width/2}]: ").strip()
    cy_str = input(f"Principal point cy [{height/2}]: ").strip()
    cx = float(cx_str) if cx_str else width / 2
    cy = float(cy_str) if cy_str else height / 2
    
    # Get baseline
    baseline_str = input("Baseline (distance between cameras) in meters [0.54]: ").strip()
    baseline = float(baseline_str) if baseline_str else 0.54
    
    return create_default_config(
        focal_length=focal_length,
        principal_point=(cx, cy),
        baseline=baseline,
        image_size=(width, height)
    )
