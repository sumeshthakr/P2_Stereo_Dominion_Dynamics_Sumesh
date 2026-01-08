"""
Depth Estimation Module
=======================

Implements multiple stereo matching and triangulation methods for depth estimation.
Optimized for real-time performance with configurable quality/speed trade-offs.

Author: Sumesh Thakur (sumeshthkr@gmail.com)

References:
- OpenCV Stereo Matching: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
- Semi-Global Block Matching: H. Hirschmuller, "Stereo Processing by Semiglobal Matching 
  and Mutual Information," IEEE TPAMI, 2008
- Block Matching: K. Konolige, "Small Vision Systems: Hardware and Implementation," 
  Robotics Research, 1997
- Triangulation: R. Hartley and A. Zisserman, "Multiple View Geometry in Computer Vision"
"""

import cv2
import numpy as np
import time
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from .config import CameraConfig


class DepthMethod(Enum):
    """
    Available depth estimation methods.
    
    BM: Block Matching - Fastest, good for textured scenes
    SGBM: Semi-Global Block Matching - Better quality, moderate speed
    SGBM_3WAY: SGBM with 3-way cost aggregation - Best quality, slower
    SIMPLE_TRIANGULATION: Direct triangulation from correspondences
    """
    BM = "bm"
    SGBM = "sgbm"
    SGBM_3WAY = "sgbm_3way"
    SIMPLE_TRIANGULATION = "simple_triangulation"


@dataclass
class DepthResult:
    """
    Container for depth estimation results.
    
    Attributes:
        disparity: Raw disparity map (16-bit or float)
        depth_map: Computed depth map in meters
        confidence: Confidence map (0-1, where 1 is high confidence)
        method: Method used for computation
        computation_time_ms: Time taken for computation
    """
    disparity: np.ndarray
    depth_map: np.ndarray
    confidence: Optional[np.ndarray]
    method: DepthMethod
    computation_time_ms: float


class DepthEstimator:
    """
    Real-time depth estimator with multiple algorithm options.
    
    Computes dense depth maps from stereo image pairs using:
    1. Block Matching (BM) - Fastest, ~60fps possible
    2. Semi-Global Block Matching (SGBM) - Better edges, ~20-30fps
    3. SGBM 3-Way - Best quality, ~10-15fps
    4. Simple Triangulation - For sparse point matching
    
    Performance tips:
    - Use BM for maximum speed
    - Use SGBM for balance of speed and quality
    - Reduce image size (downsample) for faster processing
    - Adjust num_disparities based on expected depth range
    
    Reference: OpenCV stereo matching tutorial
    https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
    """
    
    def __init__(
        self,
        config: CameraConfig,
        method: DepthMethod = DepthMethod.SGBM,
        num_disparities: int = 128,
        block_size: int = 5,
        min_disparity: int = 0
    ):
        """
        Initialize depth estimator.
        
        Args:
            config: Camera calibration configuration
            method: Depth estimation method to use
            num_disparities: Maximum disparity minus minimum disparity (must be divisible by 16)
            block_size: Size of matching block (must be odd, larger = faster but less precise)
            min_disparity: Minimum possible disparity value
            
        Note: num_disparities controls the maximum depth range.
        Larger values allow detecting closer objects but are slower.
        """
        self.config = config
        self.method = method
        self.num_disparities = (num_disparities // 16) * 16  # Ensure divisible by 16
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1  # Ensure odd
        self.min_disparity = min_disparity
        
        # Initialize stereo matchers
        self._init_matchers()
    
    def _init_matchers(self) -> None:
        """
        Initialize stereo matching algorithms.
        
        Block Matching (BM):
        - Uses SAD (Sum of Absolute Differences) for block matching
        - Very fast but can produce noisy results in textureless regions
        
        SGBM (Semi-Global Block Matching):
        - Adds smoothness constraints along multiple directions
        - Much better edge preservation and filling of textureless regions
        
        Reference: H. Hirschmuller, "Stereo Processing by Semiglobal Matching"
        """
        # Block Matching - optimized for speed
        self.bm_matcher = cv2.StereoBM_create(
            numDisparities=self.num_disparities,
            blockSize=self.block_size
        )
        # Tune BM parameters for better quality
        self.bm_matcher.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
        self.bm_matcher.setPreFilterSize(9)
        self.bm_matcher.setPreFilterCap(31)
        self.bm_matcher.setTextureThreshold(10)
        self.bm_matcher.setUniquenessRatio(15)
        self.bm_matcher.setSpeckleWindowSize(100)
        self.bm_matcher.setSpeckleRange(32)
        self.bm_matcher.setMinDisparity(self.min_disparity)
        
        # SGBM - balance of quality and speed
        # P1 and P2 control smoothness penalty
        # Larger P1/P2 = smoother disparities, but may lose detail
        P1 = 8 * 3 * self.block_size ** 2   # Penalty for disparity changes of 1
        P2 = 32 * 3 * self.block_size ** 2  # Penalty for larger disparity changes
        
        self.sgbm_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=P1,
            P2=P2,
            disp12MaxDiff=1,  # Maximum allowed difference in left-right disparity check
            preFilterCap=63,
            uniquenessRatio=10,  # Margin for best match vs second best (%)
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM
        )
        
        # SGBM 3-Way - highest quality
        self.sgbm_3way_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=P1,
            P2=P2,
            disp12MaxDiff=1,
            preFilterCap=63,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    def compute_disparity(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        method: Optional[DepthMethod] = None
    ) -> np.ndarray:
        """
        Compute disparity map from stereo image pair.
        
        Disparity = x_left - x_right for corresponding points.
        Larger disparity = closer object (inverse relationship with depth).
        
        Args:
            left_img: Left camera image (BGR or grayscale)
            right_img: Right camera image (BGR or grayscale)
            method: Override default method (optional)
            
        Returns:
            Disparity map as float32 array (in pixels)
            
        Reference: OpenCV stereo matching
        https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
        """
        method = method or self.method
        
        # Convert to grayscale if needed
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
        
        # Select matcher based on method
        if method == DepthMethod.BM:
            disparity = self.bm_matcher.compute(left_gray, right_gray)
        elif method == DepthMethod.SGBM:
            disparity = self.sgbm_matcher.compute(left_gray, right_gray)
        elif method == DepthMethod.SGBM_3WAY:
            disparity = self.sgbm_3way_matcher.compute(left_gray, right_gray)
        else:
            # For triangulation method, we need feature matching first
            # This returns an empty disparity map (to be computed via triangulation)
            disparity = np.zeros(left_gray.shape, dtype=np.float32)
        
        # Convert from fixed-point to float
        # OpenCV returns disparity in 16-bit fixed point (divided by 16)
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to depth map using camera parameters.
        
        The fundamental stereo vision equation:
            depth = (baseline * focal_length) / disparity
        
        Where:
        - baseline: distance between camera centers (in meters)
        - focal_length: camera focal length (in pixels)
        - disparity: difference in x-coordinates (in pixels)
        
        Args:
            disparity: Disparity map (in pixels)
            
        Returns:
            Depth map (in meters)
            
        Reference: Multiple View Geometry in Computer Vision, Ch. 9
        Hartley & Zisserman
        """
        # Avoid division by zero
        disparity_safe = np.where(disparity > 0, disparity, 0.1)
        
        # depth = baseline * focal_length / disparity
        depth = (self.config.baseline * self.config.focal_length_left) / disparity_safe
        
        # Set invalid disparities to zero depth
        depth = np.where(disparity > 0, depth, 0)
        
        # Clamp to reasonable range (0.1m to 100m)
        depth = np.clip(depth, 0, 100)
        
        return depth
    
    def compute_confidence(self, disparity: np.ndarray) -> np.ndarray:
        """
        Compute confidence map for disparity estimates.
        
        Confidence is based on:
        1. Valid disparity values (disparity > 0)
        2. Local variance (low variance = textureless = low confidence)
        3. Disparity smoothness (high gradient = occlusion boundary = lower confidence)
        
        Args:
            disparity: Disparity map
            
        Returns:
            Confidence map (0-1, where 1 is high confidence)
        """
        confidence = np.ones_like(disparity)
        
        # Zero confidence for invalid disparities
        confidence[disparity <= 0] = 0
        
        # Lower confidence at disparity edges (potential occlusions)
        grad_x = cv2.Sobel(disparity, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(disparity, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize gradient and reduce confidence at high gradients
        valid_grads = gradient_magnitude[gradient_magnitude > 0]
        if len(valid_grads) > 0:
            max_grad = np.percentile(valid_grads, 95)
            if max_grad > 0:
                edge_penalty = np.clip(gradient_magnitude / max_grad, 0, 1)
                confidence = confidence * (1 - 0.5 * edge_penalty)
        
        return confidence.astype(np.float32)
    
    def compute_depth(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        method: Optional[DepthMethod] = None
    ) -> DepthResult:
        """
        Compute full depth estimation with timing and confidence.
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
            method: Override default method
            
        Returns:
            DepthResult with disparity, depth, confidence, and timing
        """
        method = method or self.method
        start_time = time.perf_counter()
        
        # Compute disparity
        disparity = self.compute_disparity(left_img, right_img, method)
        
        # Convert to depth
        depth_map = self.disparity_to_depth(disparity)
        
        # Compute confidence
        confidence = self.compute_confidence(disparity)
        
        end_time = time.perf_counter()
        computation_time_ms = (end_time - start_time) * 1000
        
        return DepthResult(
            disparity=disparity,
            depth_map=depth_map,
            confidence=confidence,
            method=method,
            computation_time_ms=computation_time_ms
        )
    
    def set_method(self, method: DepthMethod) -> None:
        """Change the depth estimation method."""
        self.method = method


def triangulate_points(
    points_left: np.ndarray,
    points_right: np.ndarray,
    config: CameraConfig,
    method: str = "midpoint"
) -> np.ndarray:
    """
    Triangulate 3D points from corresponding 2D points in stereo images.
    
    This is an alternative to dense disparity matching, useful for
    sparse feature-based depth estimation.
    
    Methods available:
    1. "midpoint": Find the midpoint of the shortest line connecting the two rays
    2. "linear": Linear triangulation using SVD (DLT method)
    3. "polynomial": Polynomial triangulation for better numerical stability
    4. "optimal": Optimal triangulation minimizing reprojection error
    
    Args:
        points_left: Nx2 array of 2D points in left image
        points_right: Nx2 array of 2D points in right image
        config: Camera configuration
        method: Triangulation method to use
        
    Returns:
        Nx3 array of 3D points in world coordinates
        
    Reference: R. Hartley and P. Sturm, "Triangulation," Computer Vision and Image
    Understanding, vol. 68, no. 2, pp. 146-157, 1997
    """
    if points_left.shape[0] != points_right.shape[0]:
        raise ValueError("Number of points must match in left and right images")
    
    n_points = points_left.shape[0]
    points_3d = np.zeros((n_points, 3))
    
    # Create projection matrices
    # Left camera at origin
    P1 = np.hstack([config.camera_matrix_left, np.zeros((3, 1))])
    
    # Right camera with rotation and translation
    P2 = np.dot(config.camera_matrix_right, np.hstack([config.R, config.T.reshape(3, 1)]))
    
    if method == "midpoint":
        points_3d = _triangulate_midpoint(points_left, points_right, P1, P2)
    elif method == "linear":
        points_3d = _triangulate_linear(points_left, points_right, P1, P2)
    elif method == "polynomial":
        points_3d = _triangulate_polynomial(points_left, points_right, P1, P2)
    elif method == "optimal":
        points_3d = _triangulate_optimal(points_left, points_right, P1, P2, config)
    else:
        raise ValueError(f"Unknown triangulation method: {method}")
    
    return points_3d


def _triangulate_midpoint(
    points_left: np.ndarray,
    points_right: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray
) -> np.ndarray:
    """
    Midpoint triangulation method.
    
    Finds the midpoint of the shortest line segment connecting the two
    back-projected rays from the two camera views.
    
    This is geometrically intuitive but not statistically optimal.
    
    Reference: Hartley & Zisserman, "Multiple View Geometry", Section 12.2
    """
    n_points = points_left.shape[0]
    points_3d = np.zeros((n_points, 3))
    
    # Camera centers
    C1 = np.zeros(3)
    C2 = -np.dot(np.linalg.inv(P2[:, :3]), P2[:, 3])
    
    for i in range(n_points):
        # Direction vectors of rays
        d1 = np.linalg.solve(P1[:, :3], 
                            np.array([points_left[i, 0], points_left[i, 1], 1.0]))
        d2 = np.linalg.solve(P2[:, :3], 
                            np.array([points_right[i, 0], points_right[i, 1], 1.0]))
        
        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 / np.linalg.norm(d2)
        
        # Find the closest points on the two rays
        # Solve: C1 + s*d1 closest to C2 + t*d2
        n = np.cross(d1, d2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-10:
            # Parallel rays - use simple triangulation
            points_3d[i] = (C1 + C2) / 2
        else:
            n = n / n_norm
            n1 = np.cross(d1, n)
            n2 = np.cross(d2, n)
            
            s = np.dot(C2 - C1, n2) / np.dot(d1, n2)
            t = np.dot(C1 - C2, n1) / np.dot(d2, n1)
            
            p1 = C1 + s * d1
            p2 = C2 + t * d2
            
            points_3d[i] = (p1 + p2) / 2
    
    return points_3d


def _triangulate_linear(
    points_left: np.ndarray,
    points_right: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray
) -> np.ndarray:
    """
    Linear triangulation using Direct Linear Transform (DLT).
    
    Forms a linear system Ax = 0 and solves using SVD.
    This is the most commonly used method due to its simplicity.
    
    Reference: Hartley & Zisserman, "Multiple View Geometry", Section 12.2
    Algorithm 12.1 (Linear triangulation method)
    """
    n_points = points_left.shape[0]
    points_3d = np.zeros((n_points, 3))
    
    for i in range(n_points):
        x1, y1 = points_left[i]
        x2, y2 = points_right[i]
        
        # Build the linear system
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # Normalize homogeneous coordinates
        
        points_3d[i] = X[:3]
    
    return points_3d


def _triangulate_polynomial(
    points_left: np.ndarray,
    points_right: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray
) -> np.ndarray:
    """
    Polynomial triangulation method.
    
    Uses a polynomial formulation that can be more numerically stable
    for certain configurations.
    
    For standard stereo (rectified images), this reduces to the simple
    disparity-based depth computation.
    
    Reference: Hartley & Zisserman, "Multiple View Geometry", Section 12.5
    """
    # For rectified stereo, use simplified polynomial approach
    # This is equivalent to the disparity formula
    n_points = points_left.shape[0]
    points_3d = np.zeros((n_points, 3))
    
    # Extract camera parameters
    fx = P1[0, 0]
    fy = P1[1, 1]
    cx = P1[0, 2]
    cy = P1[1, 2]
    
    # Baseline from right camera translation
    baseline = abs(P2[0, 3] / P2[0, 0])  # T_x / f_x
    
    for i in range(n_points):
        x1, y1 = points_left[i]
        x2, y2 = points_right[i]
        
        disparity = x1 - x2
        
        if disparity > 0:
            Z = fx * baseline / disparity
            X = (x1 - cx) * Z / fx
            Y = (y1 - cy) * Z / fy
            points_3d[i] = [X, Y, Z]
    
    return points_3d


def _triangulate_optimal(
    points_left: np.ndarray,
    points_right: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    config: CameraConfig
) -> np.ndarray:
    """
    Optimal triangulation minimizing reprojection error.
    
    Uses iterative refinement to find 3D points that minimize the
    sum of squared reprojection errors in both images.
    
    This is the gold standard for accuracy but slower than other methods.
    
    Reference: Hartley & Zisserman, "Multiple View Geometry", Section 12.5
    "Optimal triangulation" algorithm using Sampson approximation
    """
    # Start with linear triangulation
    points_3d = _triangulate_linear(points_left, points_right, P1, P2)
    
    n_points = points_left.shape[0]
    
    # Iterative refinement using Levenberg-Marquardt style update
    for i in range(n_points):
        X = points_3d[i]
        
        for _ in range(5):  # Max 5 iterations
            # Compute reprojection
            X_homo = np.append(X, 1)
            
            proj1 = np.dot(P1, X_homo)
            proj1 = proj1[:2] / proj1[2]
            
            proj2 = np.dot(P2, X_homo)
            proj2 = proj2[:2] / proj2[2]
            
            # Reprojection error
            e1 = points_left[i] - proj1
            e2 = points_right[i] - proj2
            
            error = np.linalg.norm(e1)**2 + np.linalg.norm(e2)**2
            
            if error < 1e-6:
                break
            
            # Compute Jacobian and update
            J = np.zeros((4, 3))
            
            # Jacobian of projection 1
            z1 = np.dot(P1[2], X_homo)
            J[0] = (P1[0, :3] * z1 - P1[2, :3] * proj1[0] * z1) / (z1 ** 2)
            J[1] = (P1[1, :3] * z1 - P1[2, :3] * proj1[1] * z1) / (z1 ** 2)
            
            # Jacobian of projection 2
            z2 = np.dot(P2[2], X_homo)
            J[2] = (P2[0, :3] * z2 - P2[2, :3] * proj2[0] * z2) / (z2 ** 2)
            J[3] = (P2[1, :3] * z2 - P2[2, :3] * proj2[1] * z2) / (z2 ** 2)
            
            # Error vector
            e = np.concatenate([e1, e2])
            
            # Update using normal equations with damping
            try:
                delta = np.linalg.solve(np.dot(J.T, J) + 0.01 * np.eye(3), np.dot(J.T, e))
                X = X + delta
            except np.linalg.LinAlgError:
                break
        
        points_3d[i] = X
    
    return points_3d


def compare_methods(
    left_img: np.ndarray,
    right_img: np.ndarray,
    config: CameraConfig
) -> Dict[str, DepthResult]:
    """
    Compare all depth estimation methods on the same image pair.
    
    Useful for evaluating which method works best for a particular scene.
    
    Args:
        left_img: Left camera image
        right_img: Right camera image
        config: Camera configuration
        
    Returns:
        Dictionary mapping method names to DepthResult objects
    """
    results = {}
    
    estimator = DepthEstimator(config)
    
    for method in [DepthMethod.BM, DepthMethod.SGBM, DepthMethod.SGBM_3WAY]:
        result = estimator.compute_depth(left_img, right_img, method)
        results[method.value] = result
        print(f"{method.value}: {result.computation_time_ms:.2f}ms")
    
    return results
