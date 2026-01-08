"""
Visualization Module
====================

Provides comprehensive visualization for stereo vision results including:
- Disparity and depth maps
- Point clouds and 3D reconstruction
- Bird's Eye View (BEV)
- 3D bounding boxes
- Confidence indicators

Author: Sumesh Thakur (sumeshthkr@gmail.com)

References:
- Open3D Point Cloud: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
- Matplotlib 3D: https://matplotlib.org/stable/gallery/mplot3d/index.html
- OpenCV Visualization: https://docs.opencv.org/4.x/df/d24/tutorial_js_image_display.html
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Try to import open3d for 3D visualization (optional)
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

# Matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .config import CameraConfig
from .object_detection import DetectedObject, draw_detections


def colorize_disparity(disparity: np.ndarray, max_disparity: Optional[float] = None) -> np.ndarray:
    """
    Apply colormap to disparity map for visualization.
    
    Uses the TURBO colormap which provides good perceptual uniformity
    and is suitable for depth visualization.
    
    Args:
        disparity: Raw disparity map
        max_disparity: Maximum disparity for normalization (auto if None)
        
    Returns:
        Colorized disparity map (BGR format)
        
    Reference: Google AI Blog - "Turbo, An Improved Rainbow Colormap"
    https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
    """
    # Normalize disparity
    if max_disparity is None:
        max_disparity = np.percentile(disparity[disparity > 0], 95) if np.any(disparity > 0) else 128
    
    normalized = np.clip(disparity / max_disparity, 0, 1)
    normalized = (normalized * 255).astype(np.uint8)
    
    # Apply colormap
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    
    # Mask invalid regions (where disparity is 0 or negative)
    invalid_mask = disparity <= 0
    colorized[invalid_mask] = [0, 0, 0]  # Black for invalid
    
    return colorized


def colorize_depth(
    depth: np.ndarray,
    min_depth: float = 0.5,
    max_depth: float = 50.0,
    invalid_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Apply colormap to depth map for visualization.
    
    Uses inverse normalization so closer objects appear warmer (red/yellow)
    and distant objects appear cooler (blue/purple).
    
    Args:
        depth: Depth map in meters
        min_depth: Minimum depth for visualization
        max_depth: Maximum depth for visualization
        invalid_color: Color for invalid depth values (BGR)
        
    Returns:
        Colorized depth map (BGR format)
    """
    # Clip and normalize (inverse so closer = brighter)
    clipped = np.clip(depth, min_depth, max_depth)
    normalized = 1.0 - (clipped - min_depth) / (max_depth - min_depth)
    normalized = (normalized * 255).astype(np.uint8)
    
    # Apply colormap
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
    
    # Mask invalid regions
    invalid_mask = (depth <= 0) | (depth > max_depth)
    colorized[invalid_mask] = invalid_color
    
    return colorized


def colorize_confidence(confidence: np.ndarray) -> np.ndarray:
    """
    Visualize confidence map.
    
    Green = high confidence
    Red = low confidence
    
    Args:
        confidence: Confidence map (0-1)
        
    Returns:
        Colorized confidence map (BGR format)
    """
    # Create RGB image (confidence as green channel)
    h, w = confidence.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Low confidence = red, high confidence = green
    conf_uint8 = (confidence * 255).astype(np.uint8)
    colorized[:, :, 1] = conf_uint8  # Green
    colorized[:, :, 2] = 255 - conf_uint8  # Red
    
    return colorized


def create_depth_overlay(
    image: np.ndarray,
    depth: np.ndarray,
    alpha: float = 0.5,
    min_depth: float = 0.5,
    max_depth: float = 50.0
) -> np.ndarray:
    """
    Overlay colorized depth on original image.
    
    Args:
        image: Original image (BGR)
        depth: Depth map
        alpha: Blending factor (0 = only image, 1 = only depth)
        min_depth: Minimum depth for visualization
        max_depth: Maximum depth for visualization
        
    Returns:
        Blended image with depth overlay
    """
    # Ensure same size
    if depth.shape[:2] != image.shape[:2]:
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
    
    colorized = colorize_depth(depth, min_depth, max_depth)
    
    # Blend only where depth is valid
    overlay = image.copy()
    valid_mask = (depth > 0) & (depth <= max_depth)
    
    # Only blend if there are valid depth values
    if np.any(valid_mask):
        blended = cv2.addWeighted(
            image[valid_mask], 1 - alpha,
            colorized[valid_mask], alpha,
            0
        )
        if blended is not None:
            overlay[valid_mask] = blended
    
    return overlay


def depth_to_point_cloud(
    depth: np.ndarray,
    image: np.ndarray,
    config: CameraConfig,
    subsample: int = 4,
    max_depth: float = 50.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth map to 3D point cloud.
    
    Uses the pinhole camera model to back-project pixels to 3D.
    
    Args:
        depth: Depth map in meters
        image: Color image for point colors
        config: Camera configuration
        subsample: Subsample factor for performance (1 = no subsampling)
        max_depth: Maximum depth to include
        
    Returns:
        Tuple of (points Nx3, colors Nx3)
        
    Reference: Point cloud generation from depth
    https://www.open3d.org/docs/release/tutorial/geometry/rgbd_image.html
    """
    # Subsample for performance
    depth_sub = depth[::subsample, ::subsample]
    image_sub = image[::subsample, ::subsample]
    
    h, w = depth_sub.shape
    
    # Create pixel coordinate grid
    u, v = np.meshgrid(
        np.arange(0, w) * subsample,
        np.arange(0, h) * subsample
    )
    
    # Camera intrinsics
    fx = config.focal_length_left
    fy = config.focal_length_left
    cx, cy = config.principal_point_left
    
    # Back-project to 3D
    Z = depth_sub
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Filter valid points
    valid = (Z > 0) & (Z < max_depth)
    
    points = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
    
    # Get colors (convert BGR to RGB and normalize)
    colors = image_sub[valid].astype(np.float64) / 255.0
    colors = colors[:, ::-1]  # BGR to RGB
    
    return points, colors


def create_bird_eye_view(
    points: np.ndarray,
    x_range: Tuple[float, float] = (-10, 10),
    z_range: Tuple[float, float] = (0, 50),
    resolution: float = 0.1,
    height_range: Tuple[float, float] = (-2, 2)
) -> np.ndarray:
    """
    Create Bird's Eye View (BEV) image from point cloud.
    
    Projects 3D points onto the ground plane (X-Z) and encodes
    height as intensity.
    
    Args:
        points: Nx3 point cloud
        x_range: Range in X direction (meters)
        z_range: Range in Z direction (depth, meters)
        resolution: Meters per pixel
        height_range: Range of heights to consider
        
    Returns:
        BEV image (grayscale, can be colorized)
        
    Reference: BEV generation for autonomous driving
    Used in many LiDAR processing pipelines
    """
    if len(points) == 0:
        # Return empty BEV
        w = int((x_range[1] - x_range[0]) / resolution)
        h = int((z_range[1] - z_range[0]) / resolution)
        return np.zeros((h, w), dtype=np.uint8)
    
    # Filter points within range
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] < z_range[1]) &
        (points[:, 1] >= height_range[0]) & (points[:, 1] < height_range[1])
    )
    filtered = points[mask]
    
    if len(filtered) == 0:
        w = int((x_range[1] - x_range[0]) / resolution)
        h = int((z_range[1] - z_range[0]) / resolution)
        return np.zeros((h, w), dtype=np.uint8)
    
    # Calculate BEV dimensions
    w = int((x_range[1] - x_range[0]) / resolution)
    h = int((z_range[1] - z_range[0]) / resolution)
    
    # Map points to pixel coordinates
    px = ((filtered[:, 0] - x_range[0]) / resolution).astype(int)
    pz = ((filtered[:, 2] - z_range[0]) / resolution).astype(int)
    
    # Normalize height to intensity
    heights = filtered[:, 1]
    intensities = ((heights - height_range[0]) / (height_range[1] - height_range[0]) * 255)
    intensities = np.clip(intensities, 0, 255).astype(np.uint8)
    
    # Create BEV image
    bev = np.zeros((h, w), dtype=np.uint8)
    
    # Clip to bounds
    valid = (px >= 0) & (px < w) & (pz >= 0) & (pz < h)
    bev[pz[valid], px[valid]] = intensities[valid]
    
    # Flip horizontally so left/right matches camera view
    # (looking from behind the camera, left should be on left)
    bev = np.fliplr(bev)
    
    # Flip vertically so farther = top, closer = bottom (ego at bottom)
    bev = np.flipud(bev)
    
    return bev


def draw_3d_bbox_on_bev(
    bev: np.ndarray,
    objects: List[DetectedObject],
    x_range: Tuple[float, float] = (-10, 10),
    z_range: Tuple[float, float] = (0, 50),
    resolution: float = 0.1
) -> np.ndarray:
    """
    Draw 3D bounding boxes on Bird's Eye View.
    
    Args:
        bev: BEV image
        objects: List of detected objects with depth and size
        x_range: X range of BEV
        z_range: Z range of BEV
        resolution: BEV resolution
        
    Returns:
        BEV with bounding boxes drawn
    """
    # Convert grayscale to BGR
    if len(bev.shape) == 2:
        bev_color = cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    else:
        bev_color = bev.copy()
    
    h, w = bev_color.shape[:2]
    
    for obj in objects:
        if obj.depth is None:
            continue
        
        # Object center in world coordinates (assume X=0, Z=depth)
        # This is a simplification - in reality we'd need full 3D pose
        z = obj.depth
        x = 0  # Center of image
        
        # If we have size estimate, use it
        if obj.size_estimate is not None:
            obj_width = obj.size_estimate[0]
        else:
            obj_width = 1.0  # Default 1m width
        
        # Map to BEV coordinates
        bev_x = int((x - x_range[0]) / resolution)
        bev_z = int((z - z_range[0]) / resolution)
        
        # Flip X to match the horizontal flip in create_bird_eye_view
        bev_x = w - 1 - bev_x
        # Flip Z to match the vertical flip in create_bird_eye_view
        bev_z = h - 1 - bev_z
        
        half_width = int(obj_width / resolution / 2)
        depth_extent = int(1.0 / resolution)  # Assume 1m depth extent
        
        # Draw rectangle
        x1 = max(0, bev_x - half_width)
        x2 = min(w, bev_x + half_width)
        y1 = max(0, bev_z - depth_extent)
        y2 = min(h, bev_z + depth_extent)
        
        cv2.rectangle(bev_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{obj.depth:.1f}m"
        cv2.putText(bev_color, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return bev_color


def create_visualization_grid(
    left_img: np.ndarray,
    right_img: np.ndarray,
    disparity: np.ndarray,
    depth: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    objects: Optional[List[DetectedObject]] = None,
    fps: Optional[float] = None
) -> np.ndarray:
    """
    Create a grid of all visualizations for display.
    
    Layout:
    +----------------+----------------+
    |   Left Image   |  Right Image   |
    +----------------+----------------+
    |   Disparity    |     Depth      |
    +----------------+----------------+
    |   Confidence   |  Depth Overlay |
    +----------------+----------------+
    
    Args:
        left_img: Left camera image
        right_img: Right camera image
        disparity: Disparity map
        depth: Depth map
        confidence: Confidence map (optional)
        objects: Detected objects (optional)
        fps: Current FPS for display
        
    Returns:
        Combined visualization image
    """
    # Ensure all images have same size
    h, w = left_img.shape[:2]
    
    # Resize depth/disparity to match if needed
    if disparity.shape[:2] != (h, w):
        disparity = cv2.resize(disparity, (w, h))
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h))
    
    # Create visualizations
    disparity_viz = colorize_disparity(disparity)
    depth_viz = colorize_depth(depth)
    
    if confidence is not None:
        if confidence.shape[:2] != (h, w):
            confidence = cv2.resize(confidence, (w, h))
        confidence_viz = colorize_confidence(confidence)
    else:
        confidence_viz = np.zeros_like(left_img)
    
    depth_overlay = create_depth_overlay(left_img, depth)
    
    # Draw objects on left image if provided
    left_display = left_img.copy()
    if objects:
        left_display = draw_detections(left_display, objects)
    
    # Add FPS indicator
    if fps is not None:
        cv2.putText(left_display, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Create grid
    row1 = np.hstack([left_display, right_img])
    row2 = np.hstack([disparity_viz, depth_viz])
    row3 = np.hstack([confidence_viz, depth_overlay])
    
    grid = np.vstack([row1, row2, row3])
    
    return grid


def visualize_point_cloud_open3d(
    points: np.ndarray,
    colors: np.ndarray,
    objects: Optional[List[DetectedObject]] = None,
    window_name: str = "Point Cloud"
) -> None:
    """
    Visualize point cloud using Open3D.
    
    Args:
        points: Nx3 point cloud
        colors: Nx3 RGB colors (0-1)
        objects: Optional detected objects to show as boxes
        window_name: Window title
        
    Note: Requires Open3D to be installed
    
    Reference: Open3D visualization
    https://www.open3d.org/docs/release/tutorial/visualization/visualization.html
    """
    if not OPEN3D_AVAILABLE:
        print("Open3D not available. Install with: pip install open3d")
        return
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = [pcd]
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coord_frame)
    
    # Add bounding boxes for detected objects
    if objects:
        for obj in objects:
            if obj.depth is None:
                continue
            
            # Create approximate 3D box
            depth = obj.depth
            if obj.size_estimate:
                w, h = obj.size_estimate
            else:
                w, h = 1.0, 1.0
            
            # Approximate position (centered at camera x, at detected depth)
            center = [0, 0, depth]
            extent = [w, h, 1.0]
            
            bbox = o3d.geometry.OrientedBoundingBox(
                center=center,
                R=np.eye(3),
                extent=extent
            )
            bbox.color = (0, 1, 0)  # Green
            geometries.append(bbox)
    
    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1280,
        height=720
    )


def visualize_point_cloud_matplotlib(
    points: np.ndarray,
    colors: np.ndarray,
    subsample: int = 10,
    title: str = "Point Cloud"
) -> None:
    """
    Visualize point cloud using Matplotlib (fallback if Open3D unavailable).
    
    Args:
        points: Nx3 point cloud
        colors: Nx3 RGB colors
        subsample: Subsample factor for faster plotting
        title: Plot title
        
    Reference: Matplotlib 3D scatter
    https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Subsample for performance
    points_sub = points[::subsample]
    colors_sub = colors[::subsample]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        points_sub[:, 0],
        points_sub[:, 2],  # Z as depth
        points_sub[:, 1],  # Y as height
        c=colors_sub,
        s=1
    )
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_zlabel('Y (m)')
    ax.set_title(title)
    
    plt.show()


def save_point_cloud_ply(
    points: np.ndarray,
    colors: np.ndarray,
    output_path: str
) -> None:
    """
    Save point cloud to PLY file.
    
    Args:
        points: Nx3 point cloud
        colors: Nx3 RGB colors (0-1)
        output_path: Output file path (.ply)
        
    Reference: PLY file format
    https://en.wikipedia.org/wiki/PLY_(file_format)
    """
    if OPEN3D_AVAILABLE:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(output_path, pcd)
    else:
        # Manual PLY writing
        n_points = len(points)
        
        with open(output_path, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Data
            for i in range(n_points):
                p = points[i]
                c = (colors[i] * 255).astype(int)
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def add_depth_legend(
    image: np.ndarray,
    min_depth: float = 0.5,
    max_depth: float = 50.0,
    position: str = "right"
) -> np.ndarray:
    """
    Add depth color legend to image.
    
    Args:
        image: Input image
        min_depth: Minimum depth value
        max_depth: Maximum depth value
        position: Legend position ("right" or "bottom")
        
    Returns:
        Image with legend
    """
    h, w = image.shape[:2]
    
    if position == "right":
        legend_width = 40
        legend = np.zeros((h, legend_width, 3), dtype=np.uint8)
        
        # Create gradient
        for i in range(h):
            depth_val = min_depth + (max_depth - min_depth) * i / h
            depth_arr = np.array([[depth_val]])
            color = colorize_depth(depth_arr, min_depth, max_depth)[0, 0]
            legend[i, :] = color
        
        # Add labels
        cv2.putText(legend, f"{min_depth:.1f}m", (2, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(legend, f"{max_depth:.1f}m", (2, h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return np.hstack([image, legend])
    
    else:  # bottom
        legend_height = 30
        legend = np.zeros((legend_height, w, 3), dtype=np.uint8)
        
        for i in range(w):
            depth_val = min_depth + (max_depth - min_depth) * i / w
            depth_arr = np.array([[depth_val]])
            color = colorize_depth(depth_arr, min_depth, max_depth)[0, 0]
            legend[:, i] = color
        
        return np.vstack([image, legend])


def add_depth_stats_overlay(
    image: np.ndarray,
    disparity: np.ndarray,
    depth: np.ndarray,
    baseline: float,
    focal_length: float
) -> np.ndarray:
    """
    Add depth verification statistics overlay to image.
    
    Shows disparity and depth statistics for verification.
    
    Args:
        image: Input image to overlay stats on
        disparity: Disparity map
        depth: Depth map in meters
        baseline: Camera baseline in meters
        focal_length: Focal length in pixels
        
    Returns:
        Image with stats overlay
    """
    result = image.copy()
    
    # Compute statistics for valid pixels
    valid_disp = disparity[disparity > 0]
    valid_depth = depth[(depth > 0) & (depth < 100)]
    
    stats = []
    stats.append(f"Camera: B={baseline:.4f}m, f={focal_length:.1f}px")
    
    if len(valid_disp) > 0:
        stats.append(f"Disparity: min={np.min(valid_disp):.1f}, max={np.max(valid_disp):.1f}, median={np.median(valid_disp):.1f}px")
    else:
        stats.append("Disparity: No valid values")
    
    if len(valid_depth) > 0:
        stats.append(f"Depth: min={np.min(valid_depth):.2f}, max={np.max(valid_depth):.2f}, median={np.median(valid_depth):.2f}m")
        
        # Verify depth formula: depth = baseline * focal_length / disparity
        # For the median disparity, calculate expected depth
        median_disp = np.median(valid_disp) if len(valid_disp) > 0 else 0
        if median_disp > 0:
            expected_depth = baseline * focal_length / median_disp
            stats.append(f"Verification: B*f/d = {expected_depth:.2f}m (median)")
    else:
        stats.append("Depth: No valid values")
    
    # Valid pixel percentage
    valid_pct = (len(valid_disp) / disparity.size) * 100
    stats.append(f"Valid pixels: {valid_pct:.1f}%")
    
    # Draw stats on image
    y_offset = 30
    for i, stat in enumerate(stats):
        # Draw background rectangle for readability
        text_size = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(result, (5, y_offset + i * 20 - 15), 
                     (15 + text_size[0], y_offset + i * 20 + 5), (0, 0, 0), -1)
        cv2.putText(result, stat, (10, y_offset + i * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return result
