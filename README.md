# Stereo Vision Depth Estimation System

A real-time depth estimation system using stereo camera pairs with object detection, point cloud generation, and 3D visualization.

**Author:** Sumesh Thakur (sumeshthkr@gmail.com)

---

## Features

- **Real-time depth estimation** from stereo camera pairs
- **Multiple triangulation methods** with comparison tools:
  - Block Matching (BM)
  - Semi-Global Block Matching (SGBM)
  - SGBM 3-Way - Highest quality
  - Linear/Midpoint/Polynomial/Optimal triangulation for sparse points
- **Input**:
  - Dual webcam live feed
  - Left/Right video files
- **Configurable camera parameters** via JSON or interactive entry

---

## Installation

### Requirements

- Python 3.8+
- OpenCV 4.5+
- NumPy 1.19+

### Install Dependencies

```bash
cd problem2_stereo_vision

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

For 3D point cloud visualization:

```bash
pip install open3d
```

For YOLOv8 Nano ML-based object detection:

```bash
pip install onnxruntime
```

---

## Commands

### Process stereo video files

```bash
python main.py --left ./scenes/scene_1/sequence_0000_left.mp4 --right /scenes/scene_1/sequence_0000_right.mp4 --config 0000.json
```

### Use webcams

```bash
python main.py --webcam 0 1 --config calibration.json
```

### Compare depth estimation methods

```bash
python main.py --left left.mp4 --right right.mp4 --config calibration.json --compare-methods
```

---

## Usage

### Command Line Options

```
Usage: python main.py [OPTIONS]

Input Sources:
  --webcam LEFT RIGHT    Webcam indices for left and right cameras
  --left PATH            Path to left video file
  --right PATH           Path to right video file

Configuration:
  --config PATH          Camera calibration JSON file
  --interactive-config   Interactively enter camera parameters

Depth Estimation:
  --method {bm,sgbm,sgbm_3way}
                         Depth estimation method (default: sgbm)
  --compare-methods      Compare all methods on first frame
  --num-disparities N    Number of disparities (default: 128)
  --block-size N         Block size for matching (default: 5)

Object Detection:
  --detector {contour,background_sub,hog_svm,color_blob,mobilenet_ssd,yolo_nano,none}
                         Detection method (default: contour)
                         ML-based: mobilenet_ssd, yolo_nano
  --max-objects N        Maximum objects to detect (default: 2)

Performance:
  --downsample FACTOR    Downsample factor (default: 1.0)

Output:
  --output DIR           Save results to directory
  --auto-save            Automatically save frames (requires --output)
  --save-interval N      Frame interval for auto-save (default: 30)
  --save-pointcloud      Save point clouds to PLY files
  --no-display           Run headless (no GUI)
  --show-bev             Show Bird's Eye View window
  --verify-depth         Show depth verification statistics overlay

Keyboard Controls (during visualization):
  q       - Quit
  s       - Save current frame
  p       - Save point cloud
  m       - Cycle through methods
  SPACE   - Pause/Resume playback
  + or =  - Increase delay (slower playback)
  -       - Decrease delay (faster playback)
```

### Example Commands

```bash
# Basic stereo processing with SGBM
python main.py --left left.mp4 --right right.mp4 --config calibration.json

# High-speed processing with downsampling
python main.py --left left.mp4 --right right.mp4 --config calibration.json \
    --method bm --downsample 2.0

# With object detection and BEV visualization
python main.py --left left.mp4 --right right.mp4 --config calibration.json \
    --detector contour --max-objects 2 --show-bev

# Auto-save outputs every 30 frames
python main.py --left left.mp4 --right right.mp4 --config calibration.json \
    --output outputs/ --auto-save --save-interval 30

# Save with point clouds
python main.py --left left.mp4 --right right.mp4 --config calibration.json \
    --output outputs/ --auto-save --save-pointcloud

# Verify depth calculations
python main.py --left left.mp4 --right right.mp4 --config calibration.json \
    --verify-depth
```

---

## Depth Estimation Methods

### 1. Block Matching (BM)

- **Quality:** Good for textured scenes
- **Algorithm:** Sum of Absolute Differences (SAD)

```python
from src.depth_estimation import DepthEstimator, DepthMethod
estimator = DepthEstimator(config, method=DepthMethod.BM)
```

### 2. Semi-Global Block Matching (SGBM)

- **Quality:** Better edges, handles textureless regions
- **Algorithm:** Multi-directional cost aggregation

```python
estimator = DepthEstimator(config, method=DepthMethod.SGBM)
```

### 3. SGBM 3-Way

- **Quality:** Highest quality
- **Algorithm:** 3-way cost aggregation

```python
estimator = DepthEstimator(config, method=DepthMethod.SGBM_3WAY)
```

### 4. Triangulation Methods (for sparse points)

```python
from src.depth_estimation import triangulate_points

# Available methods: "linear", "midpoint", "polynomial", "optimal"
points_3d = triangulate_points(
    points_left, points_right, config,
    method="linear"  # DLT triangulation
)
```

| Method     | Description                   | Use Case                |
| ---------- | ----------------------------- | ----------------------- |
| Linear     | Direct Linear Transform (DLT) | General purpose, fast   |
| Midpoint   | Ray intersection midpoint     | Geometrically intuitive |
| Polynomial | Rectified stereo formula      | Calibrated stereo rigs  |
| Optimal    | Minimizes reprojection error  | Highest accuracy        |

---

## Object Detection

Lightweight detectors optimized for real-time performance:

### Traditional CV Methods

#### Contour Detection (Default)

```python
from src.object_detection import ObjectDetector, DetectorType
detector = ObjectDetector(detector_type=DetectorType.CONTOUR, max_objects=2)
```

#### Background Subtraction

Best for moving objects against static background.

```python
detector = ObjectDetector(detector_type=DetectorType.BACKGROUND_SUB)
```

#### HOG+SVM

Classic human/pedestrian detection.

```python
detector = ObjectDetector(detector_type=DetectorType.HOG_SVM)
```

#### Color Blob

For distinctly colored objects.

```python
detector = ObjectDetector(detector_type=DetectorType.COLOR_BLOB)
```

### Lightweight ML-Based Methods

#### MobileNet-SSD

Lightweight neural network combining MobileNet backbone with SSD detection.

- **Accuracy:** Good for general object detection
- **Classes:** 20 PASCAL VOC classes (person, car, bicycle, etc.)

```python
detector = ObjectDetector(detector_type=DetectorType.MOBILENET_SSD, max_objects=2)
```

```bash
python main.py --left left.mp4 --right right.mp4 --config calibration.json --detector mobilenet_ssd
```

#### YOLOv8 Nano

Ultra-lightweight YOLO variant optimized for edge devices.

- **Accuracy:** Best among lightweight models
- **Classes:** 80 COCO classes
- **Requires:** onnxruntime (`pip install onnxruntime`)

```python
detector = ObjectDetector(detector_type=DetectorType.YOLO_NANO, max_objects=2)
```

```bash
python main.py --left left.mp4 --right right.mp4 --config calibration.json --detector yolo_nano
```

---

## Visualization

### Disparity/Depth Maps

```python
from src.visualization import colorize_disparity, colorize_depth

disparity_colored = colorize_disparity(disparity_map)
depth_colored = colorize_depth(depth_map, min_depth=0.5, max_depth=50.0)
```

### Point Cloud Generation

```python
from src.visualization import depth_to_point_cloud, save_point_cloud_ply

points, colors = depth_to_point_cloud(depth_map, image, config)
save_point_cloud_ply(points, colors, "output.ply")
```

### Bird's Eye View

```python
from src.visualization import create_bird_eye_view

bev = create_bird_eye_view(
    points,
    x_range=(-10, 10),
    z_range=(0, 50),
    resolution=0.1
)
```

---

## Configuration

### Camera Calibration JSON Format

```json
{
  "camera_matrix_left": [
    [721.5377, 0.0, 609.5593],
    [0.0, 721.5377, 172.854],
    [0.0, 0.0, 1.0]
  ],
  "camera_matrix_right": [
    [721.5377, 0.0, 609.5593],
    [0.0, 721.5377, 172.854],
    [0.0, 0.0, 1.0]
  ],
  "dist_coeffs_left": [0.0, 0.0, 0.0, 0.0, 0.0],
  "dist_coeffs_right": [0.0, 0.0, 0.0, 0.0, 0.0],
  "R": [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ],
  "T": [-0.54, 0.0, 0.0],
  "image_size": [1242, 375]
}
```

### Parameters

| Parameter         | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `camera_matrix_*` | 3x3 intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1] |
| `dist_coeffs_*`   | Distortion coefficients (k1, k2, p1, p2, k3)         |
| `R`               | 3x3 rotation matrix between cameras                  |
| `T`               | Translation vector (baseline)                        |
| `image_size`      | [width, height]                                      |

---

---

## References

### Academic Papers

1. H. Hirschmuller, "Stereo Processing by Semiglobal Matching and Mutual Information," IEEE TPAMI, 2008
2. K. Konolige, "Small Vision Systems: Hardware and Implementation," Robotics Research, 1997
3. R. Hartley and A. Zisserman, "Multiple View Geometry in Computer Vision," Cambridge University Press
4. N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," CVPR 2005
5. R. Hartley and P. Sturm, "Triangulation," Computer Vision and Image Understanding, 1997

### Tutorials and Documentation

- [OpenCV Stereo Vision Tutorial](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/)
- [Open3D Point Cloud Tutorial](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html)

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_depth_estimation.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```
