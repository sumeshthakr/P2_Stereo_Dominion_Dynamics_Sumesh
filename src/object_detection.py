"""
Object Detection Module
=======================

Lightweight object detection for real-time stereo vision.
Limits detection to 2 objects per frame for performance optimization.

Author: Sumesh Thakur (sumeshthkr@gmail.com)

References:
- OpenCV DNN: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html
- HOG+SVM: N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection"
- Background Subtraction: https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
- MobileNet-SSD: https://arxiv.org/abs/1704.04861
- YOLOv8: https://docs.ultralytics.com/
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import urllib.request
import os

# Try to import ONNX Runtime for lightweight ML inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class DetectorType(Enum):
    """
    Available lightweight detection methods.
    
    Traditional CV methods:
    HOG_SVM: Histogram of Oriented Gradients with SVM - Classic, reliable
    BACKGROUND_SUB: Background subtraction - Good for moving objects
    CONTOUR: Simple contour detection - Fastest, basic
    COLOR_BLOB: Color-based blob detection - Good for distinct colors
    
    Lightweight ML methods:
    MOBILENET_SSD: MobileNet-SSD via OpenCV DNN - Fast neural network
    YOLO_NANO: YOLOv8n via ONNX Runtime - Best accuracy/speed tradeoff
    """
    HOG_SVM = "hog_svm"
    BACKGROUND_SUB = "background_sub"
    CONTOUR = "contour"
    COLOR_BLOB = "color_blob"
    MOBILENET_SSD = "mobilenet_ssd"
    YOLO_NANO = "yolo_nano"


# Minimum file size in bytes to consider a model file valid
MIN_MODEL_SIZE_BYTES = 1000

# Default thresholds for ML-based detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_NMS_THRESHOLD = 0.45


@dataclass
class DetectedObject:
    """
    Represents a detected object with bounding box and properties.
    
    Attributes:
        bbox: Bounding box as (x, y, width, height)
        center: Center point of the bounding box (x, y)
        confidence: Detection confidence (0-1)
        label: Object class label
        depth: Estimated depth in meters (if available)
        size_estimate: Estimated real-world size in meters (width, height)
    """
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    label: str = "object"
    depth: Optional[float] = None
    size_estimate: Optional[Tuple[float, float]] = None


class ObjectDetector:
    """
    Real-time object detector optimized for stereo vision applications.
    
    Features:
    - Multiple detection algorithms (HOG, background subtraction, contours)
    - Limits output to max 2 largest objects for performance
    - Computes depth for detected objects using disparity map
    - Estimates real-world object size
    
    Performance: Designed to add < 10ms per frame overhead
    
    Reference: OpenCV object detection tutorials
    https://docs.opencv.org/4.x/d7/d8b/tutorial_py_face_detection.html
    """
    
    def __init__(
        self,
        detector_type: DetectorType = DetectorType.CONTOUR,
        max_objects: int = 2,
        min_area: int = 1000,
        max_area: Optional[int] = None
    ):
        """
        Initialize object detector.
        
        Args:
            detector_type: Type of detection algorithm
            max_objects: Maximum number of objects to return
            min_area: Minimum bounding box area in pixels
            max_area: Maximum bounding box area (None for no limit)
        """
        self.detector_type = detector_type
        self.max_objects = max_objects
        self.min_area = min_area
        self.max_area = max_area
        
        # For background subtraction - need history
        self._bg_subtractor: Optional[cv2.BackgroundSubtractor] = None
        self._frame_count = 0
        
        # ML model paths (will be downloaded if needed)
        self._model_dir = Path(__file__).parent.parent / "models"
        self._mobilenet_net: Optional[cv2.dnn.Net] = None
        self._yolo_session: Optional[object] = None
        
        # Initialize detectors based on type
        self._init_detector()
    
    def _init_detector(self) -> None:
        """Initialize the appropriate detector."""
        if self.detector_type == DetectorType.HOG_SVM:
            # HOG descriptor for pedestrian/object detection
            # Reference: N. Dalal and B. Triggs, CVPR 2005
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
        elif self.detector_type == DetectorType.BACKGROUND_SUB:
            # MOG2 background subtractor - works well for moving objects
            # Reference: Z. Zivkovic, "Improved adaptive Gaussian mixture model"
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )
        
        elif self.detector_type == DetectorType.MOBILENET_SSD:
            self._init_mobilenet_ssd()
        
        elif self.detector_type == DetectorType.YOLO_NANO:
            self._init_yolo_nano()
    
    def _init_mobilenet_ssd(self) -> None:
        """
        Initialize MobileNet-SSD detector using OpenCV DNN.
        
        MobileNet-SSD is a lightweight object detector combining:
        - MobileNet: Efficient depthwise separable convolutions
        - SSD: Single Shot MultiBox Detector
        
        Speed: ~30-50ms per frame on CPU
        Reference: https://arxiv.org/abs/1704.04861
        """
        self._model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model file paths
        prototxt_path = self._model_dir / "MobileNetSSD_deploy.prototxt"
        caffemodel_path = self._model_dir / "MobileNetSSD_deploy.caffemodel"
        
        # Download models if not present
        # Note: Using GitHub raw content from djmv's fork (repo name has typo "MobilNet")
        base_url = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master"
        
        try:
            if not prototxt_path.exists():
                print("Downloading MobileNet-SSD prototxt...")
                urllib.request.urlretrieve(
                    f"{base_url}/MobileNetSSD_deploy.prototxt",
                    str(prototxt_path)
                )
            
            if not caffemodel_path.exists():
                print("Downloading MobileNet-SSD caffemodel...")
                # Use GitHub releases URL for caffemodel
                model_url = f"{base_url}/MobileNetSSD_deploy.caffemodel"
                urllib.request.urlretrieve(model_url, str(caffemodel_path))
        except Exception as e:
            print(f"Could not download MobileNet-SSD model: {e}")
            print("Falling back to contour detection.")
            self._mobilenet_net = None
            self._mobilenet_classes = []
            return
        
        # Load the network
        try:
            if caffemodel_path.exists() and caffemodel_path.stat().st_size > MIN_MODEL_SIZE_BYTES:
                self._mobilenet_net = cv2.dnn.readNetFromCaffe(
                    str(prototxt_path),
                    str(caffemodel_path)
                )
                self._mobilenet_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self._mobilenet_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            else:
                print("MobileNet-SSD model not available. Falling back to contour detection.")
                self._mobilenet_net = None
        except Exception as e:
            print(f"Failed to load MobileNet-SSD: {e}. Falling back to contour detection.")
            self._mobilenet_net = None
        
        # Class labels for PASCAL VOC-trained MobileNet-SSD (21 classes including background)
        self._mobilenet_classes = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]
    
    def _init_yolo_nano(self) -> None:
        """
        Initialize YOLOv8 Nano detector using ONNX Runtime.
        
        YOLOv8n is the smallest variant of YOLOv8, optimized for:
        - Edge devices and real-time applications
        - ~3.2M parameters, ~8.7 GFLOPs
        
        Speed: ~20-40ms per frame on CPU
        Reference: https://docs.ultralytics.com/
        """
        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available. Install with: pip install onnxruntime")
            print("Falling back to contour detection.")
            self._yolo_session = None
            return
        
        self._model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._model_dir / "yolov8n.onnx"
        
        # Download model if not present
        if not model_path.exists():
            print("Downloading YOLOv8n ONNX model...")
            # Use ultralytics hosted model
            model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
            try:
                urllib.request.urlretrieve(model_url, str(model_path))
            except Exception as e:
                print(f"Failed to download YOLOv8n model: {e}")
                print("Falling back to contour detection.")
                self._yolo_session = None
                return
        
        # Load ONNX model
        try:
            self._yolo_session = ort.InferenceSession(
                str(model_path),
                providers=['CPUExecutionProvider']
            )
            self._yolo_input_name = self._yolo_session.get_inputs()[0].name
            self._yolo_input_shape = self._yolo_session.get_inputs()[0].shape
        except Exception as e:
            print(f"Failed to load YOLOv8n model: {e}")
            print("Falling back to contour detection.")
            self._yolo_session = None
        
        # COCO class names (80 classes)
        self._yolo_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
    
    def detect(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
        focal_length: Optional[float] = None
    ) -> List[DetectedObject]:
        """
        Detect objects in the image.
        
        Args:
            image: Input image (BGR format)
            depth_map: Optional depth map for 3D size estimation
            focal_length: Camera focal length for size estimation
            
        Returns:
            List of DetectedObject, limited to max_objects
        """
        if self.detector_type == DetectorType.HOG_SVM:
            objects = self._detect_hog(image)
        elif self.detector_type == DetectorType.BACKGROUND_SUB:
            objects = self._detect_background_sub(image)
        elif self.detector_type == DetectorType.CONTOUR:
            objects = self._detect_contours(image)
        elif self.detector_type == DetectorType.COLOR_BLOB:
            objects = self._detect_color_blobs(image)
        elif self.detector_type == DetectorType.MOBILENET_SSD:
            objects = self._detect_mobilenet_ssd(image)
        elif self.detector_type == DetectorType.YOLO_NANO:
            objects = self._detect_yolo_nano(image)
        else:
            objects = []
        
        # Add depth information if available
        if depth_map is not None:
            for obj in objects:
                obj.depth = self._get_object_depth(obj.bbox, depth_map)
                
                if focal_length is not None and obj.depth is not None:
                    obj.size_estimate = self._estimate_size(
                        obj.bbox, obj.depth, focal_length
                    )
        
        # Sort by area (largest first) and limit
        objects.sort(key=lambda o: o.bbox[2] * o.bbox[3], reverse=True)
        return objects[:self.max_objects]
    
    def _detect_hog(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects using HOG+SVM (Histogram of Oriented Gradients).
        
        Best for: Human detection, upright objects
        Speed: ~50-100ms per frame
        
        Reference: N. Dalal and B. Triggs, "Histograms of Oriented Gradients
        for Human Detection," CVPR 2005
        """
        # Resize for speed if image is large
        scale = 1.0
        if image.shape[1] > 800:
            scale = 800 / image.shape[1]
            small = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            small = image
        
        # Detect using HOG
        boxes, weights = self.hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        objects = []
        for i, (x, y, w, h) in enumerate(boxes):
            # Scale back to original size
            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            
            area = w * h
            if area < self.min_area:
                continue
            if self.max_area and area > self.max_area:
                continue
            
            confidence = float(weights[i]) if i < len(weights) else 0.5
            
            objects.append(DetectedObject(
                bbox=(x, y, w, h),
                center=(x + w//2, y + h//2),
                confidence=min(confidence, 1.0),
                label="person"
            ))
        
        return objects
    
    def _detect_background_sub(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Detect moving objects using background subtraction.
        
        Best for: Detecting moving objects against static background
        Speed: ~5-10ms per frame
        
        Reference: Z. Zivkovic, "Improved adaptive Gaussian mixture model for
        background subtraction," ICPR 2004
        """
        if self._bg_subtractor is None:
            self._init_detector()
        
        # Apply background subtraction
        fg_mask = self._bg_subtractor.apply(image)
        
        # Remove shadows (marked as gray in MOG2)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            if self.max_area and area > self.max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Confidence based on how much of bbox is filled
            fill_ratio = area / (w * h)
            
            objects.append(DetectedObject(
                bbox=(x, y, w, h),
                center=(x + w//2, y + h//2),
                confidence=fill_ratio,
                label="moving_object"
            ))
        
        self._frame_count += 1
        return objects
    
    def _detect_contours(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Simple contour-based detection.
        
        Best for: High-contrast objects, fast processing
        Speed: ~2-5ms per frame
        
        This is the fastest method but requires distinct objects.
        Reference: OpenCV contour detection
        https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate to close gaps
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            if self.max_area and area > self.max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Confidence based on contour properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            objects.append(DetectedObject(
                bbox=(x, y, w, h),
                center=(x + w//2, y + h//2),
                confidence=min(circularity + 0.3, 1.0),
                label="object"
            ))
        
        return objects
    
    def _detect_color_blobs(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Color-based blob detection.
        
        Best for: Distinctly colored objects (e.g., red/blue/green objects)
        Speed: ~5ms per frame
        
        Reference: OpenCV color filtering and blob detection
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        objects = []
        
        # Define color ranges to detect (common object colors)
        color_ranges = [
            # Red (wraps around in HSV)
            (np.array([0, 100, 100]), np.array([10, 255, 255]), "red_object"),
            (np.array([160, 100, 100]), np.array([180, 255, 255]), "red_object"),
            # Blue
            (np.array([100, 100, 100]), np.array([130, 255, 255]), "blue_object"),
            # Green
            (np.array([40, 100, 100]), np.array([80, 255, 255]), "green_object"),
            # Yellow
            (np.array([20, 100, 100]), np.array([35, 255, 255]), "yellow_object"),
        ]
        
        for lower, upper, label in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                if self.max_area and area > self.max_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                fill_ratio = area / (w * h)
                
                objects.append(DetectedObject(
                    bbox=(x, y, w, h),
                    center=(x + w//2, y + h//2),
                    confidence=fill_ratio,
                    label=label
                ))
        
        return objects
    
    def _detect_mobilenet_ssd(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects using MobileNet-SSD neural network.
        
        MobileNet-SSD combines:
        - MobileNet backbone: Uses depthwise separable convolutions
        - SSD head: Single-shot multibox detection
        
        Best for: General object detection with good speed/accuracy tradeoff
        Speed: ~30-50ms per frame on CPU
        
        Reference: Howard et al., "MobileNets: Efficient Convolutional Neural
        Networks for Mobile Vision Applications," arXiv:1704.04861
        """
        # Fall back to contour detection if model not loaded
        if self._mobilenet_net is None:
            return self._detect_contours(image)
        
        h, w = image.shape[:2]
        
        # Prepare input blob (300x300 is MobileNet-SSD input size)
        blob = cv2.dnn.blobFromImage(
            image, 
            scalefactor=0.007843,  # 1/127.5
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False
        )
        
        # Run inference
        self._mobilenet_net.setInput(blob)
        detections = self._mobilenet_net.forward()
        
        objects = []
        
        # Parse detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < DEFAULT_CONFIDENCE_THRESHOLD:
                continue
            
            class_id = int(detections[0, 0, i, 1])
            if class_id < 0 or class_id >= len(self._mobilenet_classes):
                continue
            
            # Get bounding box coordinates (normalized)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure valid coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            area = bbox_w * bbox_h
            
            if area < self.min_area:
                continue
            if self.max_area and area > self.max_area:
                continue
            
            objects.append(DetectedObject(
                bbox=(x1, y1, bbox_w, bbox_h),
                center=(x1 + bbox_w // 2, y1 + bbox_h // 2),
                confidence=float(confidence),
                label=self._mobilenet_classes[class_id]
            ))
        
        return objects
    
    def _detect_yolo_nano(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects using YOLOv8 Nano via ONNX Runtime.
        
        YOLOv8n is optimized for edge devices:
        - Only 3.2M parameters
        - 8.7 GFLOPs computational cost
        - Excellent accuracy for its size
        
        Best for: Real-time detection with best accuracy among lightweight models
        Speed: ~20-40ms per frame on CPU
        
        Reference: Ultralytics YOLOv8, https://docs.ultralytics.com/
        """
        # Fall back to contour detection if model not loaded
        if self._yolo_session is None:
            return self._detect_contours(image)
        
        h, w = image.shape[:2]
        
        # YOLOv8 expects 640x640 input
        input_size = 640
        
        # Resize and pad to maintain aspect ratio
        scale = min(input_size / w, input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        pad_x = (input_size - new_w) // 2
        pad_y = (input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Preprocess: BGR to RGB, normalize to [0, 1], CHW format
        input_data = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_data = input_data.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
        
        # Run inference
        outputs = self._yolo_session.run(None, {self._yolo_input_name: input_data})
        
        # Parse YOLOv8 output
        # Output shape: [1, 84, 8400] where 84 = 4 (box) + 80 (classes)
        output = outputs[0]
        
        # Transpose to [8400, 84]
        output = np.transpose(output[0])
        
        objects = []
        
        boxes = []
        confidences = []
        class_ids = []
        
        for detection in output:
            # Get class scores (indices 4-83)
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence < DEFAULT_CONFIDENCE_THRESHOLD:
                continue
            
            # Get box coordinates (center_x, center_y, width, height)
            cx, cy, bw, bh = detection[:4]
            
            # Convert from padded coordinates to original image coordinates
            cx = (cx - pad_x) / scale
            cy = (cy - pad_y) / scale
            bw = bw / scale
            bh = bh / scale
            
            # Convert to corner format
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            box_w = int(bw)
            box_h = int(bh)
            
            # Ensure valid coordinates
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            box_w = max(1, min(box_w, w - x1))
            box_h = max(1, min(box_h, h - y1))
            
            area = box_w * box_h
            if area < self.min_area:
                continue
            if self.max_area and area > self.max_area:
                continue
            
            boxes.append([x1, y1, box_w, box_h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_NMS_THRESHOLD
            )
            
            for i in indices:
                # OpenCV version compatibility: older versions return nested arrays,
                # newer versions return flat indices
                idx = i if isinstance(i, int) else i[0]
                x, y, bw, bh = boxes[idx]
                
                objects.append(DetectedObject(
                    bbox=(x, y, bw, bh),
                    center=(x + bw // 2, y + bh // 2),
                    confidence=confidences[idx],
                    label=self._yolo_classes[class_ids[idx]]
                ))
        
        return objects
    
    def _get_object_depth(
        self,
        bbox: Tuple[int, int, int, int],
        depth_map: np.ndarray
    ) -> Optional[float]:
        """
        Get the depth of an object from the depth map.
        
        Uses the median depth within the bounding box to be robust
        to noise and occlusions.
        
        Args:
            bbox: Bounding box (x, y, w, h)
            depth_map: Depth map in meters
            
        Returns:
            Median depth in meters, or None if invalid
        """
        x, y, w, h = bbox
        
        # Ensure bounds are within image
        x = max(0, x)
        y = max(0, y)
        x2 = min(x + w, depth_map.shape[1])
        y2 = min(y + h, depth_map.shape[0])
        
        # Extract depth region
        region = depth_map[y:y2, x:x2]
        
        # Get valid depths (non-zero)
        valid_depths = region[region > 0]
        
        if len(valid_depths) == 0:
            return None
        
        # Use median for robustness
        return float(np.median(valid_depths))
    
    def _estimate_size(
        self,
        bbox: Tuple[int, int, int, int],
        depth: float,
        focal_length: float
    ) -> Tuple[float, float]:
        """
        Estimate real-world object size from bounding box and depth.
        
        Using similar triangles:
            real_size = (pixel_size * depth) / focal_length
        
        Args:
            bbox: Bounding box in pixels
            depth: Object depth in meters
            focal_length: Camera focal length in pixels
            
        Returns:
            Tuple of (width, height) in meters
            
        Reference: Basic pinhole camera geometry
        https://en.wikipedia.org/wiki/Pinhole_camera_model
        """
        _, _, w, h = bbox
        
        real_width = (w * depth) / focal_length
        real_height = (h * depth) / focal_length
        
        return (real_width, real_height)
    
    def reset(self) -> None:
        """Reset detector state (for background subtraction)."""
        if self._bg_subtractor is not None:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )
        self._frame_count = 0


def draw_detections(
    image: np.ndarray,
    objects: List[DetectedObject],
    show_depth: bool = True,
    show_size: bool = True
) -> np.ndarray:
    """
    Draw detection boxes and information on image.
    
    Args:
        image: Input image (will be copied)
        objects: List of detected objects
        show_depth: Whether to show depth info
        show_size: Whether to show size estimate
        
    Returns:
        Image with drawn detections
    """
    output = image.copy()
    
    for obj in objects:
        x, y, w, h = obj.bbox
        
        # Color based on confidence (green = high, red = low)
        color = (
            int((1 - obj.confidence) * 255),  # Blue
            int(obj.confidence * 255),         # Green
            0                                   # Red
        )
        
        # Draw bounding box
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        
        # Draw center point
        cv2.circle(output, obj.center, 4, color, -1)
        
        # Build label text
        label_parts = [obj.label]
        
        if show_depth and obj.depth is not None:
            label_parts.append(f"{obj.depth:.2f}m")
        
        if show_size and obj.size_estimate is not None:
            sw, sh = obj.size_estimate
            label_parts.append(f"{sw:.2f}x{sh:.2f}m")
        
        label = " | ".join(label_parts)
        
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x, y - text_h - 4), (x + text_w, y), color, -1)
        
        # Draw label text
        cv2.putText(
            output, label,
            (x, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1
        )
    
    return output
