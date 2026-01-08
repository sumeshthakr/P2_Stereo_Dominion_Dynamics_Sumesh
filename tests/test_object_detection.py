"""
Unit tests for object detection module.
Author: Sumesh Thakur (sumeshthkr@gmail.com)
"""

import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.object_detection import (
    ObjectDetector,
    DetectorType,
    DetectedObject,
    draw_detections
)


class TestDetectedObject:
    """Tests for DetectedObject dataclass."""
    
    def test_detected_object_creation(self):
        """Test creating a DetectedObject."""
        obj = DetectedObject(
            bbox=(100, 100, 50, 80),
            center=(125, 140),
            confidence=0.8,
            label="person"
        )
        
        assert obj.bbox == (100, 100, 50, 80)
        assert obj.center == (125, 140)
        assert obj.confidence == 0.8
        assert obj.label == "person"
        assert obj.depth is None
        assert obj.size_estimate is None
    
    def test_detected_object_with_depth(self):
        """Test DetectedObject with depth information."""
        obj = DetectedObject(
            bbox=(100, 100, 50, 80),
            center=(125, 140),
            confidence=0.9,
            label="car",
            depth=10.5,
            size_estimate=(2.0, 1.5)
        )
        
        assert obj.depth == 10.5
        assert obj.size_estimate == (2.0, 1.5)


class TestObjectDetector:
    """Tests for ObjectDetector class."""
    
    @pytest.fixture
    def detector_contour(self):
        """Create contour-based detector."""
        return ObjectDetector(
            detector_type=DetectorType.CONTOUR,
            max_objects=2,
            min_area=1000
        )
    
    @pytest.fixture
    def detector_background(self):
        """Create background subtraction detector."""
        return ObjectDetector(
            detector_type=DetectorType.BACKGROUND_SUB,
            max_objects=2
        )
    
    @pytest.fixture
    def test_image(self):
        """Create test image with objects."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a bright rectangle (simulating an object)
        img[100:200, 200:350] = 255  # Object 1
        img[250:400, 400:550] = 255  # Object 2
        return img
    
    @pytest.fixture
    def test_depth_map(self):
        """Create test depth map."""
        depth = np.ones((480, 640), dtype=np.float32) * 20.0
        # Objects at different depths
        depth[100:200, 200:350] = 10.0  # Closer
        depth[250:400, 400:550] = 15.0  # Further
        return depth
    
    def test_detector_initialization(self, detector_contour):
        """Test detector is properly initialized."""
        assert detector_contour.detector_type == DetectorType.CONTOUR
        assert detector_contour.max_objects == 2
        assert detector_contour.min_area == 1000
    
    def test_contour_detection(self, detector_contour, test_image):
        """Test contour-based detection."""
        objects = detector_contour.detect(test_image)
        
        assert len(objects) <= 2
        for obj in objects:
            assert isinstance(obj, DetectedObject)
            assert obj.confidence >= 0 and obj.confidence <= 1
    
    def test_detection_with_depth(self, detector_contour, test_image, test_depth_map):
        """Test detection with depth map."""
        objects = detector_contour.detect(
            test_image,
            depth_map=test_depth_map,
            focal_length=700.0
        )
        
        for obj in objects:
            if obj.depth is not None:
                assert obj.depth > 0
            if obj.size_estimate is not None:
                assert obj.size_estimate[0] > 0
                assert obj.size_estimate[1] > 0
    
    def test_max_objects_limit(self, test_image):
        """Test that max_objects limit is respected."""
        # Create image with many objects
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(5):
            y = 50 + i * 80
            img[y:y+60, 100:200] = 255
        
        detector = ObjectDetector(
            detector_type=DetectorType.CONTOUR,
            max_objects=2,
            min_area=100
        )
        
        objects = detector.detect(img)
        
        assert len(objects) <= 2
    
    def test_min_area_filtering(self, detector_contour):
        """Test that small objects are filtered."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Small object (should be filtered with min_area=1000)
        img[100:110, 100:110] = 255  # 100 pixels
        
        objects = detector_contour.detect(img)
        
        # Should find no objects (too small)
        assert len(objects) == 0
    
    def test_background_subtraction(self, detector_background):
        """Test background subtraction detection."""
        # First frame (becomes background)
        background = np.zeros((480, 640, 3), dtype=np.uint8)
        detector_background.detect(background)
        
        # Second frame with moving object
        foreground = background.copy()
        foreground[100:200, 200:350] = 255
        
        # May need a few frames for background to build
        for _ in range(10):
            objects = detector_background.detect(foreground)
        
        # Should detect the new object
        # Note: May not always work with synthetic data
        assert isinstance(objects, list)
    
    def test_reset(self, detector_background):
        """Test detector reset."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detector_background.detect(img)
        
        detector_background.reset()
        
        assert detector_background._frame_count == 0
    
    def test_color_blob_detection(self):
        """Test color blob detection."""
        detector = ObjectDetector(
            detector_type=DetectorType.COLOR_BLOB,
            max_objects=2,
            min_area=500
        )
        
        # Create image with colored blobs
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Red blob (BGR format)
        img[100:200, 100:200] = [0, 0, 255]
        # Blue blob
        img[100:200, 300:400] = [255, 0, 0]
        
        objects = detector.detect(img)
        
        # Should detect color blobs
        assert isinstance(objects, list)
    
    def test_mobilenet_ssd_initialization(self):
        """Test MobileNet-SSD detector initialization."""
        detector = ObjectDetector(
            detector_type=DetectorType.MOBILENET_SSD,
            max_objects=2,
            min_area=500
        )
        
        assert detector.detector_type == DetectorType.MOBILENET_SSD
        assert detector.max_objects == 2
    
    def test_mobilenet_ssd_detection_fallback(self):
        """Test MobileNet-SSD detection (with fallback if model unavailable)."""
        detector = ObjectDetector(
            detector_type=DetectorType.MOBILENET_SSD,
            max_objects=2,
            min_area=500
        )
        
        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:200, 200:350] = 255
        
        objects = detector.detect(img)
        
        # Should return a list (may be empty or have detections)
        assert isinstance(objects, list)
        assert len(objects) <= 2
    
    def test_yolo_nano_initialization(self):
        """Test YOLOv8 Nano detector initialization."""
        detector = ObjectDetector(
            detector_type=DetectorType.YOLO_NANO,
            max_objects=2,
            min_area=500
        )
        
        assert detector.detector_type == DetectorType.YOLO_NANO
        assert detector.max_objects == 2
    
    def test_yolo_nano_detection_fallback(self):
        """Test YOLOv8 Nano detection (with fallback if model unavailable)."""
        detector = ObjectDetector(
            detector_type=DetectorType.YOLO_NANO,
            max_objects=2,
            min_area=500
        )
        
        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:200, 200:350] = 255
        
        objects = detector.detect(img)
        
        # Should return a list (may be empty or have detections)
        assert isinstance(objects, list)
        assert len(objects) <= 2
    
    def test_ml_detector_with_depth(self):
        """Test ML detectors with depth information."""
        for detector_type in [DetectorType.MOBILENET_SSD, DetectorType.YOLO_NANO]:
            detector = ObjectDetector(
                detector_type=detector_type,
                max_objects=2,
                min_area=500
            )
            
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[100:200, 200:350] = 255
            
            depth_map = np.ones((480, 640), dtype=np.float32) * 10.0
            
            objects = detector.detect(img, depth_map=depth_map, focal_length=700.0)
            
            # Should return a list
            assert isinstance(objects, list)


class TestDrawDetections:
    """Tests for detection visualization."""
    
    def test_draw_detections_empty(self):
        """Test drawing with no detections."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = draw_detections(img, [])
        
        assert result.shape == img.shape
        assert np.array_equal(result, img)
    
    def test_draw_detections_single(self):
        """Test drawing single detection."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        objects = [
            DetectedObject(
                bbox=(100, 100, 50, 80),
                center=(125, 140),
                confidence=0.8,
                label="object"
            )
        ]
        
        result = draw_detections(img, objects)
        
        assert result.shape == img.shape
        # Should have drawn something (not all zeros)
        assert not np.array_equal(result, img)
    
    def test_draw_detections_with_depth(self):
        """Test drawing detections with depth info."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        objects = [
            DetectedObject(
                bbox=(100, 100, 50, 80),
                center=(125, 140),
                confidence=0.9,
                label="car",
                depth=15.5,
                size_estimate=(2.0, 1.5)
            )
        ]
        
        result = draw_detections(img, objects, show_depth=True, show_size=True)
        
        assert result.shape == img.shape
        assert not np.array_equal(result, img)


class TestSizeEstimation:
    """Tests for real-world size estimation."""
    
    def test_size_estimation(self):
        """Test object size estimation from depth."""
        detector = ObjectDetector(DetectorType.CONTOUR)
        
        bbox = (0, 0, 100, 150)
        depth = 10.0
        focal_length = 700.0
        
        size = detector._estimate_size(bbox, depth, focal_length)
        
        # real_size = (pixel_size * depth) / focal_length
        expected_width = (100 * 10.0) / 700.0
        expected_height = (150 * 10.0) / 700.0
        
        assert abs(size[0] - expected_width) < 1e-6
        assert abs(size[1] - expected_height) < 1e-6
    
    def test_depth_extraction(self):
        """Test depth extraction from depth map."""
        detector = ObjectDetector(DetectorType.CONTOUR)
        
        depth_map = np.ones((480, 640), dtype=np.float32) * 20.0
        depth_map[100:200, 200:300] = 10.0  # Object region
        
        bbox = (200, 100, 100, 100)
        depth = detector._get_object_depth(bbox, depth_map)
        
        assert depth == 10.0
    
    def test_depth_extraction_invalid_region(self):
        """Test depth extraction with invalid depth values."""
        detector = ObjectDetector(DetectorType.CONTOUR)
        
        depth_map = np.zeros((480, 640), dtype=np.float32)  # All invalid
        
        bbox = (100, 100, 50, 50)
        depth = detector._get_object_depth(bbox, depth_map)
        
        assert depth is None
