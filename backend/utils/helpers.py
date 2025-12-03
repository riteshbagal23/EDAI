"""
Helper utilities for detection validation and image processing.

This module contains utility functions used across the detection pipeline.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


def validate_detection(
    detection_type: str,
    conf: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int
) -> bool:
    """
    Validate detection bounding box to filter out invalid detections.
    
    Args:
        detection_type: Type of detection (e.g., 'pistol', 'knife')
        conf: Confidence score (0-1)
        x1, y1, x2, y2: Bounding box coordinates
        img_width, img_height: Image dimensions
    
    Returns:
        True if detection is valid, False otherwise
    """
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    img_area = img_width * img_height
    area_ratio = bbox_area / img_area if img_area > 0 else 0
    
    # Filter out unreasonably small boxes
    if bbox_width < 10 or bbox_height < 10:
        logger.debug(f"❌ Detection too small: {bbox_width}x{bbox_height}")
        return False
    
    # Filter out boxes that cover most of the image (likely false positives)
    if area_ratio > 0.8:
        logger.debug(f"❌ Detection too large: {area_ratio:.2%} of image")
        return False
    
    return True


def extract_region(frame: np.ndarray, bbox: Dict, padding: int = 20) -> np.ndarray:
    """
    Extract a region from the frame based on bounding box with padding.
    
    Args:
        frame: Input image (numpy array)
        bbox: Dict with keys 'x1', 'y1', 'x2', 'y2'
        padding: Pixels to add around bbox (default: 20)
    
    Returns:
        Cropped region as numpy array
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox['x1']) - padding)
    y1 = max(0, int(bbox['y1']) - padding)
    x2 = min(w, int(bbox['x2']) + padding)
    y2 = min(h, int(bbox['y2']) + padding)
    
    return frame[y1:y2, x1:x2]


def preprocess_thermal_image(frame: np.ndarray) -> np.ndarray:
    """
    Enhance thermal images for better gun detection.
    
    Applies:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Normalization
    - Sharpening
    
    Args:
        frame: Input thermal image
    
    Returns:
        Preprocessed image
    """
    try:
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Normalize pixel values
        normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        # Sharpen for edge enhancement
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(normalized, -1, kernel)
        
        # Convert back to BGR for YOLO
        preprocessed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        logger.debug("✅ Thermal image preprocessing completed")
        return preprocessed
    except Exception as e:
        logger.error(f"❌ Error in preprocessing: {e}")
        return frame


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Dict,
    label: str,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a bounding box with label on the frame.
    
    Args:
        frame: Input image
        bbox: Dict with 'x1', 'y1', 'x2', 'y2'
        label: Text label to display
        color: BGR color tuple (default: red)
        thickness: Line thickness (default: 2)
    
    Returns:
        Frame with bounding box drawn
    """
    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness
    )
    cv2.rectangle(
        frame,
        (x1, y1 - label_height - 10),
        (x1 + label_width, y1),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        frame,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        thickness
    )
    
    return frame


def resize_frame(frame: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input image
        max_width: Maximum width
        max_height: Maximum height
    
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / w, max_height / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return frame


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Dicts with 'x1', 'y1', 'x2', 'y2'
    
    Returns:
        IoU value (0-1)
    """
    # Calculate intersection area
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    # Return IoU
    return intersection / union if union > 0 else 0.0
