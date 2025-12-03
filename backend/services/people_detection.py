"""
Detection Service for counting people in frames.

This module handles people counting using YOLO models.
"""

import logging
from typing import Tuple
import numpy as np
from models.model_manager import model_manager

logger = logging.getLogger(__name__)


def count_people_from_frame(frame: np.ndarray) -> int:
    """
    Count people in a given frame using the people model.
    
    Args:
        frame: Input image frame (numpy array)
    
    Returns:
        Number of people detected
    """
    if frame is None:
        logger.warning("Empty frame received for people counting")
        return 0
    
    # Get people model (lazy loaded)
    people_model = model_manager.get_model('people')
    if people_model is None:
        logger.warning("People model not loaded")
        return 0
    
    try:
        results = people_model(frame, conf=0.5, verbose=False)
        count = 0
        
        for res in results:
            if res.boxes is None:
                continue
            
            for box in res.boxes:
                cls = int(box.cls[0])
                class_name = people_model.names.get(cls, "unknown").lower()
                
                # Only count 'person' class
                if class_name == 'person':
                    count += 1
        
        logger.info(f"👥 People count: {count}")
        return count
    except Exception as e:
        logger.error(f"❌ Error counting people: {e}", exc_info=True)
        return 0


def detect_topview_people(frame: np.ndarray) -> Tuple[np.ndarray, list, int]:
    """
    Detect people using top-view model (best (9).pt).
    
    Args:
        frame: Input image frame
    
    Returns:
        Tuple of (annotated_frame, detections_list, people_count)
    """
    topview_model = model_manager.get_model('best9_topview')
    if topview_model is None:
        logger.warning("Top-view model not loaded")
        return frame, [], 0
    
    try:
        import cv2
        from datetime import datetime, timezone
        import uuid
        
        detections_to_save = []
        people_detected = 0
        annotated_frame = frame.copy()
        
        # 30% confidence threshold for top view
        CONF_THRESHOLD = 0.30
        
        results = topview_model(frame, conf=CONF_THRESHOLD, verbose=False)
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = topview_model.names[cls].lower()
                
                # Check for person/pedestrian classes
                if 'person' in class_name or 'pedestrian' in class_name:
                    people_detected += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw green box for top view people
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'TopView {conf:.2f}', (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    detection_id = str(uuid.uuid4())
                    detections_to_save.append({
                        "id": detection_id,
                        "detection_type": "person (topview)",
                        "model": "best (9).pt",
                        "confidence": conf,
                        "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                        "camera_id": "topview_cam",
                        "camera_name": "Top View",
                        "location": {"lat": 0, "lng": 0},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "alert_sent": False
                    })
        
        return annotated_frame, detections_to_save, people_detected
        
    except Exception as e:
        logger.error(f"❌ Error in top-view detection: {e}", exc_info=True)
        return frame, [], 0
