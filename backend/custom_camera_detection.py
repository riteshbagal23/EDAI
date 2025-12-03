"""
Custom Camera Detection Module
Provides camera-specific detection functions with customized model configurations
"""

import cv2
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Model paths
ROOT_DIR = Path(__file__).parent
BEST_PT_PATH = ROOT_DIR / "best.pt"
BEST_1_PT_PATH = ROOT_DIR / "best (1).pt"
BEST_9_PT_PATH = ROOT_DIR / "best (9).pt"
BEST_2_PT_PATH = ROOT_DIR / "best (2).pt"
YOLOV8N_PATH = ROOT_DIR / "yolov8n.pt"

# Load models
logger.info("Loading models for custom camera detection...")
gun_model = YOLO(str(BEST_PT_PATH)) if BEST_PT_PATH.exists() else None
gun_verify_model = YOLO(str(BEST_1_PT_PATH)) if BEST_1_PT_PATH.exists() else None
topview_model = YOLO(str(BEST_9_PT_PATH)) if BEST_9_PT_PATH.exists() else None
violence_model = YOLO(str(BEST_2_PT_PATH)) if BEST_2_PT_PATH.exists() else None
person_model = YOLO(str(YOLOV8N_PATH)) if YOLOV8N_PATH.exists() else None
logger.info("Custom camera detection models loaded")


def extract_region(frame, bbox, padding=20):
    """Extract a region from the frame based on bounding box with padding"""
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox['x1']) - padding)
    y1 = max(0, int(bbox['y1']) - padding)
    x2 = min(w, int(bbox['x2']) + padding)
    y2 = min(h, int(bbox['y2']) + padding)
    return frame[y1:y2, x1:x2]


def run_verification_model(cropped_region, conf_threshold=0.30):
    """Run best(1).pt on cropped region"""
    if gun_verify_model is None:
        return None
    
    try:
        results = gun_verify_model(cropped_region, conf=conf_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = gun_verify_model.names[cls].lower()
                
                if 'pistol' in class_name or 'gun' in class_name:
                    return {'verified': True, 'confidence': conf, 'class': class_name}
        
        return {'verified': False}
    except Exception as e:
        logger.error(f"Error in verification model: {e}")
        return None


def detect_house_medicine(frame, camera_id="house_medicine"):
    """
    Detection for House camera
    Uses: Person detection (yolov8n.pt) + Firearm detection (best.pt + best(1).pt) at 30%
    Shows all firearm types: Machinegun, Pistol, Rifle, Shotgun
    """
    detections_to_save = []
    people_detected = 0
    guns_detected = 0
    
    # Person detection
    if person_model:
        try:
            person_results = person_model(frame, conf=0.30, verbose=False)
            for result in person_results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = person_model.names[cls].lower()
                    
                    if class_name == 'person':
                        people_detected += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(frame, f'PERSON {conf:.2f}', (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        except Exception as e:
            logger.error(f"Person detection error: {e}")
    
    # Firearm detection with OR logic - showing ALL firearm classes
    guns_detected = _detect_firearms_all_classes(frame, camera_id, detections_to_save)
    
    return frame, detections_to_save, people_detected


def detect_medicine_only(frame, camera_id="medicine_cam"):
    """
    Detection for Medicine camera
    Uses: ONLY Firearm detection (best.pt + best(1).pt) at 30%
    Shows all firearm types: Machinegun, Pistol, Rifle, Shotgun
    No person detection
    """
    detections_to_save = []
    
    # Firearm detection with OR logic - showing ALL firearm classes
    guns_detected = _detect_firearms_all_classes(frame, camera_id, detections_to_save)
    
    return frame, detections_to_save, 0  # No people counting


def _detect_firearms_all_classes(frame, camera_id, detections_to_save):
    """
    Helper function to detect all firearm classes using OR logic
    Runs BOTH best.pt and best(1).pt independently for better coverage
    Returns: number of firearms detected
    """
    guns_detected = 0
    detected_regions = []  # Track detected regions to avoid duplicate annotations
    
    # STEP 1: Run best.pt at 30% confidence
    if gun_model:
        try:
            gun_results = gun_model(frame, conf=0.30, verbose=False)
            for result in gun_results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = gun_model.names[cls].lower()
                    
                    # Check for any firearm-related class
                    if any(weapon in class_name for weapon in ['pistol', 'gun', 'rifle', 'shotgun', 'machinegun']):
                        guns_detected += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, f'{class_name.upper()} (best.pt) {conf:.2f}', (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        detections_to_save.append({
                            "id": str(uuid.uuid4()),
                            "detection_type": class_name,
                            "model": "best.pt",
                            "confidence": conf,
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "camera_id": camera_id,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        detected_regions.append((x1, y1, x2, y2))
        except Exception as e:
            logger.error(f"best.pt detection error: {e}")
    
    # STEP 2: Run best(1).pt INDEPENDENTLY at 30% confidence
    # This ensures we catch firearms (especially shotguns) that best.pt might miss
    if gun_verify_model:
        try:
            verify_results = gun_verify_model(frame, conf=0.30, verbose=False)
            
            for result in verify_results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    firearm_type = gun_verify_model.names[cls]
                    
                    # best(1).pt classes: {0: 'Machinegun', 1: 'Pistol', 2: 'Rifle', 3: 'Shotgun'}
                    # Check if this detection significantly overlaps with existing ones
                    overlaps = False
                    for (dx1, dy1, dx2, dy2) in detected_regions:
                        # Simple overlap check
                        if not (x2 < dx1 or x1 > dx2 or y2 < dy1 or y1 > dy2):
                            overlaps = True
                            break
                    
                    # Only add if it doesn't overlap (avoid duplicate boxes)
                    if not overlaps:
                        guns_detected += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, f'{firearm_type.upper()} (best1.pt) {conf:.2f}',
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        detections_to_save.append({
                            "id": str(uuid.uuid4()),
                            "detection_type": firearm_type.lower(),
                            "model": "best (1).pt",
                            "confidence": conf,
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "camera_id": camera_id,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        detected_regions.append((x1, y1, x2, y2))
        except Exception as e:
            logger.error(f"best(1).pt detection error: {e}")
    
    return guns_detected


def detect_topview_people(frame, camera_id="topview"):
    """
    Detection for Top View camera
    Uses: best(9).pt at 40% confidence (same as upload)
    """
    detections_to_save = []
    people_detected = 0
    
    if topview_model:
        try:
            results = topview_model(frame, conf=0.40, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    people_detected += 1
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'PERSON {conf:.2f}', (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    detections_to_save.append({
                        "id": str(uuid.uuid4()),
                        "detection_type": "person",
                        "model": "best (9).pt",
                        "confidence": conf,
                        "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                        "camera_id": camera_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
        except Exception as e:
            logger.error(f"Top view detection error: {e}")
    
    return frame, detections_to_save, people_detected


def detect_gun_only(frame, camera_id="gun_cam"):
    """
    Detection for Gun/Gun1/Best(1) cameras
    Uses: best.pt + best(1).pt at 30% (OR logic)
    Shows all firearm types: Machinegun, Pistol, Rifle, Shotgun
    """
    detections_to_save = []
    
    # Firearm detection with OR logic - showing ALL firearm classes
    guns_detected = _detect_firearms_all_classes(frame, camera_id, detections_to_save)
    
    return frame, detections_to_save, 0  # No people counting


def detect_violence_only(frame, camera_id="violence_cam"):
    """
    Detection for Violence camera
    Uses: best(2).pt (violence classification) - shows all classes
    """
    detections_to_save = []
    violence_detected = 0
    
    if violence_model:
        try:
            results = violence_model.predict(frame, imgsz=224, verbose=False)
            if results:
                res = results[0]
                
                # Get all class probabilities
                if hasattr(res, "probs") and res.probs is not None:
                    probs_obj = res.probs
                    
                    # Get top prediction
                    top1_label = None
                    top1_conf = 0.0
                    
                    if hasattr(probs_obj, "top1") and probs_obj.top1 is not None:
                        top1_idx = int(probs_obj.top1)
                        top1_conf = float(probs_obj.top1conf) if hasattr(probs_obj, "top1conf") else 0.0
                        
                        if hasattr(res, "names"):
                            top1_label = res.names[top1_idx]
                    
                    # Display classification result
                    if top1_label:
                        # Determine color based on class
                        if top1_label.lower() == 'violence':
                            color = (0, 0, 255)  # Red for violence
                            violence_detected = 1
                        else:
                            color = (0, 255, 0)  # Green for non-violence
                        
                        # Display main prediction
                        label = f"{top1_label.upper()}: {top1_conf:.2%}"
                        cv2.putText(frame, label, (10, 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                        
                        # Get all class probabilities for detailed display
                        if hasattr(probs_obj, "data") and hasattr(res, "names"):
                            y_offset = 80
                            for idx, prob in enumerate(probs_obj.data):
                                class_name = res.names[idx]
                                conf_val = float(prob)
                                
                                # Display each class with its probability
                                class_label = f"{class_name}: {conf_val:.2%}"
                                text_color = (255, 255, 255)  # White for other classes
                                cv2.putText(frame, class_label, (10, y_offset),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                                y_offset += 30
                        
                        # Save detection
                        img_h, img_w = frame.shape[:2]
                        detections_to_save.append({
                            "id": str(uuid.uuid4()),
                            "detection_type": top1_label.lower(),
                            "model": "best (2).pt",
                            "confidence": top1_conf,
                            "bbox": {"x1": 0, "y1": 0, "x2": img_w, "y2": img_h},
                            "camera_id": camera_id,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
        except Exception as e:
            logger.error(f"Violence detection error: {e}")
    
    return frame, detections_to_save, 0  # No people counting
