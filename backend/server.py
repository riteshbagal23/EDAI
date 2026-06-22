from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Query, Body, Form, BackgroundTasks, APIRouter, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import os
import logging
import uuid
from datetime import datetime, timezone
import cv2
import numpy as np
from ultralytics import YOLO
import aiofiles
import hashlib
import json
import threading
import time
import io
import base64
from io import BytesIO
from queue import Queue
from inference_sdk import InferenceHTTPClient
import requests
import pymongo
from functools import lru_cache
from math import radians, sin, cos, sqrt, atan2

# Import camera manager for multi-camera support
# Import camera manager for multi-camera support
from camera_manager import camera_registry, CameraType, CameraStatus
from ipfs_manager import IPFSManager
import custom_camera_detection

# Initialize IPFS Manager
ipfs = IPFSManager()

# Initialize Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="2J1WkKz6yPss5qhRiHuE"
)

# Base paths and env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Setup logging FIRST
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import performance configuration
from performance_config import get_config, get_resolution_tuple, get_frame_delay

# Load performance settings
PERF_CONFIG = get_config()
logger.info(f"🔧 Performance Mode: {PERF_CONFIG.get('description', 'Custom')}")
logger.info(f"   FPS: {PERF_CONFIG['fps']}, Resolution: {PERF_CONFIG['resolution']}")
logger.info(f"   People Counting: {PERF_CONFIG['enable_people_counting']}, Thermal Detection: {PERF_CONFIG['enable_thermal_detection']}")
logger.info(f"   Process Every N Frames: {PERF_CONFIG['process_every_n_frames']}")

# MongoDB
mongo_url = os.environ.get('MONGO_URL')
db_name = os.environ.get('DB_NAME')
if not mongo_url:
    raise ValueError("MONGO_URL environment variable is not set. Please set it in your .env file.")
if not db_name:
    raise ValueError("DB_NAME environment variable is not set. Please set it in your .env file.")
client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

# Dirs
UPLOAD_DIR = ROOT_DIR / "uploads"
DETECTIONS_DIR = ROOT_DIR / "detections"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Models
MODEL_PATH = ROOT_DIR / 'best.pt'
BEST1_MODEL_PATH = ROOT_DIR / 'best (1).pt'
PEOPLE_MODEL_PATH = ROOT_DIR / 'yolov8n.pt'
VIOLENCE_MODEL_PATH = ROOT_DIR / 'best (2).pt'
THERMAL_MODEL_PATH = ROOT_DIR / 'thermal.pt'
TOPVIEW_MODEL_PATH = ROOT_DIR / 'best (9).pt'  # Top-view human detection (30% confidence)

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    logger.info(f"Loading YOLO model from {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    logger.info(f"Model loaded successfully. Classes: {model.names}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Load best (1).pt model for verification (not continuous detection)
best1_model = None
try:
    if BEST1_MODEL_PATH.exists():
        logger.info(f"Loading best (1) model from {BEST1_MODEL_PATH} (for verification only)")
        best1_model = YOLO(str(BEST1_MODEL_PATH))
        logger.info(f"Best (1) model loaded successfully. Classes: {best1_model.names}")
    else:
        logger.warning(f"Best (1) model not found at {BEST1_MODEL_PATH}")
except Exception as e:
    logger.warning(f"Failed to load best (1) model: {e}")

people_model = None
if PERF_CONFIG['enable_people_counting']:
    try:
        if PEOPLE_MODEL_PATH.exists():
            logger.info(f"Loading YOLOv8m model for people counting from {PEOPLE_MODEL_PATH}")
            people_model = YOLO(str(PEOPLE_MODEL_PATH))
            logger.info(f"People counting model loaded successfully. Classes: {people_model.names}")
        else:
            logger.warning("People counting model not found; people counting disabled")
    except Exception as e:
        logger.warning(f"Failed to load people model: {e}")
else:
    logger.info("ℹ️ People counting disabled (performance optimization)")

# Load violence detection model
violence_model = None
try:
    if VIOLENCE_MODEL_PATH.exists():
        logger.info(f"Loading violence model from {VIOLENCE_MODEL_PATH}")
        violence_model = YOLO(str(VIOLENCE_MODEL_PATH))
        logger.info(f"Violence model loaded successfully. Classes: {violence_model.names}")
    else:
        logger.warning(f"Violence model not found at {VIOLENCE_MODEL_PATH}")
except Exception as e:
    logger.warning(f"Failed to load violence model: {e}")

# Load thermal detection model (thermal.pt)
thermal_model = None
# try:
#     if THERMAL_MODEL_PATH.exists():
#         logger.info(f"Loading thermal model from {THERMAL_MODEL_PATH}")
#         thermal_model = YOLO(str(THERMAL_MODEL_PATH))
#         logger.info(f"Thermal model loaded successfully. Classes: {thermal_model.names}")
#     else:
#         logger.warning(f"Thermal model not found at {THERMAL_MODEL_PATH}")
# except Exception as e:
#     logger.warning(f"Failed to load thermal model: {e}")

# Load drone detection model  
DRONE_MODEL_PATH = ROOT_DIR / 'drone.pt'
drone_model = None
# try:
#     if DRONE_MODEL_PATH.exists():
#         logger.info(f"Loading drone model from {DRONE_MODEL_PATH}")
#         drone_model = YOLO(str(DRONE_MODEL_PATH))
#         logger.info(f"Drone model loaded successfully. Classes: {drone_model.names}")
#     else:
#         logger.warning(f"Drone model not found at {DRONE_MODEL_PATH}")
# except Exception as e:
#     logger.warning(f"Failed to load drone model: {e}")

# Load thermal human detection model
THERMAL_HUMAN_MODEL_PATH = ROOT_DIR / 'thermalhuman.pt'
thermal_human_model = None
# try:
#     if THERMAL_HUMAN_MODEL_PATH.exists():
#         logger.info(f"Loading thermal human model from {THERMAL_HUMAN_MODEL_PATH}")
#         thermal_human_model = YOLO(str(THERMAL_HUMAN_MODEL_PATH))
#         logger.info(f"Thermal human model loaded successfully. Classes: {thermal_human_model.names}")
#     else:
#         logger.warning(f"Thermal human model not found at {THERMAL_HUMAN_MODEL_PATH}")
# except Exception as e:
#     logger.warning(f"Failed to load thermal human model: {e}")

# Load top-view human detection model (best (9).pt)
topview_model = None
TOPVIEW_MODEL_PATH = ROOT_DIR / "best (9).pt"
# try:
#     if TOPVIEW_MODEL_PATH.exists():
#         logger.info(f"Loading top-view model from {TOPVIEW_MODEL_PATH}")
#         topview_model = YOLO(str(TOPVIEW_MODEL_PATH))
#         logger.info(f"Top-view model loaded successfully. Classes: {topview_model.names}")
#     else:
#         logger.warning(f"Top-view model not found at {TOPVIEW_MODEL_PATH}")
# except Exception as e:
#     logger.warning(f"Failed to load top-view model: {e}")

# Load best (8).pt model for additional gun detection
best8_model = None
BEST8_MODEL_PATH = ROOT_DIR / "best (8).pt"
try:
    if BEST8_MODEL_PATH.exists():
        best8_model = YOLO(str(BEST8_MODEL_PATH))
        logger.info(f"Best (8) model loaded successfully. Classes: {best8_model.names}")
    else:
        logger.warning(f"Best (8) model not found at {BEST8_MODEL_PATH}")
except Exception as e:
    logger.warning(f"Failed to load best (8) model: {e}")

# Load best (10).pt model for dual primary verification (with best.pt)
best10_model = None
BEST10_MODEL_PATH = ROOT_DIR / "best (10).pt"
try:
    if BEST10_MODEL_PATH.exists():
        best10_model = YOLO(str(BEST10_MODEL_PATH))
        logger.info(f"✅ Best (10) model loaded successfully. Classes: {best10_model.names}")
    else:
        logger.warning(f"Best (10) model not found at {BEST10_MODEL_PATH}")
except Exception as e:
    logger.warning(f"Failed to load best (10) model: {e}")

# Webcam capture globals
camera = None
is_running = False
detection_stats = {
    'guns': 0,
    'knives': 0,
    'thermal_guns': 0,
    'people': 0,
    'alert': False,
    'alert_message': '',
    'last_alert_time': None,
    'latest_ipfs': None  # Store latest IPFS data for frontend
}

# IPFS upload cooldown (15 seconds)
last_ipfs_upload_time = None
ipfs_upload_cooldown = 15.0  # seconds

frame_lock = threading.Lock()
current_frame = None
alert_cooldown = 30  # seconds between alerts

# Detection queue for background saving
detection_queue = Queue()
save_worker_thread = None
save_worker_running = False

# Dual verification queue for two-stage detections
dual_verification_queue = Queue()
verification_worker_thread = None
verification_worker_running = False

# Verification cooldown to prevent spam (15 seconds)
last_verification_time = None
verification_cooldown = 15  # seconds between verification submissions


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        width, height = get_resolution_tuple(PERF_CONFIG['resolution'])
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.video.set(cv2.CAP_PROP_FPS, PERF_CONFIG['fps'])
        self.frame_counter = 0  # For frame skipping
        
    def __del__(self):
        if self.video.isOpened():
            self.video.release()
    
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        self.frame_counter += 1
        return frame
    
    def should_process_frame(self):
        """Check if current frame should be processed based on frame skipping config"""
        return (self.frame_counter % PERF_CONFIG['process_every_n_frames']) == 0

# Helpers
from twilio.rest import Client

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
EMERGENCY_CONTACTS = os.environ.get('EMERGENCY_CONTACTS', '').split(',')

twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("✅ Twilio Client initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize Twilio Client: {e}")

from utils.alerts import send_twilio_alert




def count_people_from_frame(frame) -> int:
    """Count people in frame"""
    if people_model is None:
        logger.warning("People model not loaded")
        return 0
    
    if frame is None:
        logger.warning("Empty frame received for people counting")
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

def preprocess_thermal_image(frame):
    """Enhance thermal images for better gun detection"""
    try:
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Normalize pixel values
        normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        # Slight sharpening for edge enhancement
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(normalized, -1, kernel)
        
        # Convert back to BGR for YOLO
        preprocessed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        logger.info("✅ Thermal image preprocessing completed")
        return preprocessed
    except Exception as e:
        logger.error(f"❌ Error in preprocessing: {e}")
        return frame

def detect_thermal_guns(frame, preprocess: bool = True) -> tuple:
    """Detect thermal guns using local YOLO model (thermal.pt) and draw bounding boxes"""
    THERMAL_GUN_CONFIDENCE_THRESHOLD = 0.20  # 20% confidence threshold
    
    if frame is None:
        logger.warning("Empty frame received for thermal gun detection")
        return 0, [], frame

    if thermal_model is None:
        logger.warning("Thermal model not loaded")
        return 0, [], frame

    try:
        thermal_guns_detected = 0
        detections_to_save = []
        annotated_frame = frame.copy()
        img_h, img_w = frame.shape[:2]
        
        # Apply preprocessing if enabled
        if preprocess:
            logger.info("🔧 Applying thermal image preprocessing...")
            processed_frame = preprocess_thermal_image(frame)
        else:
            logger.info("⏭️ Skipping preprocessing")
            processed_frame = frame
        
        logger.info(f"🌡️ Running thermal model (thermal.pt) with {THERMAL_GUN_CONFIDENCE_THRESHOLD*100:.0f}% threshold...")
        results = thermal_model(processed_frame, conf=THERMAL_GUN_CONFIDENCE_THRESHOLD, verbose=False)
        
        # DEBUG: Log what the model sees
        total_boxes = 0
        for result in results:
            if result.boxes is not None:
                total_boxes += len(result.boxes)
        logger.info(f"🔍 DEBUG: Thermal model returned {total_boxes} total boxes")
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get class name if available
                class_name = thermal_model.names.get(cls, "unknown")
                
                # DEBUG: Log every detection
                logger.info(f"🔍 DEBUG: Class={cls}, Name='{class_name}', Conf={conf:.3f}")
                
                thermal_guns_detected += 1
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                logger.info(f"🔫 THERMAL GUN DETECTED with confidence {conf:.2f}")
                
                # Draw bounding box - Magenta
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(annotated_frame, f'THERMAL {conf:.2f}', (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                detection_id = str(uuid.uuid4())
                detections_to_save.append({
                    "id": detection_id,
                    "detection_type": "thermal_gun",
                    "model": "thermal.pt",
                    "confidence": conf,
                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "camera_id": "webcam",
                    "camera_name": "Live Webcam",
                    "location": {"lat": 0, "lng": 0},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alert_sent": False
                })
        
        if thermal_guns_detected == 0:
            logger.info("✅ No thermal guns detected")
        
        logger.info(f"🌡️ Total thermal guns detected: {thermal_guns_detected}")
        return thermal_guns_detected, detections_to_save, annotated_frame
    except Exception as e:
        logger.error(f"❌ Error in thermal gun detection: {e}", exc_info=True)
        return 0, [], frame

def detect_topview_people(frame) -> tuple:
    """Detect people using top-view model (best (9).pt)"""
    if topview_model is None:
        logger.warning("Top-view model not loaded")
        return frame, [], 0
    
    try:
        detections_to_save = []
        people_detected = 0
        annotated_frame = frame.copy()
        
        # Default confidence for top view (30% as per user request to keep same)
        # Assuming default was 30% or 40% based on previous code snippets
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
                        "camera_id": "topview_cam", # Will be overwritten by caller if needed
                        "camera_name": "Top View",
                        "location": {"lat": 0, "lng": 0},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "alert_sent": False
                    })
        
        return annotated_frame, detections_to_save, people_detected
        
    except Exception as e:
        logger.error(f"❌ Error in top-view detection: {e}", exc_info=True)
        return frame, [], 0


def validate_detection(detection_type: str, conf: float, x1: float, y1: float, x2: float, y2: float,
                       img_width: int, img_height: int) -> bool:
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    bbox_area = bbox_width * bbox_height
    img_area = img_width * img_height
    area_ratio = bbox_area / img_area if img_area > 0 else 0
    # Relaxed validation - only check for unreasonably small boxes
    if bbox_width < 10 or bbox_height < 10:
        return False
    # Allow larger detections (up to 80% of image)
    if area_ratio > 0.8:
        return False
    return True


def extract_region(frame, bbox, padding=20):
    """
    Extract a region from the frame based on bounding box with padding
    
    Args:
        frame: Input image
        bbox: Dict with x1, y1, x2, y2
        padding: Pixels to add around bbox
    
    Returns:
        Cropped region as numpy array
    """
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox['x1']) - padding)
    y1 = max(0, int(bbox['y1']) - padding)
    x2 = min(w, int(bbox['x2']) + padding)
    y2 = min(h, int(bbox['y2']) + padding)
    
    return frame[y1:y2, x1:x2]


def run_verification_model(cropped_region, conf_threshold=0.60):
    """
    Run best (1).pt on cropped region for verification
    
    Args:
        cropped_region: Cropped image region
        conf_threshold: Minimum confidence threshold
    
    Returns:
        Dict with verification result or None
    """
    if best1_model is None:
        logger.warning("⚠️ Verification model (best (1).pt) not loaded")
        return None
    
    try:
        results = best1_model(cropped_region, conf=conf_threshold, verbose=False)
        
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            # Get highest confidence detection
            best_box = None
            best_conf = 0.0
            
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = best1_model.names[cls].lower()
                
                # Check if it's a gun/pistol detection
                if any(keyword in class_name for keyword in ['pistol', 'gun', 'knife', 'rifle', 'shotgun', 'weapon']):
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box
            
            if best_box is not None and best_conf >= conf_threshold:
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(float)
                return {
                    'confidence': best_conf,
                    'bbox': {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)},
                    'verified': True
                }
        
        return None
    except Exception as e:
        logger.error(f"❌ Error in verification model: {e}")
        return None


def handle_dual_verification(frame, detection, verification, camera_id, camera_name, location, db):
    """
    Handle detections verified by both models
    
    Actions:
    1. Send immediate Twilio call
    2. Create admin alert (pending_verifications)
    3. Save to blockchain with PENDING status
    4. Save annotated images
    """
    try:
        logger.info(f"🎯 handle_dual_verification called for {camera_name}")
        verification_id = str(uuid.uuid4())
        detection_id = str(uuid.uuid4())
        
        # Determine status based on verification result
        is_verified = verification.get('verified', False)
        det_type = detection.get('detection_type', 'pistol')
        
        logger.info(f"📊 Detection: {det_type}, Verified: {is_verified}, Conf: {detection['confidence']:.2f}")
        
        # Save full frame
        full_frame_fname = f"dual_full_{detection_id}.jpg"
        full_frame_path = DETECTIONS_DIR / full_frame_fname
        cv2.imwrite(str(full_frame_path), frame)
        logger.info(f"💾 Saved full frame: {full_frame_fname}")
        
        # Save cropped region
        cropped_region = extract_region(frame, detection['bbox'])
        cropped_fname = f"dual_crop_{detection_id}.jpg"
        cropped_path = DETECTIONS_DIR / cropped_fname
        cv2.imwrite(str(cropped_path), cropped_region)
        logger.info(f"💾 Saved cropped region: {cropped_fname}")
        
        # Save annotated image with both model results
        annotated = frame.copy()
        bbox = detection['bbox']
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        
        # Draw bbox (Red if unverified, Green if verified)
        color = (0, 255, 0) if is_verified else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        
        # Add labels
        label1 = f"best.pt: {detection['confidence']:.2f}"
        label2 = f"best(1).pt: {verification.get('confidence', 0.0):.2f}"
        cv2.putText(annotated, label1, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated, label2, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_verified else (0, 0, 255), 2)
        
        annotated_fname = f"dual_annotated_{detection_id}.jpg"
        annotated_path = DETECTIONS_DIR / annotated_fname
        cv2.imwrite(str(annotated_path), annotated)
        logger.info(f"💾 Saved annotated image: {annotated_fname}")
        
        # Send Twilio call immediately (if not already sent in Stage 1)
        # Note: We are now sending it in Stage 1, so this might be redundant or a fallback.
        # But let's keep it safe.
        avg_confidence = (detection['confidence'] + verification.get('confidence', 0.0)) / 2
        status_str = "VERIFIED" if is_verified else "UNVERIFIED"
        twilio_message = f"{status_str} ALERT: {det_type} detected. Conf: {avg_confidence:.1%}. Location: {location.get('lat', 0)}, {location.get('lng', 0)}"
        
        twilio_sid = None
        try:
            # Send Twilio alert (only if verified? User said "Stage 1 or 2", so maybe always?)
            # But we already sent it in Stage 1! 
            # Let's NOT send it here to avoid duplicate calls, OR check a flag.
            # For now, we'll assume Stage 1 handled it.
            pass 
        except Exception as e:
            logger.error(f"Failed to send Twilio alert: {e}")
        
        # Create pending verification record
        verification_doc = {
            "id": verification_id,
            "detection_id": detection_id,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bbox": detection['bbox'],
            "best_pt_confidence": detection['confidence'],
            "best1_pt_confidence": verification.get('confidence', 0.0),
            "full_frame_path": f"/detections/{full_frame_fname}",
            "cropped_region_path": f"/detections/{cropped_fname}",
            "annotated_image_path": f"/detections/{annotated_fname}",
            "location": location,
            "status": "pending",
            "admin_decision": None,

            "blockchain_verified": False,
            "twilio_call_sent": True, # Assumed sent in Stage 1
            "twilio_call_sid": "stage1_sent"
        }
        
        logger.info(f"📝 Inserting verification record to pending_verifications: {verification_id}")
        result = db.pending_verifications.insert_one(verification_doc)
        logger.info(f"✅ Inserted to pending_verifications with _id: {result.inserted_id}")
        
        # Create blockchain entry with PENDING status
        blockchain_data = {
            **verification_doc,
            "verification_status": "pending_admin_review"
        }

        
        # Update verification with blockchain hash
        db.pending_verifications.update_one(
            {"id": verification_id},
            {"$set": {"blockchain_hash": "pending"}} # Placeholder
        )
        
        # Also save to detections collection
        detection_doc = {
            "id": detection_id,
            "verification_id": verification_id,
            "detection_type": det_type,
            "model": "best.pt + best (1).pt",
            "confidence": avg_confidence,
            "bbox": detection['bbox'],
            "image_path": f"/detections/{annotated_fname}",
            "camera_id": camera_id,
            "camera_name": camera_name,
            "location": location,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verification_status": "dual_verified" if is_verified else "unverified",
            "requires_admin_review": True,

            "alert_sent": True
        }
        
        logger.info(f"📝 Inserting detection record to detections collection")
        det_result = db.detections.insert_one(detection_doc)
        logger.info(f"✅ Inserted to detections with _id: {det_result.inserted_id}")
        
        logger.info(f"✅ Dual verification saved successfully: {verification_id}")
        
        return verification_id
        
    except Exception as e:
        logger.error(f"❌ Error in handle_dual_verification: {e}", exc_info=True)
        return None


def log_single_model_detection(detection):
    """Log detections that only one model detected (for analysis)"""
    logger.info(f"ℹ️ Single-model detection: {detection['detection_type']} @ {detection['confidence']:.2f}")


def detect_weapons_and_people(frame, gun_conf_threshold=0.80, verification_conf_threshold=0.60, use_or_logic=False, use_violence_model=False, camera_id="webcam", camera_type=CameraType.WEBCAM):
    """Detect weapons and count people in frame"""
    global detection_stats
    global last_verification_time
    
    guns_detected = 0
    knives_detected = 0
    violence_detected = 0
    thermal_guns_detected = 0
    people_detected = 0
    detections_to_save = []
    
    # Detect weapons (guns and knives) - TWO-STAGE VERIFICATION FOR GUNS
    # Use a lower base threshold for the initial sweep to ensure we catch candidates
    base_conf = min(0.50, gun_conf_threshold)
    weapon_results = model(frame, conf=base_conf, verbose=False)

    # Process weapon detections (guns / knives)
    for result in weapon_results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class name
            class_name = model.names[cls].lower()

            # Handle GUN detections with DUAL VERIFICATION
            if 'pistol' in class_name or 'gun' in class_name:
                # Stage 1: Primary detection by best.pt
                
                # If using OR logic, we accept lower confidence from best.pt as a valid detection
                # OR we verify with best(1).pt
                
                if use_or_logic:
                    # Relaxed Logic: if best.pt > threshold OR best(1).pt > threshold
                    
                    # Case 1: best.pt is confident enough
                    if conf >= gun_conf_threshold:
                        logger.warning(f"🚨 OR Logic: Gun detected by best.pt @ {conf:.2f}")
                        guns_detected += 1
                        
                        # Draw bounding box
                        color = (0, 0, 255) # Red
                        label = f'GUN (best.pt) {conf:.2f}'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Save detection
                        detection_id = str(uuid.uuid4())
                        detections_to_save.append({
                            "id": detection_id,
                            "detection_type": "pistol",
                            "model": "best.pt",
                            "confidence": conf,
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "camera_id": camera_id,
                            "camera_name": "Live Webcam" if camera_type == CameraType.WEBCAM else f"Camera {camera_id}",
                            "location": {"lat": 0, "lng": 0},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "alert_sent": False
                        })
                        continue

                    # Case 2: best.pt is weak, try best(1).pt
                    # We need a minimum base confidence to even bother checking
                    if conf < 0.10: 
                        continue
                        
                    bbox_dict = {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}
                    cropped_region = extract_region(frame, bbox_dict)
                    
                    # Run verification with the same threshold
                    verification_result = run_verification_model(cropped_region, conf_threshold=verification_conf_threshold)
                    
                    if verification_result and verification_result['verified']:
                        logger.warning(f"🚨 OR Logic: Gun detected by best(1).pt @ {verification_result['confidence']:.2f}")
                        guns_detected += 1
                        
                        color = (0, 255, 0) # Green
                        label = f'GUN (best1.pt) {verification_result["confidence"]:.2f}'
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Save detection
                        detection_id = str(uuid.uuid4())
                        detections_to_save.append({
                            "id": detection_id,
                            "detection_type": "pistol",
                            "model": "best (1).pt",
                            "confidence": verification_result['confidence'],
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "camera_id": camera_id,
                            "camera_name": "Live Webcam" if camera_type == CameraType.WEBCAM else f"Camera {camera_id}",
                            "location": {"lat": 0, "lng": 0},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "alert_sent": False
                        })
                    continue

                # Standard Dual Verification Logic (AND Logic)
                if conf < gun_conf_threshold:
                    continue
                
                logger.info(f"🔴 Stage 1: GUN detected by best.pt @ {conf:.2f}")
                
                # IMMEDIATE ALERT ON STAGE 1 (User Request)
                # We draw the box and send alert NOW, regardless of verification
                
                # Prepare detection data
                detection_id = str(uuid.uuid4())
                detection_data = {
                    "id": detection_id,
                    "detection_type": "pistol",
                    "model": "best.pt",
                    "confidence": conf,
                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "camera_id": camera_id,
                    "camera_name": "Live Webcam" if camera_type == CameraType.WEBCAM else f"Camera {camera_id}",
                    "location": {"lat": 0, "lng": 0},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alert_sent": True,
                    "verification_status": "unverified"
                }
                
                # Trigger Alert Immediately
                try:
                    logger.info(f"🚨 Triggering Immediate Alert for Gun (Stage 1)...")
                    send_twilio_alert(detection_data)
                except Exception as e:
                    logger.error(f"❌ Failed to send gun alert: {e}")

                # Draw Initial Box (Red)
                color = (0, 0, 255) # Red for unverified
                label = f'GUN {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                detections_to_save.append(detection_data)

                # Stage 2: Verification (Optional for Alert, but used for Admin Queue)
                bbox_dict = {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}
                cropped_region = extract_region(frame, bbox_dict)
                
                verification_result = run_verification_model(cropped_region, conf_threshold=verification_conf_threshold)
                
                if verification_result and verification_result['verified']:
                    # Update to Verified Status
                    logger.warning(f"✅ Stage 2: DUAL VERIFICATION SUCCESS! best(1).pt @ {verification_result['confidence']:.2f}")
                    guns_detected += 1
                    
                    # Update visual to Green
                    color = (0, 255, 0)
                    label = f'VERIFIED GUN {conf:.2f}/{verification_result["confidence"]:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Queue for Admin
                    current_time = time.time()
                    if last_verification_time is None or (current_time - last_verification_time) >= verification_cooldown:
                        if camera_type == CameraType.WEBCAM:
                            verification_data = {
                                'frame': frame.copy(),
                                'detection': {'confidence': conf, 'bbox': bbox_dict, 'detection_type': 'pistol'},
                                'verification': verification_result,
                                'camera_id': camera_id,
                                'camera_name': "Live Webcam",
                                'location': {'lat': 0, 'lng': 0}
                            }
                            dual_verification_queue.put(verification_data)
                            last_verification_time = current_time
                else:
                    logger.info(f"⚠️ Stage 2: best(1).pt FAILED verification - Sending to Admin as UNVERIFIED")
                    # Queue for Admin
                    current_time = time.time()
                    if last_verification_time is None or (current_time - last_verification_time) >= verification_cooldown:
                        if camera_type == CameraType.WEBCAM:
                            verification_data = {
                                'frame': frame.copy(),
                                'detection': {'confidence': conf, 'bbox': bbox_dict, 'detection_type': 'pistol'},
                                'verification': {'verified': False, 'confidence': 0.0}, # Failed verification
                                'camera_id': camera_id,
                                'camera_name': "Live Webcam",
                                'location': {'lat': 0, 'lng': 0}
                            }
                            logger.info(f"📤 Queueing UNVERIFIED detection to admin panel (conf: {conf:.2f})")
                            dual_verification_queue.put(verification_data)
                            last_verification_time = current_time
                            logger.info(f"✅ Detection added to verification queue. Queue size: {dual_verification_queue.qsize()}")
            elif 'knife' in class_name:
                # KNIFE detection
                if conf < 0.80:
                    continue
                
                logger.info(f"🔪 Stage 1: KNIFE detected by best.pt @ {conf:.2f}")
                
                # IMMEDIATE ALERT ON STAGE 1
                detection_id = str(uuid.uuid4())
                detection_data = {
                    "id": detection_id,
                    "detection_type": "knife",
                    "confidence": conf,
                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "camera_id": "webcam",
                    "camera_name": "Live Webcam",
                    "location": {"lat": 0, "lng": 0},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alert_sent": True,
                    "verification_status": "unverified"
                }
                
                try:
                    logger.info(f"🚨 Triggering Immediate Alert for Knife (Stage 1)...")
                    send_twilio_alert(detection_data)
                except Exception as e:
                    logger.error(f"❌ Failed to send knife alert: {e}")

                # Draw Initial Box (Red)
                color = (0, 0, 255)
                label = f'KNIFE {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                detections_to_save.append(detection_data)

                # Stage 2: Verification (Optional)
                bbox_dict = {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}
                cropped_region = extract_region(frame, bbox_dict)
                
                best10_verified = False
                best10_conf = 0.0
                
                if best10_model is not None:
                    try:
                        best10_results = best10_model(cropped_region, conf=0.60, verbose=False)
                        for res in best10_results:
                            if res.boxes is not None:
                                for b in res.boxes:
                                    if 'knife' in best10_model.names.get(int(b.cls[0]), '').lower():
                                        best10_conf = float(b.conf[0])
                                        if best10_conf >= 0.60:
                                            best10_verified = True
                                            break
                    except Exception:
                        pass
                
                if best10_verified:
                    logger.warning(f"✅ Stage 2: KNIFE VERIFIED by best (10).pt @ {best10_conf:.2f}")
                    knives_detected += 1
                    color = (0, 165, 255)  # Orange
                    label = f'VERIFIED KNIFE {conf:.2f}/{best10_conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    
    # Detect violence (Classification Model)
    if use_violence_model and violence_model is not None:
        try:
            # Run inference on the whole frame
            violence_results = violence_model.predict(frame, imgsz=224, verbose=False)
            if violence_results:
                res = violence_results[0]
                
                # Extract probabilities robustly
                top1_label = None
                top1_conf = 0.0
                
                if hasattr(res, "probs"):
                    p = res.probs
                    # Try to get top1 index and confidence
                    if hasattr(p, "top1") and p.top1 is not None:
                        top1_idx = int(p.top1)
                        top1_conf = float(p.top1conf) if hasattr(p, "top1conf") else 0.0
                        
                        if hasattr(res, "names"):
                            top1_label = res.names[top1_idx]
                
                # Check if violence is detected
                # Classes are {1: 'not_violence', 2: 'violence'}
                if top1_label and top1_label.lower() == 'violence':
                    if top1_conf > 0.50:  # 50% threshold
                        violence_detected += 1
                        
                        # Draw label on frame (since no bbox)
                        label = f"VIOLENCE {top1_conf:.2f}"
                        cv2.putText(frame, label, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # Create detection record
                        detection_id = str(uuid.uuid4())
                        img_h, img_w = frame.shape[:2]
                        
                        detections_to_save.append({
                            "id": detection_id,
                            "detection_type": "violence",
                            "model": "best (2).pt",
                            "confidence": top1_conf,
                            "bbox": {"x1": 0, "y1": 0, "x2": img_w, "y2": img_h}, # Full frame
                            "camera_id": camera_id,
                            "camera_name": "Live Webcam" if camera_type == CameraType.WEBCAM else f"Camera {camera_id}",
                            "location": {"lat": 0, "lng": 0},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "alert_sent": False
                        })
        except Exception as e:
            logger.error(f"Error in violence detection: {e}")


    # Run thermal detection if enabled
    # (count only, no boxes shown)
    if people_model and PERF_CONFIG['enable_people_counting']:
        people_results = people_model(frame, conf=0.5, verbose=False)
        for result in people_results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                cls = int(box.cls[0])
                class_name = people_model.names[cls].lower()

                # Only count 'person' class - NO BOUNDING BOXES DRAWN
                if class_name == 'person':
                    people_detected += 1
                    # Removed: cv2.rectangle and cv2.putText for people
    
    # Update detection stats
    detection_stats['guns'] = guns_detected
    detection_stats['knives'] = knives_detected
    detection_stats['thermal_guns'] = thermal_guns_detected
    detection_stats['people'] = people_detected
    
    # Trigger alert if weapon detected
    if guns_detected > 0 or knives_detected > 0:
        trigger_alert(guns_detected, knives_detected, 0, people_detected)
    else:
        detection_stats['alert'] = False
        detection_stats['alert_message'] = ''
    
    # Add stats overlay to frame
    # Reduced size and removed thermal guns as requested
    overlay_height = 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, overlay_height), (255, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Smaller font size (0.6) and adjusted positions
    cv2.putText(frame, f'Guns: {guns_detected}', (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f'Knives: {knives_detected}', (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    # Removed Thermal Guns line
    cv2.putText(frame, f'People: {people_detected}', (10, 75), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'Violence: {violence_detected}', (10, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, (frame.shape[1] - 350, frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # IPFS upload is now handled in generate_frames to avoid duplication and ensure correct keys
    # if detections_to_save: ... (logic moved to generate_frames)
    
    return frame, detections_to_save, people_detected


def trigger_alert(guns, knives, thermal_guns, people):
    """Trigger alert if weapon detected"""
    global detection_stats
    
    current_time = time.time()
    
    # Check if we're in cooldown period
    if detection_stats['last_alert_time'] is not None:
        if current_time - detection_stats['last_alert_time'] < alert_cooldown:
            return
    
    # Create alert message
    weapons = []
    if guns > 0:
        weapons.append(f"{guns} gun(s)")
    if knives > 0:
        weapons.append(f"{knives} knife(s)")
    if thermal_guns > 0:
        weapons.append(f"{thermal_guns} thermal gun(s)")
    
    alert_message = f"THREAT DETECTED: {' and '.join(weapons)} detected"
    if people > 0:
        alert_message += f" with {people} person(s) in frame"
    
    detection_stats['alert'] = True
    detection_stats['alert_message'] = alert_message
    detection_stats['last_alert_time'] = current_time
    
    logger.warning(f"⚠️ ALERT: {alert_message}")


def detection_save_worker():
    """Background worker thread that saves detections from queue"""
    import asyncio
    
    async def save_detection_async(detection):
        try:
            logger.info(f"💾 Saving detection: {detection['detection_type']}")
            
            # Add to blockchain

            logger.info(f"⛓️ Blockchain hash: {block_hash[:16]}...")
            
            # Save to detections collection
            result = await db.detections.insert_one(detection)
            logger.info(f"✅ Detection saved to DB: {detection['detection_type']} (ID: {detection['id']}, DB ID: {result.inserted_id})")
        except Exception as e:
            logger.error(f"❌ Error saving detection to DB: {e}", exc_info=True)
    
    async def worker_loop():
        while save_worker_running:
            try:
                # Get detection from queue with timeout (non-blocking)
                try:
                    detection = detection_queue.get_nowait()
                except:
                    # Queue is empty, wait a bit
                    await asyncio.sleep(0.5)
                    continue
                
                if detection is None:  # Sentinel value to stop worker
                    break
                
                # Save detection
                await save_detection_async(detection)
                detection_queue.task_done()
            except Exception as e:
                logger.error(f"❌ Worker loop error: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    # Create and run event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(worker_loop())
    finally:
        loop.close()
    logger.info("✅ Detection save worker stopped")


def start_save_worker():
    """Start the background detection save worker"""
    global save_worker_thread, save_worker_running
    
    # Worker disabled - detections from live stream are not saved to DB
    # Only uploaded image detections are saved (handled synchronously)
    logger.info("ℹ️ Detection save worker disabled (live stream detections not saved to DB)")


def stop_save_worker():
    """Stop the background detection save worker"""
    global save_worker_thread, save_worker_running
    
    # Worker is disabled, nothing to stop
    logger.info("ℹ️ Detection save worker already disabled")


def dual_verification_worker():
    """Background worker thread that processes dual verifications"""
    # Use synchronous PyMongo client for this thread to avoid event loop issues
    try:
        client = pymongo.MongoClient(MONGO_URL)
        db_sync = client[DB_NAME]
        logger.info("✅ Dual verification worker connected to MongoDB (Sync)")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB in worker: {e}")
        return

    while verification_worker_running:
        try:
            # Get verification from queue with timeout
            try:
                logger.debug("📭 Checking verification queue...")
                verification_data = dual_verification_queue.get(timeout=1.0)
                logger.info(f"📬 Got verification data from queue!")
            except:
                # Queue is empty, continue
                continue
            
            if verification_data is None:  # Sentinel value to stop worker
                logger.info("🛑 Received stop signal (None)")
                break
            
            # Process verification
            try:
                logger.info(f"🔍 Processing dual verification for camera: {verification_data.get('camera_name', 'unknown')}")
                handle_dual_verification(db=db_sync, **verification_data)
                dual_verification_queue.task_done()
                logger.info("✅ Verification processed successfully")
            except Exception as e:
                logger.error(f"❌ Error processing dual verification: {e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"❌ Verification worker loop error: {e}", exc_info=True)
            time.sleep(0.1)
    
    # Close connection
    try:
        client.close()
    except:
        pass
    logger.info("✅ Dual verification worker stopped")


def start_verification_worker():
    """Start the background dual verification worker"""
    global verification_worker_thread, verification_worker_running
    
    if verification_worker_running:
        logger.warning("⚠️ Verification worker already running")
        return
    
    verification_worker_running = True
    verification_worker_thread = threading.Thread(target=dual_verification_worker, daemon=True)
    verification_worker_thread.start()
    logger.info("✅ Dual verification worker started")


def stop_verification_worker():
    """Stop the background dual verification worker"""
    global verification_worker_thread, verification_worker_running
    
    if not verification_worker_running:
        logger.info("ℹ️ Verification worker already stopped")
        return
    
    logger.info("⏹️ Stopping dual verification worker...")
    verification_worker_running = False
    
    # Send sentinel value to wake up worker
    dual_verification_queue.put(None)
    
    if verification_worker_thread:
        verification_worker_thread.join(timeout=5)
        if verification_worker_thread.is_alive():
            logger.warning("⚠️ Verification worker did not stop gracefully")
        else:
            logger.info("✅ Verification worker stopped")


def generate_frames():
    """Generate frames for video streaming"""
    global current_frame, is_running
    
    while is_running:
        if camera is None:
            time.sleep(0.1)
            continue
            
        frame = camera.get_frame()
        if frame is None:
            continue
        
        # Perform detection
        frame, detections_to_save, people_detected = detect_weapons_and_people(frame)
        
        # Note: Live stream detections are not queued for DB saving to avoid event loop issues
        # Only uploaded image detections are saved to DB
        # Logging disabled to reduce terminal output
        # if detections_to_save:
        #     logger.debug(f"📍 Found {len(detections_to_save)} detections in live stream (not saving to DB)")
        # else:
        #     logger.debug("No detections in this frame")
        
        # Store current frame
        with frame_lock:
            current_frame = frame.copy()
        
        # --- LIVE IPFS UPLOAD LOGIC ---
        # If we have high-confidence threats, upload to IPFS (with cooldown)
        current_time = time.time()
        global last_ipfs_upload_time
        
        # DEBUG LOGGING
        if detections_to_save:
            logger.info(f"🔍 DEBUG: detections_to_save has {len(detections_to_save)} items")
            for d in detections_to_save:
                logger.info(f"   - Type: {d.get('detection_type')}, Conf: {d.get('confidence')}")
        
        if detections_to_save and (last_ipfs_upload_time is None or (current_time - last_ipfs_upload_time) > ipfs_upload_cooldown):
            # Check for high confidence threats (> 10% for testing)
            high_conf_threats = [d for d in detections_to_save if d.get('confidence', 0) > 0.10]
            
            if high_conf_threats:
                logger.info(f"🚀 Initiating background IPFS upload for {len(high_conf_threats)} high-confidence threats...")
                last_ipfs_upload_time = current_time
                
                # Run upload in background thread to not block video feed
                threading.Thread(target=handle_live_ipfs_upload, args=(frame.copy(), high_conf_threats)).start()
        
        # Encode frame with configured quality
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, PERF_CONFIG['jpeg_quality']])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Minimal delay for smoother streaming
        time.sleep(0.01)  # 10ms delay instead of FPS-based delay

def handle_live_ipfs_upload(frame, detections):
    """Background handler for uploading live detections to IPFS"""
    try:
        # 1. Save frame to temp file
        temp_filename = f"live_evidence_{uuid.uuid4()}.jpg"
        temp_path = UPLOAD_DIR / temp_filename
        cv2.imwrite(str(temp_path), frame)
        
        # 2. Upload to IPFS
        metadata = {
            "source": "live_webcam",
            "timestamp": datetime.now().isoformat(),
            "threats": [d['detection_type'] for d in detections]
        }
        
        result = ipfs.upload_to_ipfs(str(temp_path), metadata)
        
        if result and 'ipfs_hash' in result:  # Changed from 'IpfsHash' to 'ipfs_hash'
            ipfs_hash = result['ipfs_hash']
            ipfs_link = result['url']  # Use the url from result instead of constructing it
            
            # 3. Update global stats for frontend
            detection_stats['latest_ipfs'] = {
                "hash": ipfs_hash,
                "link": ipfs_link,
                "timestamp": datetime.now().isoformat(),
                "threat_type": detections[0]['detection_type']
            }
            logger.info(f"✅ Live evidence uploaded to IPFS: {ipfs_hash}")
            logger.info(f"🔗 IPFS URL: {ipfs_link}")
            
            # Clean up temp file
            if temp_path.exists():
                os.remove(temp_path)
        else:
            logger.error(f"❌ Failed to upload live evidence to IPFS: {result}")
            
    except Exception as e:
        logger.error(f"❌ Error in handle_live_ipfs_upload: {e}")

async def process_live_detection(image_data, verbose: bool = False):
    """Detect weapons in image - simplified version"""
    try:
        if image_data is None or image_data.size == 0:
            logger.warning("Empty image data received")
            return []
        
        img_h, img_w = image_data.shape[:2]
        logger.info(f"🔍 Processing image: {img_w}x{img_h}")
        
        # Run detection with higher confidence to reduce false positives
        results = model(image_data, conf=0.50, verbose=False)
        detections_list = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            logger.info(f"📦 Found {len(result.boxes)} boxes")
            
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                det_type = model.names.get(cls, "unknown")
                
                logger.info(f"  Box: class={cls}, type={det_type}, conf={conf:.2f}")
                
                # Only accept guns and knives
                if det_type not in ['pistol', 'knife']:
                    continue
                
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                
                logger.info(f"✅ DETECTED {det_type.upper()} at ({x1:.0f},{y1:.0f}) - ({x2:.0f},{y2:.0f}) with conf {conf:.2f}")
                
                detections_list.append({
                    "detection_type": det_type,
                    "confidence": conf,
                    "bbox": {
                        "x1": float(x1), 
                        "y1": float(y1), 
                        "x2": float(x2), 
                        "y2": float(y2), 
                        "width": img_w, 
                        "height": img_h
                    }
                })
        
        logger.info(f"📊 Total detections: {len(detections_list)}")
        return detections_list
    except Exception as e:
        logger.error(f"❌ Error in process_live_detection: {e}", exc_info=True)
        return []





def preprocess_image(image):
    """Preprocess image for better detection quality"""
    # Resize if too large (maintain aspect ratio)
    max_dim = 1280
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Enhance image quality
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

async def process_detection(image_data, camera_id: str, camera_name: str, location: dict, detection_mode: str = "normal"):
    """
    Process image with specific models based on detection_mode:
    - normal: Runs both best.pt and best (1).pt (for backward compatibility/comparison)
    - model1: Runs ONLY best.pt
    - model2: Runs ONLY best (1).pt
    - people: Runs ONLY people counting model
    """
    detections_list = []
    
    # Preprocess image
    logger.info("🔧 Preprocessing image...")
    processed_image = preprocess_image(image_data)
    img_h, img_w = processed_image.shape[:2]
    logger.info(f"✅ Image preprocessed: {img_w}x{img_h}")
    
    # --- Top View Mode ---
    if detection_mode == "topview":
        logger.info("🔭 Running Top View Person Detection...")
        _, detections, _ = custom_camera_detection.detect_topview_people(processed_image, camera_id=camera_id)
        
        # Add location and other missing fields
        for det in detections:
            det['location'] = location
            det['camera_name'] = camera_name
            det['alert_sent'] = False
            
            # Save annotated image if not already saved (detect_topview_people doesn't save it by default)
            # But wait, detect_topview_people draws on the frame but doesn't save it to disk/return path
            # We need to handle that.
            
            # Let's save the frame if we have detections
            if not det.get('image_path'):
                fname = f"topview_{det['id']}.jpg"
                path = DETECTIONS_DIR / fname
                cv2.imwrite(str(path), processed_image)
                det['image_path'] = f"/detections/{fname}"
                
            await db.detections.insert_one(det)
            
        detections_list.extend(detections)
        logger.info(f"✅ Top view people detected: {len(detections_list)}")
        return detections_list
    
    # --- People Counting Mode ---
    if detection_mode == "people":
        if people_model is None:
            logger.warning("⚠️ People counting model not loaded")
            return []
            
        logger.info("👥 Running People Counting model...")
        results_people = people_model(processed_image, conf=0.35, iou=0.5)
        
        for result in results_people:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls not in people_model.names:
                    continue
                label = people_model.names[cls]
                
                if label != 'person':
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                
                detection_id = str(uuid.uuid4())
                fname = f"people_{detection_id}.jpg"
                path = DETECTIONS_DIR / fname
                image_path_val = None
                try:
                    # Create annotated image with custom labels
                    annotated = processed_image.copy()
                    # Draw bounding box
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Draw label background
                    label = f"PERSON {conf:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (int(x1), int(y1) - label_h - 10), (int(x1) + label_w, int(y1)), (0, 255, 0), -1)
                    # Draw label text
                    cv2.putText(annotated, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    cv2.imwrite(str(path), annotated)
                    image_path_val = f"/detections/{fname}"
                except Exception as e:
                    logger.warning(f"Failed to create annotated image: {e}")
                    image_path_val = None

                detection_data = {
                    "id": detection_id,
                    "camera_id": camera_id or "upload",
                    "camera_name": camera_name,
                    "detection_type": "person",
                    "confidence": conf,
                    "model": "yolov8m.pt",
                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "image_path": image_path_val,
                    "location": location,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alert_sent": False 
                }
                # For people, we might not want to alert/blockchain every single one, 
                # but for consistency in "upload testing", we'll save it to DB.
                await db.detections.insert_one(detection_data)
                detections_list.append(detection_data)
        
        logger.info(f"✅ People detected: {len(detections_list)}")
        return detections_list

    # --- Violence Detection Mode ---
    if detection_mode == "violence":
        if violence_model is None:
            logger.warning("⚠️ Violence model not loaded")
            return []
            
        logger.info("👊 Running Violence Detection model...")
        violence_results = violence_model.predict(processed_image, imgsz=224, verbose=False)
        
        if violence_results:
            res = violence_results[0]
            top1_label = None
            top1_conf = 0.0
            
            if hasattr(res, "probs"):
                p = res.probs
                if hasattr(p, "top1") and p.top1 is not None:
                    top1_idx = int(p.top1)
                    top1_conf = float(p.top1conf) if hasattr(p, "top1conf") else 0.0
                    if hasattr(res, "names"):
                        top1_label = res.names[top1_idx]
            
            logger.info(f"👊 Violence Model Result: Label={top1_label}, Conf={top1_conf:.2f}")

            # Check if violence is detected (Threshold 50%)
            # Note: Classes are {0: '_classes', 1: 'not_violence', 2: 'violence'}
            if top1_label and top1_label.lower() == 'violence' and top1_conf > 0.50:
                detection_id = str(uuid.uuid4())
                fname = f"violence_{detection_id}.jpg"
                path = DETECTIONS_DIR / fname
                image_path_val = None
                
                try:
                    # Create annotated image
                    annotated = processed_image.copy()
                    label = f"VIOLENCE {top1_conf:.2f}"
                    # Draw label at top left
                    cv2.putText(annotated, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imwrite(str(path), annotated)
                    image_path_val = f"/detections/{fname}"
                except Exception as e:
                    logger.warning(f"Failed to create annotated image: {e}")
                
                detection_data = {
                    "id": detection_id,
                    "camera_id": camera_id or "upload",
                    "camera_name": camera_name,
                    "detection_type": "violence",
                    "confidence": top1_conf,
                    "model": "best (2).pt",
                    "bbox": {"x1": 0, "y1": 0, "x2": float(img_w), "y2": float(img_h)}, # Whole image
                    "image_path": image_path_val,
                    "location": location,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alert_sent": False
                }
                
                await db.detections.insert_one(detection_data)
                detections_list.append(detection_data)
                logger.info(f"✅ Violence detected: {top1_conf:.2f}")
            else:
                logger.info(f"ℹ️ No violence detected (Label: {top1_label}, Conf: {top1_conf:.2f})")
                
        return detections_list

    # --- Weapon Detection Modes ---
    
    # Run best.pt model (Model 1) - if mode is 'normal' or 'model1'
    if detection_mode in ["normal", "model1"]:
        logger.info("🔴 Running best.pt model...")
        results_model1 = model(processed_image, conf=0.25, iou=0.5)
        
        # Process best.pt model results
        for result in results_model1:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls not in model.names:
                    continue
                det_type = model.names[cls] # Keep original case for det_type
                class_name = det_type.lower() # Use lowercased for comparison
                
                # Apply class-specific confidence thresholds
                if class_name == 'knife':
                    # Higher threshold for knives to reduce false positives
                    if conf < 0.80:  # 80% confidence for knives
                        continue
                elif class_name == 'pistol':
                    # Medium threshold for pistols
                    if conf < 0.50:  # 50% confidence for pistols
                        continue
                else:
                    # If it's not a pistol or knife, skip it
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                if not validate_detection(det_type, conf, x1, y1, x2, y2, img_w, img_h):
                    continue
                
                detection_id = str(uuid.uuid4())
                fname = f"model1_{detection_id}.jpg"
                path = DETECTIONS_DIR / fname
                image_path_val = None
                try:
                    # Create annotated image with custom labels
                    annotated = processed_image.copy()
                    # Draw bounding box (red for weapons)
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    # Draw label background
                    label = f"{det_type.upper()} {conf:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated, (int(x1), int(y1) - label_h - 10), (int(x1) + label_w, int(y1)), (0, 0, 255), -1)
                    # Draw label text
                    cv2.putText(annotated, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imwrite(str(path), annotated)
                    image_path_val = f"/detections/{fname}"
                    logger.info(f"✅ Saved annotated image to: {path}")
                except Exception as e:
                    logger.error(f"❌ Failed to create annotated image: {e}")
                    image_path_val = None

                detection_data = {
                    "id": detection_id,
                    "camera_id": camera_id or "upload",
                    "camera_name": camera_name,
                    "detection_type": det_type,
                    "confidence": conf,
                    "model": "best.pt",  # Tag with model name
                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "image_path": image_path_val,
                    "location": location,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alert_sent": False
                }
                
                # Upload to IPFS (Async/Background)
                if image_path_val:
                    logger.info(f"🚀 Attempting IPFS upload for: {image_path_val}")
                    try:
                        # ACTUAL IMPLEMENTATION:
                        ipfs_result = ipfs.upload_to_ipfs(str(path), metadata={"detection_type": det_type})
                        
                        if ipfs_result and 'ipfs_hash' in ipfs_result:
                            detection_data['ipfs_data'] = ipfs_result
                            
                            # Update global stats for frontend
                            detection_stats['latest_ipfs'] = {
                                'ipfs_hash': ipfs_result['ipfs_hash'],
                                'sha256_hash': ipfs_result['sha256_hash'],
                                'url': ipfs_result['url'],
                                'timestamp': datetime.now(timezone.utc).isoformat(),
                                'type': det_type
                            }
                            
                            logger.info(f"✅ Live Detection uploaded to IPFS: {ipfs_result['ipfs_hash']}")
                            logger.info(f"📊 Updated global stats: {detection_stats['latest_ipfs']}")
                        else:
                            logger.warning("⚠️ IPFS upload returned no result or missing hash")
                            
                    except Exception as e:
                        logger.error(f"❌ Failed to upload live detection to IPFS: {e}")
                else:
                    logger.warning("⚠️ No image path available for IPFS upload")


                await db.detections.insert_one(detection_data)
                send_twilio_alert(detection_data)
                detection_data['alert_sent'] = True
                await db.detections.update_one({"id": detection_id}, {"$set": {"alert_sent": True}})
                detections_list.append(detection_data)
        
        logger.info(f"✅ best.pt detected: {len([d for d in detections_list if d.get('model') == 'best.pt'])} weapons")
    
    # Thermal detection is handled separately in the upload endpoint, 
    # but if we wanted to include it here we could. 
    # For now, the upload endpoint calls process_thermal_only separately.
    
    # --- Violence Detection Mode ---
    # --- Violence Detection Mode ---
    if detection_mode == "violence":
        if violence_model is None:
            logger.warning("⚠️ Violence model not loaded")
            return []
            
        logger.info("👊 Running Violence Detection model...")
        # Run inference
        results = violence_model.predict(source=processed_image, imgsz=224, save=False)
        res = results[0]

        # Extract probabilities and names robustly
        probs = None
        top1_idx = None
        top1_conf = None
        
        if hasattr(res, "probs"):
            # Ultralytics Probs object
            p = res.probs
            # top1 index and confidence
            top1_idx = int(p.top1) if hasattr(p, "top1") and p.top1 is not None else None
            top1_conf = float(p.top1conf) if hasattr(p, "top1conf") and p.top1conf is not None else None
            
            # convert probs to numpy if needed
            try:
                probs = p.cpu().numpy()
            except Exception:
                try:
                    probs = p.numpy()
                except Exception:
                    probs = None
        else:
            logger.warning("Couldn't find classification probabilities on result object.")
            return []

        # Get class names
        names = res.names if hasattr(res, "names") else None
        if names is None:
            logger.warning("Class names not found on result object.")
            return []

        # If top1_idx not available, compute from probs array
        if top1_idx is None and probs is not None:
            if hasattr(probs, "data"):
                 arr = probs.data.cpu().numpy().ravel()
            else:
                 arr = np.array(probs).ravel()
            top1_idx = int(arr.argmax())
            top1_conf = float(arr[top1_idx])

        if top1_idx is not None:
            top1_label = names[top1_idx]
            logger.info(f"=== PREDICTION === Top-1 class: {top1_label}, Confidence: {top1_conf:.4f}")
            
            # Check if the top class is 'violence'
            if 'violence' in top1_label.lower() and 'non' not in top1_label.lower():
                # Threshold for violence detection
                VIOLENCE_THRESHOLD = 0.50
                
                if top1_conf > VIOLENCE_THRESHOLD:
                    logger.info(f"👊 Violence detected with confidence {top1_conf:.4f}")
                    
                    detection_id = str(uuid.uuid4())
                    fname = f"violence_{detection_id}.jpg"
                    path = DETECTIONS_DIR / fname
                    image_path_val = None
                    
                    try:
                        annotated = processed_image.copy()
                        label = f"VIOLENCE {top1_conf:.2f}"
                        # Draw text on image since we don't have a bounding box
                        cv2.putText(annotated, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imwrite(str(path), annotated)
                        image_path_val = f"/detections/{fname}"
                    except Exception as e:
                        logger.warning(f"Failed to create annotated image: {e}")
                        image_path_val = None

                    detection_data = {
                        "id": detection_id,
                        "camera_id": camera_id or "upload",
                        "camera_name": camera_name,
                        "detection_type": "violence",
                        "confidence": top1_conf,
                        "model": "best (2).pt",
                        "bbox": {"x1": 0.0, "y1": 0.0, "x2": float(img_w), "y2": float(img_h)}, # Whole image
                        "image_path": image_path_val,
                        "location": location,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "alert_sent": False
                    }
                    
                    await db.detections.insert_one(detection_data)
                    detections_list.append(detection_data)
                else:
                     logger.info(f"ℹ️ Violence confidence {top1_conf:.4f} below threshold {VIOLENCE_THRESHOLD}")
            else:
                logger.info(f"ℹ️ Top class is {top1_label} (not violence)")

    # If detections exist but none have an image_path, create a combined
    # annotated image by drawing boxes on the original uploaded image and
    # attach that image to all detections so frontend can display boxes.
    try:
        # Check if at least one detection lacks image_path
        need_fallback = any(det.get('image_path') in (None, '', False) for det in detections_list) and len(detections_list) > 0
        if need_fallback:
            annotated_fname = f"annotated_{uuid.uuid4()}.jpg"
            annotated_path = DETECTIONS_DIR / annotated_fname
            annotated_img = image_data.copy()
            # Draw boxes for each detection (use bbox if available)
            for det in detections_list:
                bbox = det.get('bbox')
                if bbox:
                    try:
                        x1 = int(bbox.get('x1', 0))
                        y1 = int(bbox.get('y1', 0))
                        x2 = int(bbox.get('x2', 0))
                        y2 = int(bbox.get('y2', 0))
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(annotated_img, f"{det.get('detection_type', '').upper()} {det.get('confidence', 0):.2f}", (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    except Exception:
                        continue
            cv2.imwrite(str(annotated_path), annotated_img)
            for det in detections_list:
                if not det.get('image_path'):
                    det['image_path'] = f"/detections/{annotated_fname}"
    except Exception as e:
        logger.warning(f"Failed to create fallback annotated image: {e}")

    return detections_list

async def process_thermal_only(image_data, camera_id: str, camera_name: str, location: dict):
    """Process image with weapon detection (best.pt + best (1).pt only) @ 25% confidence"""
    detections_list = []
    annotated_images = {}  # Store different annotated versions
    
    img_h, img_w = image_data.shape[:2]
    
    # Run best.pt weapon detection
    best_pt_annotated = image_data.copy()
    best_pt_found = False
    try:
        logger.info("🔴 Running best.pt weapon detection (25% conf)...")
        results = model(image_data, conf=0.25, verbose=False)
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls].lower()
                
                if any(keyword in class_name for keyword in ['pistol', 'gun', 'knife', 'rifle', 'shotgun', 'weapon']):
                    best_pt_found = True
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Draw on annotated image
                    color = (0, 0, 255) if 'gun' in class_name or 'pistol' in class_name else (0, 165, 255)
                    cv2.rectangle(best_pt_annotated, (x1, y1), (x2, y2), color, 3)
                    label = f'{class_name.upper()} {conf:.2f}'
                    cv2.putText(best_pt_annotated, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Add to detections list
                    detection_id = str(uuid.uuid4())
                    det_data = {
                        "id": detection_id,
                        "detection_type": "pistol" if 'gun' in class_name or 'pistol' in class_name else "knife",
                        "confidence": conf,
                        "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                        "camera_id": camera_id,
                        "camera_name": camera_name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "location": location,
                        "model": "best.pt"
                    }
                    detections_list.append(det_data)
        
        if best_pt_found:
            best_pt_fname = f"best_pt_{uuid.uuid4()}.jpg"
            best_pt_path = DETECTIONS_DIR / best_pt_fname
            cv2.imwrite(str(best_pt_path), best_pt_annotated)
            annotated_images['best_pt'] = f"/detections/{best_pt_fname}"
            logger.info(f"💾 Saved best.pt annotated image: {best_pt_fname}")
    except Exception as e:
        logger.error(f"❌ Error during best.pt detection: {e}", exc_info=True)
    
    # Run best (1).pt weapon detection
    best1_annotated = image_data.copy()
    best1_found = False
    try:
        logger.info("🟢 Running best (1).pt weapon detection (25% conf)...")
        results = best1_model(image_data, conf=0.25, verbose=False)
        
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = best1_model.names[cls].lower()
                
                if any(keyword in class_name for keyword in ['pistol', 'gun', 'knife', 'rifle', 'shotgun', 'weapon']):
                    best1_found = True
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Draw on annotated image
                    color = (255, 255, 0)  # Cyan for best (1).pt
                    cv2.rectangle(best1_annotated, (x1, y1), (x2, y2), color, 3)
                    label = f'BEST1-{class_name.upper()} {conf:.2f}'
                    cv2.putText(best1_annotated, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Add to detections list (if not duplicate - simple check)
                    # For now, we add all detections from both models to show comprehensive results
                    detection_id = str(uuid.uuid4())
                    det_data = {
                        "id": detection_id,
                        "detection_type": "pistol" if 'gun' in class_name or 'pistol' in class_name else "knife",
                        "confidence": conf,
                        "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                        "camera_id": camera_id,
                        "camera_name": camera_name,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "location": location,
                        "model": "best (1).pt"
                    }
                    detections_list.append(det_data)
        
        if best1_found:
            best1_fname = f"best1_{uuid.uuid4()}.jpg"
            best1_path = DETECTIONS_DIR / best1_fname
            cv2.imwrite(str(best1_path), best1_annotated)
            annotated_images['best1'] = f"/detections/{best1_fname}"
            logger.info(f"💾 Saved best (1).pt annotated image: {best1_fname}")
    except Exception as e:
        logger.error(f"❌ Error during best (1).pt detection: {e}", exc_info=True)
    
    # Process detections (blockchain, DB, alerts) if ANY weapon found
    if best_pt_found or best1_found:
        logger.warning(f"🚨 Weapon detected in upload! (best.pt: {best_pt_found}, best1: {best1_found})")
        # We already added to detections_list, now process them
        for det in detections_list:
            # Add image path based on model
            if det['model'] == 'best.pt' and 'best_pt' in annotated_images:
                det['image_path'] = annotated_images['best_pt']
            elif det['model'] == 'best (1).pt' and 'best1' in annotated_images:
                det['image_path'] = annotated_images['best1']
            

            await db.detections.insert_one(det)
            # Only send alert for the first detection to avoid spam
            if det == detections_list[0]:
                send_twilio_alert(det)
                det['alert_sent'] = True
                await db.detections.update_one({"id": det['id']}, {"$set": {"alert_sent": True}})
    
    return {
        "detections": detections_list,
        "annotated_images": annotated_images
    }

# App and routes
app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"📥 {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"📤 {request.method} {request.url} - {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"❌ Request failed: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Create indexes
        await db.detections.create_index([("timestamp", -1)])
        await db.pending_verifications.create_index([("status", 1), ("timestamp", -1)])
        logger.info("✅ Database indexes created")
    except Exception as e:
        logger.warning(f"⚠️ Could not create indexes: {e}")

    try:
        from custom_cameras import register_custom_cameras
        register_custom_cameras(ROOT_DIR)
    except Exception as e:
        logger.error(f"Failed to register custom cameras: {e}")

# Middleware: disable caching for /detections to ensure frontend always
# fetches latest annotated images (prevents 304 stale responses hiding boxes)
class NoCacheDetectionsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        try:
            path = request.url.path or ""
            if path.startswith("/detections"):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        except Exception:
            pass
        return response

app.add_middleware(NoCacheDetectionsMiddleware)
api_router = APIRouter(prefix="/api")

@api_router.get("/")
async def root():
    return {"message": "Gun & Knife Detection System API (camera features removed)"}

@api_router.get("/test-model")
async def test_model():
    return {"status": "ok", "model_exists": MODEL_PATH.exists(), "classes": model.names}

@api_router.post("/detect-drone-video")
async def detect_drone_video(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.50),
    frame_skip: int = Form(5)  # Process every 5th frame (like live camera)
):
    """
    Detect people in drone videos using drone.pt model.
    Processes video frame-by-frame and returns all detections.
    
    Args:
        file: Video file to process
        conf_threshold: Confidence threshold (0.0 - 1.0)
        frame_skip: Process every Nth frame (1=all frames, 5=every 5th, 30=every 30th)
    """
    try:
        # Validate file extension
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid video format. Supported: .mp4, .avi, .mov, .mkv")
        
        # Use topview_model if drone_model is not available
        model_to_use = drone_model if drone_model else topview_model
        model_name = "drone.pt" if drone_model else "best (5).pt (top-view)"
        
        if not model_to_use:
            raise HTTPException(status_code=500, detail="Neither drone model nor top-view model is loaded")
        
        logger.info(f"🎥 Using model: {model_name} for video detection")
        
        # Save uploaded video
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        filename = f"drone_video_{file_id}{file_ext}"
        file_path = UPLOAD_DIR / filename
        
        logger.info(f"🎥 Drone video upload: {file.filename}")
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        logger.info(f"📁 Saved video to: {file_path}")
        
        # Process video
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Failed to open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"📊 Video stats: {total_frames} frames @ {fps} FPS")
        
        all_detections = []
        frame_count = 0
        processed_frames = 0
        annotated_samples = []  # Store sample annotated frames
        
        # Use configurable frame_skip (default 5 = like live camera)
        logger.info(f"🎬 Processing every {frame_skip} frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_skip == 0:
                img_h, img_w = frame.shape[:2]
                frame_detections_count = 0
                annotated_frame = frame.copy()
                
                # Run detection with selected model
                results = model_to_use(frame, conf=conf_threshold, verbose=False)
                
                for result in results:
                    if result.boxes is None:
                        continue
                    
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model_to_use.names[cls].lower()
                        
                        # Drone model detects 'pedestrian', not 'person'
                        if 'pedestrian' in class_name or 'person' in class_name:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Filter out tiny detections (likely false positives)
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            bbox_area = bbox_width * bbox_height
                            img_area = img_w * img_h
                            area_ratio = bbox_area / img_area if img_area > 0 else 0
                            
                            # Skip detections that are too small (< 0.5% of image) or too large (> 80%)
                            if area_ratio < 0.005 or area_ratio > 0.8:
                                continue
                            
                            # Also skip very thin boxes (likely artifacts)
                            if bbox_width < 20 or bbox_height < 20:
                                continue
                            
                            # Draw bounding box on frame
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f'PERSON (PEDESTRIAN) {conf:.2f}'
                            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            detection = {
                                "id": str(uuid.uuid4()),
                                "frame_number": frame_count,
                                "timestamp_seconds": round(frame_count / fps, 2) if fps > 0 else 0,
                                "class": "person (drone)",
                                "confidence": round(conf, 3),
                                "bbox": {
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2,
                                    "width": img_w,
                                    "height": img_h
                                },
                                "model": model_name,
                                "type": "person"
                            }
                            all_detections.append(detection)
                            frame_detections_count += 1
                
                # Save annotated frame if it has detections (max 5 samples)
                if frame_detections_count > 0 and len(annotated_samples) < 5:
                    sample_filename = f"drone_frame_{file_id}_f{frame_count}.jpg"
                    sample_path = DETECTIONS_DIR / sample_filename
                    cv2.imwrite(str(sample_path), annotated_frame)
                    annotated_samples.append({
                        "frame_number": frame_count,
                        "timestamp": round(frame_count / fps, 2) if fps > 0 else 0,
                        "detections": frame_detections_count,
                        "image_url": f"/detections/{sample_filename}"
                    })
                    logger.info(f"💾 Saved annotated frame {frame_count} with {frame_detections_count} detections")
                
                processed_frames += 1
                
                # Log progress every 100 processed frames
                if processed_frames % 100 == 0:
                    logger.info(f"🔄 Processed {processed_frames} frames, found {len(all_detections)} people")
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"✅ Video processing complete")
        logger.info(f"📊 Total frames: {frame_count}, Processed: {processed_frames}, Detections: {len(all_detections)}")
        
        # Calculate statistics
        unique_people_per_frame = {}
        for det in all_detections:
            frame_num = det['frame_number']
            if frame_num not in unique_people_per_frame:
                unique_people_per_frame[frame_num] = 0
            unique_people_per_frame[frame_num] += 1
        
        max_people_in_frame = max(unique_people_per_frame.values()) if unique_people_per_frame else 0
        avg_people_per_frame = sum(unique_people_per_frame.values()) / len(unique_people_per_frame) if unique_people_per_frame else 0
        
        return {
            "status": "ok",
            "video_info": {
                "filename": file.filename,
                "total_frames": frame_count,
                "processed_frames": processed_frames,
                "fps": fps,
                "duration_seconds": round(frame_count / fps, 2) if fps > 0 else 0
            },
            "detection_summary": {
                "total_detections": len(all_detections),
                "frames_with_people": len(unique_people_per_frame),
                "max_people_in_single_frame": max_people_in_frame,
                "avg_people_per_frame": round(avg_people_per_frame, 2)
            },
            "annotated_samples": annotated_samples,  # Sample frames with detections drawn
            "detections": all_detections,
            "conf_threshold": conf_threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in detect_drone_video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/test-detection")
async def test_detection(file: UploadFile = File(...)):
    try:
        logger.info(f"🎬 TEST-DETECTION endpoint called with file: {file.filename}")
        contents = await file.read()
        logger.info(f"📥 Received file size: {len(contents)} bytes")
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("❌ Failed to decode image")
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        logger.info(f"✅ Image decoded successfully: {img.shape}")
        detections = await process_live_detection(img)
        
        logger.info(f"🎯 Returning {len(detections)} detections")
        return {"status": "ok", "detections": detections}
    except Exception as e:
        logger.error(f"❌ Error in test_detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/count-people")
async def count_people_endpoint(file: UploadFile = File(...)):
    temp_path = UPLOAD_DIR / f"temp_{uuid.uuid4()}.jpg"
    contents = await file.read()
    async with aiofiles.open(temp_path, 'wb') as f:
        await f.write(contents)
    try:
        count = count_people_from_frame(cv2.imread(str(temp_path)))
        return {"status": "ok", "people_count": count}
    finally:
        if temp_path.exists():
            temp_path.unlink()

@api_router.post("/detect-thermal-guns")
async def detect_thermal_guns_endpoint(file: UploadFile = File(...), preprocess: bool = True):
    """Detect thermal guns in uploaded image using local YOLO model (thermal.pt)"""
    try:
        logger.info(f"🌡️ THERMAL-GUN-DETECTION endpoint called with file: {file.filename}, preprocess: {preprocess}")
        contents = await file.read()
        logger.info(f"📥 Received file size: {len(contents)} bytes")
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("❌ Failed to decode image")
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        logger.info(f"✅ Image decoded successfully: {img.shape}")
        
        # Call the updated detection function with preprocessing parameter
        thermal_guns_count, detections, annotated_img = detect_thermal_guns(img, preprocess=preprocess)

        # Save annotated image and attach image_path to detections for frontend
        annotated_image_path = None
        try:
            if annotated_img is not None and len(detections) > 0:
                annotated_fname = f"thermal_annotated_{uuid.uuid4()}.jpg"
                annotated_path = DETECTIONS_DIR / annotated_fname
                ret = cv2.imwrite(str(annotated_path), annotated_img)
                logger.info(f"💾 Saved annotated thermal image: {annotated_fname} (success: {ret})")
                if ret:
                    annotated_image_path = f"/detections/{annotated_fname}"
                    for det in detections:
                        det['image_path'] = annotated_image_path
        except Exception as e:
            logger.warning(f"Failed to save annotated thermal image: {e}")

        logger.info(f"🎯 Returning {thermal_guns_count} thermal gun detections")
        return {"status": "ok", "thermal_guns_count": thermal_guns_count, "detections": detections, "annotated_image": annotated_image_path}
    except Exception as e:
        logger.error(f"❌ Error in detect_thermal_guns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/test-upload")
async def test_upload(
    file: UploadFile = File(...),
    model_type: str = Form("all"), # all, weapon, violence, person
    camera_id: str = Form("test")
):
    """Test detection on uploaded image or video"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / f"test_{uuid.uuid4()}_{file.filename}"
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
            
        logger.info(f"🎥 Video upload detected for {camera_id}: {file.filename}")
        
        detections = []
        annotated_img = None
        
        # Check if video
        if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(str(file_path))
            
            # Setup Video Writer
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_filename = f"analyzed_{uuid.uuid4()}.mp4"
            output_path = UPLOAD_DIR / output_filename
            
            # Use mp4v codec for compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_count = 0
            processed_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 30th frame for detection, but write ALL frames
                # To make it efficient, we'll just run detection on every frame for the "analyzed video" request
                # OR we can stick to 1 fps detection and just draw the last known boxes?
                # Let's run detection on every frame for better video quality, but it might be slow.
                # Given the user wants "analyzed video", they probably expect continuous boxes.
                # Let's stick to every 5th frame for performance/smoothness balance?
                # Or just every 30th frame as before but hold the boxes?
                
                # Let's do every frame for "Top View" since it's a demo
                # Actually, let's stick to the existing logic but write the frame
                
                annotated_frame = frame.copy()
                
                if frame_count % 30 == 0:
                    img_h, img_w = frame.shape[:2]
                    
                    # === TOP VIEW DETECTION ===
                    if "topview" in camera_id.lower() or model_type == "topview":
                         _, frame_detections, _ = custom_camera_detection.detect_topview_people(frame.copy(), camera_id="topview")
                         # Convert to standard format
                         for d in frame_detections:
                             detections.append({
                                 "id": d['id'],
                                 "class": "person",
                                 "confidence": d['confidence'],
                                 "bbox": [d['bbox']['x1'], d['bbox']['y1'], d['bbox']['x2'], d['bbox']['y2']],
                                 "model": "best (9).pt",
                                 "type": "person",
                                 "timestamp": d['timestamp']
                             })
                         
                         # Draw boxes on the frame
                         for det in frame_detections:
                             x1, y1, x2, y2 = int(det['bbox']['x1']), int(det['bbox']['y1']), int(det['bbox']['x2']), int(det['bbox']['y2'])
                             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                             cv2.putText(annotated_frame, f"Person {det['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # === STANDARD DETECTION (all other modes) ===
                    else:
                        frame_detections = []
                        
                        # Run detection based on model_type
                        if model_type == 'weapon':
                            # Weapon detection
                            results = model(frame, conf=0.50, verbose=False)
                            for result in results:
                                if result.boxes is not None:
                                    for box in result.boxes:
                                        cls = int(box.cls[0])
                                        conf = float(box.conf[0])
                                        class_name = model.names[cls].lower()
                                        if 'pistol' in class_name or 'gun' in class_name or 'knife' in class_name or 'rifle' in class_name or 'shotgun' in class_name:
                                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                                            frame_detections.append({
                                                "class": class_name,
                                                "confidence": conf,
                                                "bbox": [x1, y1, x2, y2],
                                                "model": "best.pt",
                                                "color": (0, 0, 255)  # Red
                                            })
                        
                        elif model_type == 'gun-fusion':
                            # Gun fusion - multiple models
                            results = model(frame, conf=0.50, verbose=False)
                            for result in results:
                                if result.boxes is not None:
                                    for box in result.boxes:
                                        cls = int(box.cls[0])
                                        conf = float(box.conf[0])
                                        class_name = model.names[cls].lower()
                                        if 'pistol' in class_name or 'gun' in class_name:
                                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                                            frame_detections.append({
                                                "class": f"{class_name} (best.pt)",
                                                "confidence": conf,
                                                "bbox": [x1, y1, x2, y2],
                                                "model": "best.pt",
                                                "color": (255, 0, 0)  # Blue
                                            })
                            
                            if best1_model:
                                results1 = best1_model(frame, conf=0.30, verbose=False)
                                for result in results1:
                                    if result.boxes is not None:
                                        for box in result.boxes:
                                            cls = int(box.cls[0])
                                            conf = float(box.conf[0])
                                            class_name = best1_model.names[cls].lower()
                                            if 'pistol' in class_name or 'gun' in class_name:
                                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                                frame_detections.append({
                                                    "class": f"{class_name} (best1.pt)",
                                                    "confidence": conf,
                                                    "bbox": [x1, y1, x2, y2],
                                                    "model": "best (1).pt",
                                                    "color": (255, 255, 0)  # Cyan
                                                })
                            
                            if best8_model:
                                results8 = best8_model(frame, conf=0.30, verbose=False)
                                for result in results8:
                                    if result.boxes is not None:
                                        for box in result.boxes:
                                            cls = int(box.cls[0])
                                            conf = float(box.conf[0])
                                            class_name = best8_model.names[cls].lower()
                                            if 'pistol' in class_name or 'gun' in class_name:
                                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                                frame_detections.append({
                                                    "class": f"{class_name} (best8.pt)",
                                                    "confidence": conf,
                                                    "bbox": [x1, y1, x2, y2],
                                                    "model": "best (8).pt",
                                                    "color": (0, 255, 255)  # Yellow
                                                })
                        
                        elif model_type == 'person':
                            if people_model:
                                results = people_model(frame, conf=0.50, verbose=False)
                                for result in results:
                                    if result.boxes is not None:
                                        for box in result.boxes:
                                            cls = int(box.cls[0])
                                            conf = float(box.conf[0])
                                            class_name = people_model.names[cls].lower()
                                            if class_name == 'person':
                                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                                frame_detections.append({
                                                    "class": "person",
                                                    "confidence": conf,
                                                    "bbox": [x1, y1, x2, y2],
                                                    "model": "yolov8n.pt",
                                                    "color": (0, 255, 0)  # Green
                                                })
                        
                        elif model_type == 'all':
                            # Run all models - weapon + people + violence
                            results = model(frame, conf=0.50, verbose=False)
                            for result in results:
                                if result.boxes is not None:
                                    for box in result.boxes:
                                        cls = int(box.cls[0])
                                        conf = float(box.conf[0])
                                        class_name = model.names[cls].lower()
                                        if 'pistol' in class_name or 'gun' in class_name or 'knife' in class_name:
                                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                                            frame_detections.append({
                                                "class": class_name,
                                                "confidence": conf,
                                                "bbox": [x1, y1, x2, y2],
                                                "model": "best.pt",
                                                "color": (0, 0, 255)  # Red
                                            })
                            
                            if people_model:
                                results_people = people_model(frame, conf=0.50, verbose=False)
                                for result in results_people:
                                    if result.boxes is not None:
                                        for box in result.boxes:
                                            cls = int(box.cls[0])
                                            conf = float(box.conf[0])
                                            class_name = people_model.names[cls].lower()
                                            if class_name == 'person':
                                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                                frame_detections.append({
                                                    "class": "person",
                                                    "confidence": conf,
                                                    "bbox": [x1, y1, x2, y2],
                                                    "model": "yolov8n.pt",
                                                    "color": (0, 255, 0)  # Green
                                                })
                        
                        # Draw detections on annotated_frame
                        for det in frame_detections:
                            x1, y1, x2, y2 = det['bbox']
                            color = det.get('color', (0, 255, 0))
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{det['class']} {det['confidence']:.2f}"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Add to overall detections list
                            detections.append({
                                "id": str(uuid.uuid4()),
                                "class": det['class'],
                                "confidence": det['confidence'],
                                "bbox": det['bbox'],
                                "model": det['model'],
                                "frame_number": frame_count,
                                "timestamp": frame_count / fps
                            })
                else:
                     # For frames in between, we could draw the previous boxes, but for now let's just write the raw frame
                     # or maybe just skip detection to keep it fast.
                     # If the user wants to SEE the analysis, we should probably persist the boxes for 30 frames.
                     pass

                # Write frame to output video
                out.write(annotated_frame)
                frame_count += 1
                
            cap.release()
            out.release()
            
            # If we have an annotated image (last frame), save it
            if annotated_frame is not None:
                annotated_fname = f"test_annotated_{uuid.uuid4()}.jpg"
                annotated_path = DETECTIONS_DIR / annotated_fname
                cv2.imwrite(str(annotated_path), annotated_frame)
                annotated_image_url = f"/detections/{annotated_fname}"
            else:
                annotated_image_url = None
                
            return {
                "status": "ok",
                "detections": detections,
                "annotated_image_url": annotated_image_url,
                "video_output": f"/uploads/{output_filename}",
                "video_stats": {
                    "resolution": f"{width}x{height}",
                    "fps": int(fps),
                    "total_frames": frame_count,
                    "total_detections": len(detections)
                },
                "processing_time": 0,  # Could add actual timing if needed
                "message": f"Processed {frame_count} frames with {len(detections)} detections"
            }

        # Handle Image
        image = cv2.imread(str(file_path))
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        img_h, img_w = image.shape[:2]
        
        # === TOP VIEW DETECTION (Image) ===
        if "topview" in camera_id.lower() or model_type == "topview":
             _, frame_detections, _ = custom_camera_detection.detect_topview_people(image.copy(), camera_id="topview")
             for d in frame_detections:
                 detections.append({
                     "id": d['id'],
                     "class": "person",
                     "confidence": d['confidence'],
                     "bbox": [d['bbox']['x1'], d['bbox']['y1'], d['bbox']['x2'], d['bbox']['y2']],
                     "model": "best (9).pt",
                     "type": "person"
                 })
             image = image.copy() # Prepare for annotation below
        
        # === WEAPON DETECTION (best.pt + best (1).pt) ===
        elif model_type == 'weapon':
            try:
                # Run best.pt for weapon detection
                results = model(image, conf=0.50, verbose=False)
                for result in results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls].lower()
                        
                        if 'pistol' in class_name or 'gun' in class_name or 'knife' in class_name or 'rifle' in class_name or 'shotgun' in class_name:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detections.append({
                                "id": str(uuid.uuid4()),
                                "class": class_name,
                                "confidence": conf,
                                "bbox": [x1, y1, x2, y2],
                                "model": "best.pt",
                                "type": "weapon"
                            })
                
                # Run best (1).pt for verification
                if best1_model:
                    results1 = best1_model(image, conf=0.30, verbose=False)
                    for result in results1:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = best1_model.names[cls].lower()
                            
                            if 'pistol' in class_name or 'gun' in class_name or 'knife' in class_name or 'rifle' in class_name or 'shotgun' in class_name:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append({
                                    "id": str(uuid.uuid4()),
                                    "class": class_name,
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "model": "best (1).pt",
                                    "type": "weapon"
                                })
            except Exception as e:
                logger.error(f"❌ Weapon detection error: {e}")
        
        # === GUN FUSION DETECTION ===
        elif model_type == 'gun-fusion':
            try:
                # Run best.pt (50% threshold)
                results = model(image, conf=0.50, verbose=False)
                for result in results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls].lower()
                        
                        if 'pistol' in class_name or 'gun' in class_name:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detections.append({
                                "id": str(uuid.uuid4()),
                                "class": f"{class_name} (best.pt)",
                                "confidence": conf,
                                "bbox": [x1, y1, x2, y2],
                                "model": "best.pt@50%",
                                "type": "gun_fusion"
                            })
                
                # Run best (1).pt (30% threshold)
                if best1_model:
                    results1 = best1_model(image, conf=0.30, verbose=False)
                    for result in results1:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = best1_model.names[cls].lower()
                            
                            if 'pistol' in class_name or 'gun' in class_name:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append({
                                    "id": str(uuid.uuid4()),
                                    "class": f"{class_name} (best1.pt)",
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "model": "best (1).pt@30%",
                                    "type": "gun_fusion"
                                })
                
                # Run best (8).pt (30% threshold)
                if best8_model:
                    results8 = best8_model(image, conf=0.30, verbose=False)
                    for result in results8:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = best8_model.names[cls].lower()
                            
                            if 'pistol' in class_name or 'gun' in class_name:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append({
                                    "id": str(uuid.uuid4()),
                                    "class": f"{class_name} (best8.pt)",
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "model": "best (8).pt@30%",
                                    "type": "gun_fusion"
                                })
                
                # Run thermal model if available (20% threshold)
                if thermal_model:
                    results_thermal = thermal_model(image, conf=0.20, verbose=False)
                    for result in results_thermal:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detections.append({
                                "id": str(uuid.uuid4()),
                                "class": "thermal_gun",
                                "confidence": conf,
                                "bbox": [x1, y1, x2, y2],
                                "model": "thermal.pt@20%",
                                "type": "gun_fusion"
                            })
            except Exception as e:
                logger.error(f"❌ Gun fusion detection error: {e}")
        
        # === DRONE DETECTION (for images) ===
        elif model_type == 'drone':
            try:
                if drone_model:
                    results = drone_model(image, conf=0.50, verbose=False)
                    for result in results:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = drone_model.names[cls].lower()
                            
                            if 'person' in class_name or 'pedestrian' in class_name:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append({
                                    "id": str(uuid.uuid4()),
                                    "class": class_name,
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "model": "drone.pt",
                                    "type": "drone_person"
                                })
                else:
                    logger.warning("Drone model not loaded")
            except Exception as e:
                logger.error(f"❌ Drone detection error: {e}")
        
        # === THERMAL HUMAN DETECTION ===
        elif model_type == 'thermal-human':
            try:
                # Run thermal human model
                if thermal_human_model:
                    results = thermal_human_model(image, conf=0.30, verbose=False)
                    for result in results:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = thermal_human_model.names[cls].lower()
                            
                            if 'person' in class_name or 'human' in class_name:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append({
                                    "id": str(uuid.uuid4()),
                                    "class": "thermal_human",
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "model": "thermalhuman.pt",
                                    "type": "thermal_human"
                                })
                else:
                    logger.warning("Thermal human model not loaded")
                
                # Also run standard people model
                if people_model:
                    results = people_model(image, conf=0.50, verbose=False)
                    for result in results:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = people_model.names[cls].lower()
                            
                            if class_name == 'person':
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append({
                                    "id": str(uuid.uuid4()),
                                    "class": "person",
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "model": "yolov8n.pt",
                                    "type": "person"
                                })
            except Exception as e:
                logger.error(f"❌ Thermal human detection error: {e}")
        
        # === ALL MODELS (COMPREHENSIVE) ===
        elif model_type == 'all':
            try:
                # 1. Weapon detection (best.pt)
                results = model(image, conf=0.50, verbose=False)
                for result in results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls].lower()
                        if 'pistol' in class_name or 'gun' in class_name or 'knife' in class_name:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detections.append({
                                "id": str(uuid.uuid4()),
                                "class": class_name,
                                "confidence": conf,
                                "bbox": [x1, y1, x2, y2],
                                "model": "best.pt",
                                "type": "weapon"
                            })
                
                # 2. Verification model (best (1).pt)
                if best1_model:
                    results1 = best1_model(image, conf=0.30, verbose=False)
                    for result in results1:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = best1_model.names[cls].lower()
                            if 'pistol' in class_name or 'gun' in class_name or 'knife' in class_name:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append({
                                    "id": str(uuid.uuid4()),
                                    "class": class_name,
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "model": "best (1).pt",
                                    "type": "weapon"
                                })
                
                # 3. People detection
                if people_model:
                    results_people = people_model(image, conf=0.50, verbose=False)
                    for result in results_people:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = people_model.names[cls].lower()
                            if class_name == 'person':
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append({
                                    "id": str(uuid.uuid4()),
                                    "class": "person",
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "model": "yolov8n.pt",
                                    "type": "person"
                                })
                
                # 4. Violence detection (Roboflow)
                temp_infer_path = UPLOAD_DIR / f"temp_infer_{uuid.uuid4()}.jpg"
                cv2.imwrite(str(temp_infer_path), image)
                result = CLIENT.infer(str(temp_infer_path), model_id="violence-not_violence-ziv7b/2")
                if temp_infer_path.exists():
                    temp_infer_path.unlink()
                
                if result and 'predictions' in result:
                    predictions = result['predictions']
                    if isinstance(predictions, dict):
                        predicted_class = None
                        max_confidence = 0.0
                        for class_name, class_data in predictions.items():
                            if isinstance(class_data, dict) and 'confidence' in class_data:
                                conf = class_data['confidence']
                                if conf > max_confidence:
                                    max_confidence = conf
                                    predicted_class = class_name
                        
                        if predicted_class and max_confidence > 0.5:
                            img_h, img_w = image.shape[:2]
                            detections.append({
                                "id": str(uuid.uuid4()),
                                "class": predicted_class,
                                "confidence": max_confidence,
                                "bbox": [0, 0, img_w, img_h],
                                "model": "roboflow-violence",
                                "type": "violence"
                            })
                
                # 5. Top-view detection
                if topview_model:
                    results_topview = topview_model(image, conf=0.30, verbose=False)
                    for result in results_topview:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = topview_model.names[cls].lower()
                            if 'person' in class_name or 'pedestrian' in class_name:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append({
                                    "id": str(uuid.uuid4()),
                                    "class": "person (topview)",
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "model": "best (9).pt",
                                    "type": "person"
                                })
                
                logger.info(f"🔍 All models run - {len(detections)} total detections")
            except Exception as e:
                logger.error(f"❌ All models detection error: {e}")
        
        # === VIOLENCE DETECTION ===
        elif model_type in ['all', 'violence']:
            try:
                # Save temporary image for inference
                temp_infer_path = UPLOAD_DIR / f"temp_infer_{uuid.uuid4()}.jpg"
                cv2.imwrite(str(temp_infer_path), image)
                
                logger.info("🚨 Sending image to Roboflow for violence detection...")
                result = CLIENT.infer(str(temp_infer_path), model_id="violence-not_violence-ziv7b/2")
                logger.info(f"✅ Roboflow response: {result}")
                
                # Clean up temp file
                if temp_infer_path.exists():
                    temp_infer_path.unlink()
                
                # Parse result - This is a CLASSIFICATION model
                # Response format: {'predictions': {'class_name': {'confidence': 0.99, 'class_id': 1}, ...}}
                
                if result and 'predictions' in result:
                    predictions = result['predictions']
                    
                    # Classification format: {'violence': {'confidence': 0.99, 'class_id': 1}, 'non_violence': {...}}
                    if isinstance(predictions, dict):
                        # Find the predicted class (highest confidence)
                        predicted_class = None
                        max_confidence = 0.0
                        
                        for class_name, class_data in predictions.items():
                            if isinstance(class_data, dict) and 'confidence' in class_data:
                                conf = class_data['confidence']
                                if conf > max_confidence:
                                    max_confidence = conf
                                    predicted_class = class_name
                        
                        # Create a detection entry with the classification result
                        # Use the whole image as bbox for visualization
                        if predicted_class and max_confidence > 0.5:  # Threshold for positive detection
                            img_h, img_w = image.shape[:2]
                            
                            # Create bbox covering the whole image
                            detections.append({
                                "id": str(uuid.uuid4()),
                                "class": predicted_class,
                                "confidence": max_confidence,
                                "bbox": [0, 0, img_w, img_h],
                                "model": "roboflow-violence-classifier",
                                "type": "violence_classification"
                            })
                            
                            logger.info(f"🚨 Violence Classification: {predicted_class} @ {max_confidence:.2%}")
                    
                    # Detection format (list of bounding boxes) - fallback
                    elif isinstance(predictions, list):
                        for pred in predictions:
                            if 'x' in pred and 'y' in pred:
                                x = pred['x']
                                y = pred['y']
                                w = pred['width']
                                h = pred['height']
                                conf = pred['confidence']
                                class_name = pred['class']
                                
                                # Convert center x,y to top-left x1,y1
                                x1 = int(x - w / 2)
                                y1 = int(y - h / 2)
                                x2 = int(x + w / 2)
                                y2 = int(y + h / 2)
                                
                                detections.append({
                                    "id": str(uuid.uuid4()),
                                    "class": class_name,
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "model": "roboflow-violence-detector",
                                    "type": "violence"
                                })

            except Exception as e:
                logger.error(f"❌ Roboflow Inference Error: {e}")
                # Fallback or just log
            
        # === PERSON DETECTION (General) ===
        elif model_type == 'person':
            try:
                if people_model:
                    results = people_model(image, conf=0.50, verbose=False)
                    for result in results:
                        if result.boxes is None:
                            continue
                        for box in result.boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = people_model.names[cls].lower()
                            
                            if class_name == 'person':
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append({
                                    "id": str(uuid.uuid4()),
                                    "class": "person",
                                    "confidence": conf,
                                    "bbox": [x1, y1, x2, y2],
                                    "model": "yolov8n.pt",
                                    "type": "person"
                                })
                else:
                    logger.warning("People model not loaded")
            except Exception as e:
                logger.error(f"❌ Person detection error: {e}")
            
        # If any detections were made, annotate the image
        if len(detections) > 0 and image is not None:
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class']} {det['confidence']:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            annotated_fname = f"test_annotated_{uuid.uuid4()}.jpg"
            annotated_path = DETECTIONS_DIR / annotated_fname
            cv2.imwrite(str(annotated_path), image)
            annotated_image_url = f"/detections/{annotated_fname}"
        else:
            annotated_image_url = None
            
        return {
            "status": "ok",
            "detections": detections,
            "annotated_image_url": annotated_image_url,
            "processing_time": 0,
            "message": "Image processed"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in test_upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the uploaded file
        if file_path.exists():
            file_path.unlink()

@api_router.post("/detect/upload")
async def detect_upload(file: UploadFile = File(...), camera_id: str = "", camera_name: str = "Upload", lat: float = 18.5204, lng: float = 73.8567):
    file_path = UPLOAD_DIR / file.filename
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    if file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        cap = cv2.VideoCapture(str(file_path))
        all_detections = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:
                # Determine mode based on camera_id
                mode = "normal"
                if "topview" in camera_id.lower():
                    mode = "topview"
                elif "people" in camera_id.lower():
                    mode = "people"
                    
                dets = await process_detection(frame, camera_id or "upload", camera_name, {"lat": lat, "lng": lng}, detection_mode=mode)
                all_detections.extend(dets)
            frame_count += 1
        cap.release()
        # Remove _id field for JSON serialization
        for det in all_detections:
            det.pop('_id', None)
        # Provide an annotated_image top-level field (first available)
        annotated_image = None
        for det in all_detections:
            if det.get('image_path'):
                annotated_image = det.get('image_path')
                break
        return {"detections": all_detections, "frames_processed": frame_count, "annotated_image": annotated_image}
    else:
        image = cv2.imread(str(file_path))
        # Determine mode based on camera_id
        mode = "normal"
        if "topview" in camera_id.lower():
            mode = "topview"
        elif "people" in camera_id.lower():
            mode = "people"
            
        detections = await process_detection(image, camera_id or "upload", camera_name, {"lat": lat, "lng": lng}, detection_mode=mode)
        # Remove _id field for JSON serialization
        for det in detections:
            det.pop('_id', None)
        # Provide an annotated_image top-level field (first available)
        annotated_image = None
        for det in detections:
            if det.get('image_path'):
                annotated_image = det.get('image_path')
                break
        return {"detections": detections, "annotated_image": annotated_image}

class DetectionModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    camera_id: str = "unknown"
    camera_name: str = "unknown"
    detection_type: str = "unknown"
    confidence: float = 0.0
    image_path: str = ""
    location: dict = Field(default_factory=lambda: {"lat": 0, "lng": 0})

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    alert_sent: bool = False

@api_router.get("/detections", response_model=List[DetectionModel])
async def get_detections():
    records = await db.detections.find({}, {"_id": 0}).sort("timestamp", -1).to_list(1000)
    return records

@api_router.get("/detections/{detection_id}")
async def get_detection(detection_id: str):
    det = await db.detections.find_one({"id": detection_id}, {"_id": 0})
    if not det:
        raise HTTPException(status_code=404, detail="Detection not found")
    return det



# ===== ADMIN VERIFICATION ENDPOINTS =====

@api_router.get("/pending-verifications")
async def get_pending_verifications(
    status: str = "pending",
    limit: int = 50,
    skip: int = 0
):
    """Get pending admin verifications"""
    try:
        query = {}
        if status and status != "all":
            query["status"] = status
        
        verifications = await db.pending_verifications.find(
            query,
            {"_id": 0}
        ).sort("timestamp", -1).skip(skip).limit(limit).to_list(limit)
        
        total = await db.pending_verifications.count_documents(query)
        
        return {
            "verifications": verifications,
            "total": total,
            "limit": limit,
            "skip": skip
        }
    except Exception as e:
        logger.error(f"Error fetching pending verifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/pending-verifications/{verification_id}")
async def get_verification_details(verification_id: str):
    """Get detailed verification data"""
    try:
        verification = await db.pending_verifications.find_one(
            {"id": verification_id},
            {"_id": 0}
        )
        
        if not verification:
            raise HTTPException(status_code=404, detail="Verification not found")
        
        return verification
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching verification details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class VerificationDecision(BaseModel):
    decision: str
    admin_id: str = "admin"
    notes: str = ""

@api_router.post("/verify-detection/{verification_id}")
async def verify_detection(
    verification_id: str,
    data: VerificationDecision
):
    """Admin confirms or rejects a detection"""
    try:
        if data.decision not in ["confirm", "reject"]:
            raise HTTPException(status_code=400, detail="Decision must be 'confirm' or 'reject'")
        
        verification = await db.pending_verifications.find_one({"id": verification_id})
        
        if not verification:
            raise HTTPException(status_code=404, detail="Verification not found")
        
        if verification.get("status") != "pending":
            raise HTTPException(status_code=400, detail="Verification already processed")
        
        # Update verification record
        admin_decision = {
            "admin_id": data.admin_id,
            "decision": data.decision,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": data.notes
        }
        
        new_status = "confirmed" if data.decision == "confirm" else "rejected"
        
        await db.pending_verifications.update_one(
            {"id": verification_id},
            {
                "$set": {
                    "status": new_status,
                    "admin_decision": admin_decision,
                    "blockchain_verified": True
                }
            }
        )
        
        # Update blockchain with verification
        blockchain_update = {
            "id": str(uuid.uuid4()),
            "verification_id": verification_id,
            "admin_decision": admin_decision,
            "final_status": new_status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        

        
        # Update detection record
        await db.detections.update_one(
            {"verification_id": verification_id},
            {
                "$set": {
                    "verification_status": f"admin_{new_status}",
                    "admin_decision": admin_decision
                }
            }
        )
        
        # Log for model retraining
        logger.info(f"📊 Admin {decision}: {verification_id} - Logged for retraining")
        
        return {
            "status": "success",
            "verification_id": verification_id,
            "decision": decision,
            "blockchain_updated": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing verification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/verification-stats")
async def get_verification_stats():
    """Get verification statistics"""
    try:
        pending_count = await db.pending_verifications.count_documents({"status": "pending"})
        confirmed_count = await db.pending_verifications.count_documents({"status": "confirmed"})
        rejected_count = await db.pending_verifications.count_documents({"status": "rejected"})
        
        return {
            "pending": pending_count,
            "confirmed": confirmed_count,
            "rejected": rejected_count,
            "total": pending_count + confirmed_count + rejected_count
        }
    except Exception as e:
        logger.error(f"Error fetching verification stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/police-stations")
async def get_police_stations():
    stations = [
        {"name": "Pune City Police Headquarters", "lat": 18.5204, "lng": 73.8567},
        {"name": "Shivajinagar Police Station", "lat": 18.5309, "lng": 73.8433},
        {"name": "Deccan Police Station", "lat": 18.5165, "lng": 73.8460}
    ]
    return stations


# Cache for Overpass API results
overpass_cache = {}
CACHE_DURATION = 3600  # 1 hour cache

@api_router.get("/emergency-context")
async def emergency_context(lat: float, lng: float, radius: float = 10.0):
    """Return nearest police/hospital/fire station info within specified radius (in km)
    
    Args:
        lat: User latitude
        lng: User longitude
        radius: Search radius in kilometers (default: 10km)
    """
    
    import math
    
    def calculate_distance(lat1, lng1, lat2, lng2):
        """Calculate distance in km using Haversine formula"""
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    logger.info(f"🚑 Fetching emergency context for {lat}, {lng} within {radius}km radius")
    
    # Comprehensive Pune area data
    police_data = [
        {"name": "Pune City Police Commissionerate", "phone": "100", "lat": 18.5204, "lng": 73.8567},
        {"name": "Shivajinagar Police Station", "phone": "020-25532854", "lat": 18.5314, "lng": 73.8446},
        {"name": "Deccan Police Station", "phone": "020-25533695", "lat": 18.5165, "lng": 73.8460},
        {"name": "Kothrud Police Station", "phone": "020-25434101", "lat": 18.5074, "lng": 73.8077},
        {"name": "Swargate Police Station", "phone": "020-24456503", "lat": 18.5018, "lng": 73.8636},
        {"name": "Hadapsar Police Station", "phone": "020-26993670", "lat": 18.5089, "lng": 73.9260},
        {"name": "Koregaon Park Police Station", "phone": "020-26059157", "lat": 18.5362, "lng": 73.8958}
    ]
    
    # General hospitals
    hospital_data = [
        {"name": "City Hospital", "phone": "020-24458181", "lat": 18.5196, "lng": 73.8553, "type": "general"},
        {"name": "Sancheti Hospital", "phone": "020-26051983", "lat": 18.5304, "lng": 73.8997, "type": "general"},
        {"name": "Aditya Birla Hospital", "phone": "020-71074444", "lat": 18.5642, "lng": 73.7769, "type": "general"},
        {"name": "Sahyadri Hospital (Deccan)", "phone": "020-67206666", "lat": 18.5174, "lng": 73.8422, "type": "general"}
    ]
    
    # Multi-specialty hospitals
    multispecialty_data = [
        {"name": "Ruby Hall Clinic", "phone": "020-66455100", "lat": 18.5308, "lng": 73.8774, "type": "multispecialty", "specialties": ["Cardiology", "Neurology", "Oncology", "Orthopedics"]},
        {"name": "Jehangir Hospital", "phone": "020-66811000", "lat": 18.5293, "lng": 73.8744, "type": "multispecialty", "specialties": ["Emergency", "ICU", "Surgery", "Pediatrics"]},
        {"name": "Deenanath Mangeshkar Hospital", "phone": "020-66206000", "lat": 18.4996, "lng": 73.8265, "type": "multispecialty", "specialties": ["Cardiac", "Trauma", "Neurosurgery", "Oncology"]},
        {"name": "Sahyadri Super Speciality Hospital", "phone": "020-67206666", "lat": 18.5588, "lng": 73.7816, "type": "multispecialty", "specialties": ["Cardiac", "Neuro", "Orthopedic", "Cancer"]},
        {"name": "Poona Hospital & Research Centre", "phone": "020-66969696", "lat": 18.5314, "lng": 73.8883, "type": "multispecialty", "specialties": ["Multi-organ Transplant", "Cardiac", "Neuro", "Oncology"]},
        {"name": "KEM Hospital", "phone": "020-26123296", "lat": 18.5287, "lng": 73.8814, "type": "multispecialty", "specialties": ["Emergency", "Trauma", "Surgery", "Critical Care"]}
    ]
    
    fire_data = [
        {"name": "Central Fire Station", "phone": "101", "lat": 18.5123, "lng": 73.8645},
        {"name": "Deccan Fire Station", "phone": "101", "lat": 18.5168, "lng": 73.8350},
        {"name": "Kothrud Fire Station", "phone": "101", "lat": 18.5047, "lng": 73.8137}
    ]
    
    # Calculate distances and filter by radius
    police_stations = []
    for p in police_data:
        dist_km = calculate_distance(lat, lng, p["lat"], p["lng"])
        if dist_km <= radius:  # Filter by radius
            police_stations.append({
                "name": p["name"],
                "phone": p["phone"],
                "lat": p["lat"],
                "lng": p["lng"],
                "distance_km": round(dist_km, 2),
                "distance_m": int(dist_km * 1000)
            })
    
    hospitals = []
    for h in hospital_data:
        dist_km = calculate_distance(lat, lng, h["lat"], h["lng"])
        if dist_km <= radius:  # Filter by radius
            hospitals.append({
                "name": h["name"],
                "phone": h["phone"],
                "lat": h["lat"],
                "lng": h["lng"],
                "type": h["type"],
                "distance_km": round(dist_km, 2),
                "distance_m": int(dist_km * 1000)
            })
    
    multispecialty_hospitals = []
    for m in multispecialty_data:
        dist_km = calculate_distance(lat, lng, m["lat"], m["lng"])
        if dist_km <= radius:  # Filter by radius
            multispecialty_hospitals.append({
                "name": m["name"],
                "phone": m["phone"],
                "lat": m["lat"],
                "lng": m["lng"],
                "type": m["type"],
                "specialties": m["specialties"],
                "distance_km": round(dist_km, 2),
                "distance_m": int(dist_km * 1000)
            })
    
    fire_stations = []
    for f in fire_data:
        dist_km = calculate_distance(lat, lng, f["lat"], f["lng"])
        if dist_km <= radius:  # Filter by radius
            fire_stations.append({
                "name": f["name"],
                "phone": f["phone"],
                "lat": f["lat"],
                "lng": f["lng"],
                "distance_km": round(dist_km, 2),
                "distance_m": int(dist_km * 1000)
            })
    
    # Sort by distance
    police_stations.sort(key=lambda x: x["distance_km"])
    hospitals.sort(key=lambda x: x["distance_km"])
    multispecialty_hospitals.sort(key=lambda x: x["distance_km"])
    fire_stations.sort(key=lambda x: x["distance_km"])
    
    # Return only the NEAREST of each type (first item after sorting)
    return {
        "police_stations": police_stations[:1] if police_stations else [],  # Nearest police station
        "hospitals": hospitals[:1] if hospitals else [],  # Nearest general hospital
        "multispecialty_hospitals": multispecialty_hospitals[:1] if multispecialty_hospitals else [],  # Nearest multi-specialty
        "fire_stations": fire_stations[:1] if fire_stations else [],  # Nearest fire station
        "radius_km": radius,
        "total_found": {
            "police": len(police_stations),
            "hospitals": len(hospitals),
            "multispecialty": len(multispecialty_hospitals),
            "fire": len(fire_stations)
        },
        "emergency_numbers": {
            "police": "100",
            "ambulance": "108",
            "fire": "101"
        }
    }


# Webcam monitoring endpoints
@api_router.post("/start")
async def start_monitoring():
    """Start the detection system"""
    global camera, is_running
    
    if not is_running:
        is_running = True
        camera = VideoCamera()
        start_save_worker()  # Start the background worker
        start_verification_worker()  # Start dual verification worker
        logger.info("✅ Detection system started")
        return {"status": "started"}
    return {"status": "already_running"}


@api_router.post("/stop")
async def stop_monitoring():
    """Stop the detection system"""
    global camera, is_running
    
    if is_running:
        is_running = False
        stop_save_worker()  # Stop the background worker
        stop_verification_worker()  # Stop dual verification worker
        if camera is not None:
            del camera
            camera = None
        
        # Reset stats
        detection_stats.update({
            'guns': 0,
            'knives': 0,
            'thermal_guns': 0,
            'people': 0,
            'alert': False,
            'alert_message': ''
        })
        logger.info("⏹️ Detection system stopped")
        return {"status": "stopped"}
    return {"status": "not_running"}


@api_router.get("/status")
async def get_status():
    """Get current detection status"""
    return detection_stats


@api_router.get("/video_feed")
async def video_feed():
    """Video streaming route"""
    return StreamingResponse(generate_frames(),
                           media_type="multipart/x-mixed-replace; boundary=frame")


# ============================================================================
# MULTI-CAMERA API ENDPOINTS
# ============================================================================

class CameraCreateRequest(BaseModel):
    """Request model for creating a new camera"""
    name: str
    type: str  # webcam, rtsp, ip, file
    source: str  # camera index, URL, or file path
    location: dict = {"lat": 0.0, "lng": 0.0}
    settings: dict = {}


class CameraUpdateRequest(BaseModel):
    """Request model for updating camera"""
    name: Optional[str] = None
    location: Optional[dict] = None
    settings: Optional[dict] = None


@api_router.post("/cameras")
async def create_camera(camera_data: CameraCreateRequest):
    """Register a new camera"""
    logger.info(f"📝 Received camera creation request: {camera_data}")
    try:
        # Validate camera type
        try:
            camera_type = CameraType(camera_data.type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid camera type: {camera_data.type}")
        
        # Register camera
        camera_id = camera_registry.register_camera(
            name=camera_data.name,
            camera_type=camera_type,
            source=camera_data.source,
            location=camera_data.location,
            settings=camera_data.settings
        )
        
        # Save to database
        camera_doc = {
            "id": camera_id,
            "name": camera_data.name,
            "type": camera_data.type,
            "source": camera_data.source,
            "location": camera_data.location,
            "settings": camera_data.settings,
            "status": "inactive",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.cameras.insert_one(camera_doc)
        
        logger.info(f"✅ Created camera: {camera_id} - {camera_data.name}")
        return {"status": "success", "camera_id": camera_id, "message": "Camera registered successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating camera: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/cameras")
async def list_cameras():
    """List all registered cameras"""
    try:
        # Get from registry (includes runtime status)
        cameras = camera_registry.list_cameras()
        return {"cameras": cameras, "count": len(cameras)}
    except Exception as e:
        logger.error(f"Error listing cameras: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/cameras/{camera_id}")
async def get_camera(camera_id: str):
    """Get details of a specific camera"""
    try:
        camera = camera_registry.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        return camera.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera {camera_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.put("/cameras/{camera_id}")
async def update_camera(camera_id: str, update_data: CameraUpdateRequest):
    """Update camera configuration"""
    try:
        camera = camera_registry.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        # Update fields
        if update_data.name:
            camera.name = update_data.name
        if update_data.location:
            camera.location = update_data.location
        if update_data.settings:
            camera.settings.update(update_data.settings)
        
        # Update in database
        update_fields = {}
        if update_data.name:
            update_fields["name"] = update_data.name
        if update_data.location:
            update_fields["location"] = update_data.location
        if update_data.settings:
            update_fields["settings"] = update_data.settings
        
        if update_fields:
            await db.cameras.update_one(
                {"id": camera_id},
                {"$set": update_fields}
            )
        
        logger.info(f"✅ Updated camera: {camera_id}")
        return {"status": "success", "message": "Camera updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating camera {camera_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    """Delete a camera"""
    try:
        # Unregister from registry
        success = camera_registry.unregister_camera(camera_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        # Delete from database
        await db.cameras.delete_one({"id": camera_id})
        
        logger.info(f"🗑️ Deleted camera: {camera_id}")
        return {"status": "success", "message": "Camera deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting camera {camera_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/cameras/{camera_id}/start")
async def start_camera(camera_id: str):
    """Start a specific camera"""
    try:
        success = camera_registry.start_camera(camera_id)
        if not success:
            camera = camera_registry.get_camera(camera_id)
            error_msg = camera.error_message if camera else "Camera not found"
            raise HTTPException(status_code=400, detail=f"Failed to start camera: {error_msg}")
        
        # Update status in database
        await db.cameras.update_one(
            {"id": camera_id},
            {"$set": {"status": "active", "last_active": datetime.now(timezone.utc).isoformat()}}
        )
        
        logger.info(f"✅ Started camera: {camera_id}")
        return {"status": "success", "message": "Camera started successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting camera {camera_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/cameras/{camera_id}/stop")
async def stop_camera(camera_id: str):
    """Stop a specific camera"""
    try:
        success = camera_registry.stop_camera(camera_id)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to stop camera")
        
        # Update status in database
        await db.cameras.update_one(
            {"id": camera_id},
            {"$set": {"status": "inactive"}}
        )
        
        logger.info(f"⏹️ Stopped camera: {camera_id}")
        return {"status": "success", "message": "Camera stopped successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping camera {camera_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/cameras/{camera_id}/status")
async def get_camera_status(camera_id: str):
    """Get status and statistics for a specific camera"""
    try:
        camera = camera_registry.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        return {
            "camera_id": camera_id,
            "name": camera.name,
            "status": camera.status,
            "stats": camera.get_stats(),
            "error_message": camera.error_message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera status {camera_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def generate_camera_frames(camera_id: str):
    """Generate frames for a specific camera"""
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        logger.error(f"Camera {camera_id} not found for streaming")
        return
    
    logger.info(f"Starting video stream for camera {camera_id}")
    
    while camera.is_running:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # Route to appropriate detection function based on camera configuration
        detection_mode = camera.settings.get('detection_mode', 'default')
        
        if detection_mode == 'house_medicine':
            # House/Medicine: Person + Gun detection at 30%
            frame_with_detections, detections_to_save, people_detected = custom_camera_detection.detect_house_medicine(
                frame, camera_id=camera.camera_id
            )
        elif detection_mode == 'topview':
            # Top View: People counting with best(9).pt at 40%
            frame_with_detections, detections_to_save, people_detected = custom_camera_detection.detect_topview_people(
                frame, camera_id=camera.camera_id
            )
        elif detection_mode == 'gun_only':
            # Gun/Gun1/Best(1): Gun detection only with best.pt + best(1).pt at 30%
            frame_with_detections, detections_to_save, people_detected = custom_camera_detection.detect_gun_only(
                frame, camera_id=camera.camera_id
            )
        elif detection_mode == 'medicine_only':
            # Medicine: Firearm detection only (no person) with best.pt + best(1).pt at 30%
            frame_with_detections, detections_to_save, people_detected = custom_camera_detection.detect_medicine_only(
                frame, camera_id=camera.camera_id
            )
        elif detection_mode == 'violence_only':
            # Violence: Violence classification with best(2).pt at 50%
            frame_with_detections, detections_to_save, people_detected = custom_camera_detection.detect_violence_only(
                frame, camera_id=camera.camera_id
            )
        else:
            # Default: Use original detection logic (for live webcam, etc.)
            gun_conf = camera.settings.get('gun_conf', 0.80)
            verify_conf = camera.settings.get('verify_conf', 0.60)
            use_or_logic = camera.settings.get('use_or_logic', False)
            use_violence_model = camera.settings.get('use_violence_model', False)
            
            frame_with_detections, detections_to_save, people_detected = detect_weapons_and_people(
                frame, 
                gun_conf_threshold=gun_conf, 
                verification_conf_threshold=verify_conf,
                use_or_logic=use_or_logic,
                use_violence_model=use_violence_model,
                camera_id=camera.camera_id,
                camera_type=camera.camera_type
            )
        
        # Update camera stats
        guns = sum(1 for d in detections_to_save if d.get('detection_type') == 'pistol')
        knives = sum(1 for d in detections_to_save if d.get('detection_type') == 'knife')
        # Note: thermal guns and people counting would need to be integrated here
        camera.update_stats(guns=guns, knives=knives, people=people_detected)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame_with_detections, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS

@api_router.get("/api/verify/{detection_id}")
async def verify_detection(detection_id: str):
    """Verify detection integrity using IPFS and SHA-256"""
    try:
        # Find detection by ID (checking both string ID and ObjectId)
        detection = await db.detections.find_one({"id": detection_id})
        if not detection:
            try:
                from bson import ObjectId
                detection = await db.detections.find_one({"_id": ObjectId(detection_id)})
            except:
                pass
        
        if not detection:
            raise HTTPException(status_code=404, detail="Detection not found")
            
        ipfs_data = detection.get('ipfs_data')
        if not ipfs_data or 'ipfs_hash' not in ipfs_data:
            return {
                "verified": False,
                "error": "No IPFS data found for this detection (Evidence might pre-date IPFS integration)"
            }
            
        # Verify integrity
        verification = ipfs.verify_integrity(
            ipfs_hash=ipfs_data['ipfs_hash'],
            stored_sha256=ipfs_data['sha256_hash']
        )
        
        return {
            "detection_id": detection_id,
            "timestamp": detection.get('timestamp'),
            "ipfs_url": ipfs_data.get('url'),
            **verification
        }
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/api/test-alert")
async def test_twilio_alert_endpoint():
    """Test Twilio Alert Configuration"""
    try:
        logger.info("🧪 Testing Twilio Alert...")
        test_data = {
            'detection_type': 'TEST_ALERT',
            'confidence': 1.0,
            'camera_name': 'Manual Test',
            'timestamp': datetime.now().isoformat(),
            'location': {'lat': 0, 'lng': 0}
        }
        
        # Reload utils.alerts to ensure it picks up env vars if they were late
        import importlib
        import utils.alerts
        importlib.reload(utils.alerts)
        from utils.alerts import send_twilio_alert as send_alert_reloaded
        
        success = send_alert_reloaded(test_data)
        
        if success:
            return {"status": "success", "message": "Alert sent successfully"}
        else:
            return {"status": "error", "message": "Failed to send alert (check logs)"}
            
    except Exception as e:
        logger.error(f"Test alert failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@api_router.get("/cameras/{camera_id}/video_feed")
async def camera_video_feed(camera_id: str):
    """Video streaming route for a specific camera"""
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    if camera.status != CameraStatus.ACTIVE:
        raise HTTPException(status_code=400, detail=f"Camera {camera_id} is not active")
    
    return StreamingResponse(
        generate_camera_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# Mount and middleware
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/detections", StaticFiles(directory=str(DETECTIONS_DIR)), name="detections")
app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)
