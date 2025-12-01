from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
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
from functools import lru_cache
from math import radians, sin, cos, sqrt, atan2

# Import camera manager for multi-camera support
from camera_manager import camera_registry, CameraType, CameraStatus

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
PEOPLE_MODEL_PATH = ROOT_DIR / 'yolov8m.pt'

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    logger.info(f"Loading YOLO model from {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    logger.info(f"Model loaded successfully. Classes: {model.names}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Load best (1).pt model for comparison
best1_model = None
try:
    if BEST1_MODEL_PATH.exists():
        logger.info(f"Loading best (1) model from {BEST1_MODEL_PATH}")
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

# Roboflow Thermal Gun Detection Clients
thermal_gun_client_1 = None  # shield-ai/2
thermal_gun_client_2 = None  # thermal-pistol/1

try:
    thermal_gun_client_1 = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="X1rITTS4fXVEbuQn3TxQ"
    )
    logger.info("✅ Roboflow Thermal Gun Detection client 1 (shield-ai/2) initialized successfully")
except Exception as e:
    logger.warning(f"⚠️ Failed to initialize Roboflow client 1: {e}")

try:
    thermal_gun_client_2 = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="X1rITTS4fXVEbuQn3TxQ"
    )
    logger.info("✅ Roboflow Thermal Gun Detection client 2 (thermal-pistol/1) initialized successfully")
except Exception as e:
    logger.warning(f"⚠️ Failed to initialize Roboflow client 2: {e}")

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
    'last_alert_time': None
}
frame_lock = threading.Lock()
current_frame = None
alert_cooldown = 30  # seconds between alerts

# Detection queue for background saving
detection_queue = Queue()
save_worker_thread = None
save_worker_running = False


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
async def create_blockchain_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

async def add_to_blockchain(detection_data: dict) -> str:
    last_block = await db.blockchain.find_one(sort=[("timestamp", -1)])
    previous_hash = last_block['hash'] if last_block else "0" * 64
    block_data = {
        "detection_id": detection_data['id'],
        "previous_hash": previous_hash,
        "data": detection_data,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    block_hash = await create_blockchain_hash(block_data)
    block_data['hash'] = block_hash
    await db.blockchain.insert_one(block_data)
    return block_hash

async def send_twilio_alert(_: dict):
    logger.debug("Twilio alerts are disabled in this build")
    return

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

def detect_thermal_guns(frame) -> tuple:
    """Detect thermal guns using both Roboflow models and draw bounding boxes"""
    THERMAL_GUN_CONFIDENCE_THRESHOLD = 0.30  # 30% confidence threshold
    
    if frame is None:
        logger.warning("Empty frame received for thermal gun detection")
        return 0, [], frame
    
    try:
        # Encode frame to JPEG for API
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.error("Failed to encode frame for thermal gun detection")
            return 0, [], frame
        
        thermal_guns_detected = 0
        detections_to_save = []
        annotated_frame = frame.copy()
        img_h, img_w = frame.shape[:2]
        
        # Convert buffer to base64 string (used for both models)
        image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        # Model 1: shield-ai/2
        if thermal_gun_client_1 is not None:
            try:
                logger.info("🌡️ Calling Roboflow API Model 1 (shield-ai/2)...")
                result1 = thermal_gun_client_1.infer(
                    image_base64,
                    model_id="shield-ai/2"
                )
                
                if result1 and 'predictions' in result1:
                    predictions = result1['predictions']
                    logger.info(f"📊 Model 1 returned {len(predictions)} predictions")
                    
                    for pred in predictions:
                        confidence = pred.get('confidence', 0)
                        
                        if confidence < THERMAL_GUN_CONFIDENCE_THRESHOLD:
                            continue
                        
                        thermal_guns_detected += 1
                        
                        # Extract bounding box coordinates
                        x = pred.get('x', 0)
                        y = pred.get('y', 0)
                        width = pred.get('width', 0)
                        height = pred.get('height', 0)
                        
                        x1 = int(max(0, x - width / 2))
                        y1 = int(max(0, y - height / 2))
                        x2 = int(min(img_w, x + width / 2))
                        y2 = int(min(img_h, y + height / 2))
                        
                        logger.info(f"🔫 THERMAL GUN (Model 1) DETECTED with confidence {confidence:.2f}")
                        
                        # Draw bounding box - Magenta for Model 1
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                        cv2.putText(annotated_frame, f'THERMAL-1 {confidence:.2f}', (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
                        
                        detection_id = str(uuid.uuid4())
                        detections_to_save.append({
                            "id": detection_id,
                            "detection_type": "thermal_gun_model1",
                            "model": "shield-ai/2",
                            "confidence": confidence,
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "camera_id": "webcam",
                            "camera_name": "Live Webcam",
                            "location": {"lat": 0, "lng": 0},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "alert_sent": False
                        })
            except Exception as api_error:
                logger.error(f"❌ Model 1 API error: {api_error}", exc_info=True)
        
        # Model 2: thermal-pistol/1
        if thermal_gun_client_2 is not None:
            try:
                logger.info("🌡️ Calling Roboflow API Model 2 (thermal-pistol/1)...")
                result2 = thermal_gun_client_2.infer(
                    image_base64,
                    model_id="thermal-pistol/1"
                )
                
                if result2 and 'predictions' in result2:
                    predictions = result2['predictions']
                    logger.info(f"📊 Model 2 returned {len(predictions)} predictions")
                    
                    for pred in predictions:
                        confidence = pred.get('confidence', 0)
                        
                        if confidence < THERMAL_GUN_CONFIDENCE_THRESHOLD:
                            continue
                        
                        thermal_guns_detected += 1
                        
                        # Extract bounding box coordinates
                        x = pred.get('x', 0)
                        y = pred.get('y', 0)
                        width = pred.get('width', 0)
                        height = pred.get('height', 0)
                        
                        x1 = int(max(0, x - width / 2))
                        y1 = int(max(0, y - height / 2))
                        x2 = int(min(img_w, x + width / 2))
                        y2 = int(min(img_h, y + height / 2))
                        
                        logger.info(f"🔫 THERMAL GUN (Model 2) DETECTED with confidence {confidence:.2f}")
                        
                        # Draw bounding box - Cyan for Model 2
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                        cv2.putText(annotated_frame, f'THERMAL-2 {confidence:.2f}', (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                        
                        detection_id = str(uuid.uuid4())
                        detections_to_save.append({
                            "id": detection_id,
                            "detection_type": "thermal_gun_model2",
                            "model": "thermal-pistol/1",
                            "confidence": confidence,
                            "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                            "camera_id": "webcam",
                            "camera_name": "Live Webcam",
                            "location": {"lat": 0, "lng": 0},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "alert_sent": False
                        })
            except Exception as api_error:
                logger.error(f"❌ Model 2 API error: {api_error}", exc_info=True)
        
        if thermal_guns_detected == 0:
            logger.info("✅ No thermal guns detected in this frame")
        
        logger.info(f"🌡️ Total thermal guns detected: {thermal_guns_detected}")
        return thermal_guns_detected, detections_to_save, annotated_frame
    except Exception as e:
        logger.error(f"❌ Error in thermal gun detection: {e}", exc_info=True)
        return 0, [], frame

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


def detect_weapons_and_people(frame):
    """Detect weapons and count people in frame"""
    global detection_stats
    
    guns_detected = 0
    knives_detected = 0
    thermal_guns_detected = 0
    people_detected = 0
    detections_to_save = []
    
    # Detect weapons (guns and knives) - higher confidence to reduce false positives
    weapon_results = model(frame, conf=PERF_CONFIG['weapon_confidence'], verbose=False)

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

            # Draw bounding box for weapons
            if 'pistol' in class_name or 'gun' in class_name:
                # Apply threshold for guns (45%)
                if conf < 0.45:
                    continue
                guns_detected += 1
                color = (0, 0, 255)  # Red for guns
                label = f'GUN {conf:.2f}'
                # logger.info(f"🔴 GUN DETECTED with confidence {conf:.2f}")
                det_type = 'pistol'
            elif 'knife' in class_name:
                # Apply 45% confidence threshold for knives
                if conf < 0.45:
                    continue
                knives_detected += 1
                color = (0, 165, 255)  # Orange for knives
                label = f'KNIFE {conf:.2f}'
                # logger.info(f"🟠 KNIFE DETECTED with confidence {conf:.2f}")
                det_type = 'knife'
            else:
                continue

            # Store detection data for saving to DB
            detection_id = str(uuid.uuid4())
            detections_to_save.append({
                "id": detection_id,
                "detection_type": det_type,
                "confidence": conf,
                "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                "camera_id": "webcam",
                "camera_name": "Live Webcam",
                "location": {"lat": 0, "lng": 0},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert_sent": False
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Process best (1).pt model if available
    if best1_model is not None:
        try:
            best1_results = best1_model(frame, conf=PERF_CONFIG['weapon_confidence'], verbose=False)
            for result in best1_results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = best1_model.names[cls].lower()
                    
                    # Process gun detections
                    if any(keyword in class_name for keyword in ['pistol', 'gun']):
                        if conf < 0.45:
                            continue
                        guns_detected += 1
                        color = (255, 255, 0)  # Cyan for best (1)
                        label = f'BEST1-GUN {conf:.2f}'
                        det_type = 'pistol'
                    # Process knife detections
                    elif 'knife' in class_name:
                        if conf < 0.45:
                            continue
                        knives_detected += 1
                        color = (255, 0, 255)  # Magenta for best (1) knives
                        label = f'BEST1-KNIFE {conf:.2f}'
                        det_type = 'knife'
                    else:
                        continue
                    
                    # Store detection
                    detection_id = str(uuid.uuid4())
                    detections_to_save.append({
                        "id": detection_id,
                        "detection_type": det_type,
                        "model": "best1",
                        "confidence": conf,
                        "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                        "camera_id": "webcam",
                        "camera_name": "Live Webcam",
                        "location": {"lat": 0, "lng": 0},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "alert_sent": False
                    })
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except Exception as e:
            logger.error(f"Error with best (1) model: {e}")
    
    # Run people detection if enabled (count only, no boxes shown)
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
    overlay_height = 150
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (450, overlay_height), (255, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, f'Guns: {guns_detected}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f'Knives: {knives_detected}', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    cv2.putText(frame, f'Thermal Guns: {thermal_guns_detected}', (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(frame, f'People: {people_detected}', (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, (frame.shape[1] - 350, frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame, detections_to_save


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
            block_hash = await add_to_blockchain(detection)
            detection['blockchain_hash'] = block_hash
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
        frame, detections_to_save = detect_weapons_and_people(frame)
        
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
        
        # Encode frame with configured quality
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, PERF_CONFIG['jpeg_quality']])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Minimal delay for smoother streaming
        time.sleep(0.01)  # 10ms delay instead of FPS-based delay

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
                    annotated = result.plot()
                    cv2.imwrite(str(path), annotated)
                    image_path_val = f"/detections/{fname}"
                except Exception:
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
                det_type = model.names[cls]
                if det_type not in ['pistol', 'knife']:
                    continue
                
                # Apply 30% threshold for best.pt
                if conf < 0.30:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                if not validate_detection(det_type, conf, x1, y1, x2, y2, img_w, img_h):
                    continue
                
                detection_id = str(uuid.uuid4())
                fname = f"model1_{detection_id}.jpg"
                path = DETECTIONS_DIR / fname
                image_path_val = None
                try:
                    annotated = result.plot()
                    cv2.imwrite(str(path), annotated)
                    image_path_val = f"/detections/{fname}"
                except Exception:
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
                block_hash = await add_to_blockchain(detection_data)
                detection_data['blockchain_hash'] = block_hash
                await db.detections.insert_one(detection_data)
                await send_twilio_alert(detection_data)
                detection_data['alert_sent'] = True
                await db.detections.update_one({"id": detection_id}, {"$set": {"alert_sent": True}})
                detections_list.append(detection_data)
        
        logger.info(f"✅ best.pt detected: {len([d for d in detections_list if d.get('model') == 'best.pt'])} weapons")
    
    # Run best (1).pt model (Model 2) - if mode is 'normal' or 'model2'
    if detection_mode in ["normal", "model2"] and best1_model is not None:
        logger.info("💙 Running best (1).pt model...")
        results_model2 = best1_model(processed_image, conf=0.25, iou=0.5)
        
        for result in results_model2:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls not in best1_model.names:
                    continue
                class_name = best1_model.names[cls].lower()
                
                # Map class names to detection types
                if any(keyword in class_name for keyword in ['pistol', 'gun', 'machinegun', 'rifle', 'shotgun']):
                    det_type = 'pistol'
                else:
                    continue
                
                # Apply 30% threshold for best (1).pt  
                if conf < 0.30:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                if not validate_detection(det_type, conf, x1, y1, x2, y2, img_w, img_h):
                    continue
                
                detection_id = str(uuid.uuid4())
                fname = f"model2_{detection_id}.jpg"
                path = DETECTIONS_DIR / fname
                image_path_val = None
                try:
                    annotated = result.plot()
                    cv2.imwrite(str(path), annotated)
                    image_path_val = f"/detections/{fname}"
                except Exception:
                    image_path_val = None

                detection_data = {
                    "id": detection_id,
                    "camera_id": camera_id or "upload",
                    "camera_name": camera_name,
                    "detection_type": det_type,
                    "confidence": conf,
                    "model": "best (1).pt",  # Tag with model name
                    "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
                    "image_path": image_path_val,
                    "location": location,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "alert_sent": False
                }
                block_hash = await add_to_blockchain(detection_data)
                detection_data['blockchain_hash'] = block_hash
                await db.detections.insert_one(detection_data)
                await send_twilio_alert(detection_data)
                detection_data['alert_sent'] = True
                await db.detections.update_one({"id": detection_id}, {"$set": {"alert_sent": True}})
                detections_list.append(detection_data)
        
        logger.info(f"✅ best (1).pt detected: {len([d for d in detections_list if d.get('model') == 'best (1).pt'])} weapons")
    
    # Thermal detection is handled separately in the upload endpoint, 
    # but if we wanted to include it here we could. 
    # For now, the upload endpoint calls process_thermal_only separately.
    
    return detections_list
    
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
    """Process image with thermal detection ONLY (Roboflow API)"""
    detections_list = []
    
    img_h, img_w = image_data.shape[:2]
    
    # Only run thermal detection (Roboflow)
    if PERF_CONFIG['enable_thermal_detection']:
        try:
            logger.info("🌡️ Running thermal gun detection...")
            thermal_detections, annotated_frame = await detect_thermal_guns(image_data)
            
            # Save annotated image
            annotated_fname = None
            if thermal_detections and len(thermal_detections) > 0 and annotated_frame is not None:
                annotated_fname = f"thermal_{uuid.uuid4()}.jpg"
                annotated_path = DETECTIONS_DIR / annotated_fname
                cv2.imwrite(str(annotated_path), annotated_frame)
                logger.info(f"💾 Saved thermal annotated image: {annotated_fname}")
            
            for thermal_det in thermal_detections:
                thermal_det['camera_id'] = camera_id
                thermal_det['camera_name'] = camera_name
                thermal_det['location'] = location
                
                if annotated_fname:
                    thermal_det['image_path'] = f"/detections/{annotated_fname}"
                
                block_hash = await add_to_blockchain(thermal_det)
                thermal_det['blockchain_hash'] = block_hash
                await db.detections.insert_one(thermal_det)
                await send_twilio_alert(thermal_det)
                thermal_det['alert_sent'] = True
                await db.detections.update_one({"id": thermal_det['id']}, {"$set": {"alert_sent": True}})
                detections_list.append(thermal_det)
        except Exception as e:
            logger.error(f"❌ Error during thermal detection: {e}", exc_info=True)
    else:
        logger.warning("⚠️ Thermal detection is disabled in performance config")
    
    return detections_list

# App and routes
app = FastAPI()

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

@api_router.post("/upload")
async def upload_with_mode(
    file: UploadFile = File(...),
    camera_id: str = Form("upload"),
    camera_name: str = Form("Manual Upload"),
    latitude: str = Form("0"),
    longitude: str = Form("0"),
    detection_mode: str = Form("normal")  # 'normal' or 'thermal'
):
    """
    Upload endpoint that routes to different detection methods based on mode:
    - normal: Uses both best.pt and best (1).pt models
    - thermal: Uses Roboflow thermal detection API only
    """
    lat = float(latitude)
    lng = float(longitude)
    
    file_path = UPLOAD_DIR / file.filename
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    image = cv2.imread(str(file_path))
    
    if detection_mode == "thermal":
        # Thermal detection only - uses Roboflow
        logger.info("🌡️ Using thermal detection mode")
        detections = await process_thermal_only(image, camera_id, camera_name, {"lat": lat, "lng": lng})
    else:
        # Normal, Model1, Model2, or People detection
        logger.info(f"🔍 Using detection mode: {detection_mode}")
        detections = await process_detection(image, camera_id, camera_name, {"lat": lat, "lng": lng}, detection_mode=detection_mode)
    
    # Remove _id field for JSON serialization
    for det in detections:
        det.pop('_id', None)
    
    # Get annotated image path
    annotated_image = None
    for det in detections:
        if det.get('image_path'):
            annotated_image = det.get('image_path')
            break
    
    return {"detections": detections, "annotated_image": annotated_image}


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
async def detect_thermal_guns_endpoint(file: UploadFile = File(...)):
    """Detect thermal guns in uploaded image using Roboflow API"""
    try:
        logger.info(f"🌡️ THERMAL-GUN-DETECTION endpoint called with file: {file.filename}")
        contents = await file.read()
        logger.info(f"📥 Received file size: {len(contents)} bytes")
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("❌ Failed to decode image")
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        logger.info(f"✅ Image decoded successfully: {img.shape}")
        thermal_guns_count, detections, annotated_img = detect_thermal_guns(img)

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
                dets = await process_detection(frame, camera_id or "upload", camera_name, {"lat": lat, "lng": lng})
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
        detections = await process_detection(image, camera_id or "upload", camera_name, {"lat": lat, "lng": lng})
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
    camera_id: str
    camera_name: str
    detection_type: str
    confidence: float
    image_path: str
    location: dict
    blockchain_hash: str
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

@api_router.get("/blockchain/verify/{detection_id}")
async def verify_blockchain(detection_id: str):
    block = await db.blockchain.find_one({"detection_id": detection_id}, {"_id": 0})
    if not block:
        raise HTTPException(status_code=404, detail="Block not found")
    calc = await create_blockchain_hash({
        "detection_id": block['detection_id'],
        "previous_hash": block['previous_hash'],
        "data": block['data'],
        "timestamp": block['timestamp']
    })
    return {"verified": calc == block['hash'], "stored_hash": block['hash'], "calculated_hash": calc}

@api_router.get("/blockchain")
async def get_blockchain():
    blocks = await db.blockchain.find({}, {"_id": 0}).sort("timestamp", -1).to_list(1000)
    return blocks

@api_router.get("/police-stations")
async def get_police_stations():
    stations = [
        {"name": "Pune City Police Headquarters", "lat": 18.5204, "lng": 73.8567},
        {"name": "Shivajinagar Police Station", "lat": 18.5309, "lng": 73.8433},
        {"name": "Deccan Police Station", "lat": 18.5165, "lng": 73.8460}
    ]
    return stations


@api_router.get("/emergency-context")
async def emergency_context(lat: float, lng: float):
    """Return nearest police/hospital/fire station info using Overpass API"""
    
    # Initialize variables for caching
    current_time = time.time()
    lat_rounded = round(lat, 3)  # Round to ~100m precision for caching
    lng_rounded = round(lng, 3)
    
    def calculate_distance(lat1, lng1, lat2, lng2):
        """Calculate distance in km using Haversine formula"""
        R = 6371  # Earth's radius in km
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    def query_overpass(amenity_type, radius=5000):
        """Query Overpass API for specific amenity type"""
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="{amenity_type}"](around:{radius},{lat},{lng});
          way["amenity"="{amenity_type}"](around:{radius},{lat},{lng});
        );
        out center;
        """
        
        try:
            response = requests.post(overpass_url, data={'data': query}, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Overpass API error for {amenity_type}: {e}")
            return None
    
    def parse_overpass_results(data, amenity_type):
        """Parse Overpass API results and calculate distances"""
        if not data or 'elements' not in data:
            return []
        
        results = []
        for element in data['elements']:
            # Get coordinates
            if element['type'] == 'node':
                elem_lat = element['lat']
                elem_lng = element['lon']
            elif 'center' in element:
                elem_lat = element['center']['lat']
                elem_lng = element['center']['lon']
            else:
                continue
            
            # Get name
            name = element.get('tags', {}).get('name', f'Unnamed {amenity_type.title()}')
            phone = element.get('tags', {}).get('phone', 'N/A')
            
            # Calculate distance
            distance = calculate_distance(lat, lng, elem_lat, elem_lng)
            
            results.append({
                'name': name,
                'lat': elem_lat,
                'lng': elem_lng,
                'phone': phone,
                'distance_km': round(distance, 2),
                'distance_m': round(distance * 1000, 0)
            })
        
        # Sort by distance and return
        return sorted(results, key=lambda x: x['distance_km'])
    
    def get_cached_or_query(amenity_type):
        cache_key = (lat_rounded, lng_rounded, amenity_type)
        
        # Check cache
        if cache_key in overpass_cache:
            data, timestamp = overpass_cache[cache_key]
            if current_time - timestamp < CACHE_DURATION:
                return data
        
        # Query API if not cached or expired
        try:
            # specific radius for different services
            radius = 3000 if amenity_type == 'hospital' else 2000
            data = query_overpass(amenity_type, radius)
            if data:
                overpass_cache[cache_key] = (data, current_time)
            return data
        except Exception as e:
            logger.error(f"Overpass query failed for {amenity_type}: {e}")
            return None

    # Query with caching
    police_data = get_cached_or_query('police')
    hospital_data = get_cached_or_query('hospital')
    fire_data = get_cached_or_query('fire_station')
    
    # Parse results
    police_stations = parse_overpass_results(police_data, 'police')[:3]
    hospitals = parse_overpass_results(hospital_data, 'hospital')[:3]
    fire_stations = parse_overpass_results(fire_data, 'fire_station')[:2]
    
    # Fallback to hardcoded data if API fails
    if not police_stations:
        police_stations = [
            {"name": "Pune City Police Headquarters", "lat": 18.5204, "lng": 73.8567, "phone": "020-26126296", "distance_km": 0.5, "distance_m": 500},
            {"name": "Shivajinagar Police Station", "lat": 18.5309, "lng": 73.8433, "phone": "020-25532941", "distance_km": 1.2, "distance_m": 1200},
        ]
    
    if not hospitals:
        hospitals = [
            {"name": "Sassoon General Hospital", "lat": 18.5196, "lng": 73.8553, "phone": "020-26053251", "distance_km": 0.3, "distance_m": 300},
            {"name": "Ruby Hall Clinic", "lat": 18.5314, "lng": 73.8446, "phone": "020-66455000", "distance_km": 0.8, "distance_m": 800},
        ]
    
    if not fire_stations:
        fire_stations = [
            {"name": "Pune Fire Brigade HQ", "lat": 18.5204, "lng": 73.8567, "phone": "101", "distance_km": 0.5, "distance_m": 500},
        ]
    
    return {
        "police_stations": police_stations,
        "hospitals": hospitals,
        "fire_stations": fire_stations,
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
        
        # Perform detection on frame
        frame_with_detections, detections_to_save = detect_weapons_and_people(frame)
        
        # Update camera stats
        guns = sum(1 for d in detections_to_save if d.get('detection_type') == 'pistol')
        knives = sum(1 for d in detections_to_save if d.get('detection_type') == 'knife')
        # Note: thermal guns and people counting would need to be integrated here
        camera.update_stats(guns=guns, knives=knives)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame_with_detections, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS


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
