"""
Multi-Camera Management System
Handles registration, lifecycle, and streaming for multiple camera sources
"""

import cv2
import threading
import time
import logging
from typing import Dict, Optional, List
from datetime import datetime, timezone
from enum import Enum
import uuid

# Import performance configuration
try:
    from performance_config import get_config, get_resolution_tuple
    PERF_CONFIG = get_config()
    USE_PERF_CONFIG = True
except ImportError:
    USE_PERF_CONFIG = False
    PERF_CONFIG = {
        'resolution': '1280x720',
        'fps': 30
    }

logger = logging.getLogger(__name__)


class CameraType(str, Enum):
    """Camera source types"""
    WEBCAM = "webcam"
    RTSP = "rtsp"
    IP = "ip"
    FILE = "file"


class CameraStatus(str, Enum):
    """Camera operational status"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"
    STARTING = "starting"
    STOPPING = "stopping"


class CameraInstance:
    """Represents a single camera instance with its own capture and processing"""
    
    def __init__(self, camera_id: str, name: str, camera_type: CameraType, 
                 source: str, location: dict, settings: dict = None):
        self.camera_id = camera_id
        self.name = name
        self.camera_type = camera_type
        self.source = source
        self.location = location
        self.settings = settings or {}
        
        self.status = CameraStatus.INACTIVE
        self.capture = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = False
        self.capture_thread = None
        self.error_message = None
        
        # Statistics
        self.stats = {
            'guns': 0,
            'knives': 0,
            'thermal_guns': 0,
            'people': 0,
            'frames_processed': 0,
            'last_detection_time': None
        }
        self.stats_lock = threading.Lock()
        
        # Metadata
        self.created_at = datetime.now(timezone.utc)
        self.last_active = None
    
    def _get_capture_source(self) -> any:
        """Convert source string to OpenCV capture source"""
        if self.camera_type == CameraType.WEBCAM:
            try:
                return int(self.source)
            except ValueError:
                logger.error(f"Invalid webcam index: {self.source}")
                return 0
        elif self.camera_type in [CameraType.RTSP, CameraType.IP, CameraType.FILE]:
            return self.source
        else:
            logger.warning(f"Unknown camera type: {self.camera_type}, defaulting to source string")
            return self.source
    
    def start(self) -> bool:
        """Start camera capture"""
        if self.is_running:
            logger.warning(f"Camera {self.camera_id} is already running")
            return False
        
        try:
            self.status = CameraStatus.STARTING
            logger.info(f"Starting camera {self.camera_id} ({self.name}) - Type: {self.camera_type}, Source: {self.source}")
            
            # Initialize video capture
            capture_source = self._get_capture_source()
            self.capture = cv2.VideoCapture(capture_source)
            
            if not self.capture.isOpened():
                raise Exception(f"Failed to open camera source: {self.source}")
            
            # Apply settings (use performance config defaults if not specified)
            resolution = self.settings.get('resolution', PERF_CONFIG['resolution'])
            if USE_PERF_CONFIG:
                width, height = get_resolution_tuple(resolution)
            else:
                width, height = map(int, resolution.split('x'))
            fps = self.settings.get('fps', PERF_CONFIG['fps'])
            
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_FPS, fps)
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.status = CameraStatus.ACTIVE
            self.last_active = datetime.now(timezone.utc)
            self.error_message = None
            
            logger.info(f"✅ Camera {self.camera_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start camera {self.camera_id}: {e}", exc_info=True)
            self.status = CameraStatus.ERROR
            self.error_message = str(e)
            self.is_running = False
            if self.capture:
                self.capture.release()
                self.capture = None
            return False
    
    def stop(self) -> bool:
        """Stop camera capture"""
        if not self.is_running:
            logger.warning(f"Camera {self.camera_id} is not running")
            return False
        
        try:
            self.status = CameraStatus.STOPPING
            logger.info(f"Stopping camera {self.camera_id} ({self.name})")
            
            self.is_running = False
            
            # Wait for capture thread to finish
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            
            # Release capture
            if self.capture:
                self.capture.release()
                self.capture = None
            
            with self.frame_lock:
                self.current_frame = None
            
            self.status = CameraStatus.INACTIVE
            logger.info(f"⏹️ Camera {self.camera_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping camera {self.camera_id}: {e}", exc_info=True)
            self.status = CameraStatus.ERROR
            self.error_message = str(e)
            return False
    
    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        logger.info(f"Capture loop started for camera {self.camera_id}")
        consecutive_failures = 0
        max_failures = 10
        
        while self.is_running:
            try:
                if not self.capture or not self.capture.isOpened():
                    logger.error(f"Camera {self.camera_id} capture is not open")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        self.status = CameraStatus.ERROR
                        self.error_message = "Camera disconnected or unavailable"
                        break
                    time.sleep(1)
                    continue
                
                success, frame = self.capture.read()
                
                if not success or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame from camera {self.camera_id} (attempt {consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        self.status = CameraStatus.ERROR
                        self.error_message = "Failed to read frames from camera"
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Successfully read frame
                consecutive_failures = 0
                
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                with self.stats_lock:
                    self.stats['frames_processed'] += 1
                
                self.last_active = datetime.now(timezone.utc)
                
                # Control frame rate
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in capture loop for camera {self.camera_id}: {e}", exc_info=True)
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self.status = CameraStatus.ERROR
                    self.error_message = f"Capture loop error: {str(e)}"
                    break
                time.sleep(0.5)
        
        logger.info(f"Capture loop ended for camera {self.camera_id}")
        self.is_running = False
    
    def get_frame(self) -> Optional[any]:
        """Get the current frame (thread-safe)"""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def update_stats(self, guns: int = 0, knives: int = 0, thermal_guns: int = 0, people: int = 0):
        """Update detection statistics"""
        with self.stats_lock:
            self.stats['guns'] = guns
            self.stats['knives'] = knives
            self.stats['thermal_guns'] = thermal_guns
            self.stats['people'] = people
            if guns > 0 or knives > 0 or thermal_guns > 0:
                self.stats['last_detection_time'] = datetime.now(timezone.utc).isoformat()
    
    def get_stats(self) -> dict:
        """Get current statistics (thread-safe)"""
        with self.stats_lock:
            return self.stats.copy()
    
    def to_dict(self) -> dict:
        """Convert camera instance to dictionary"""
        return {
            'id': self.camera_id,
            'name': self.name,
            'type': self.camera_type,
            'source': self.source,
            'location': self.location,
            'status': self.status,
            'settings': self.settings,
            'stats': self.get_stats(),
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat() if self.last_active else None
        }


class CameraRegistry:
    """Manages multiple camera instances"""
    
    def __init__(self):
        self.cameras: Dict[str, CameraInstance] = {}
        self.registry_lock = threading.Lock()
        logger.info("Camera Registry initialized")
    
    def register_camera(self, name: str, camera_type: CameraType, source: str, 
                       location: dict, settings: dict = None, camera_id: str = None) -> str:
        """Register a new camera"""
        with self.registry_lock:
            # Generate unique ID if not provided
            if camera_id is None:
                camera_id = f"cam_{uuid.uuid4().hex[:8]}"
            
            # Check if camera already exists
            if camera_id in self.cameras:
                raise ValueError(f"Camera with ID {camera_id} already exists")
            
            # Create camera instance
            camera = CameraInstance(
                camera_id=camera_id,
                name=name,
                camera_type=camera_type,
                source=source,
                location=location,
                settings=settings
            )
            
            self.cameras[camera_id] = camera
            logger.info(f"📹 Registered camera: {camera_id} - {name} ({camera_type})")
            
            return camera_id
    
    def unregister_camera(self, camera_id: str) -> bool:
        """Unregister and remove a camera"""
        with self.registry_lock:
            if camera_id not in self.cameras:
                logger.warning(f"Camera {camera_id} not found in registry")
                return False
            
            camera = self.cameras[camera_id]
            
            # Stop camera if running
            if camera.is_running:
                camera.stop()
            
            # Remove from registry
            del self.cameras[camera_id]
            logger.info(f"🗑️ Unregistered camera: {camera_id}")
            
            return True
    
    def get_camera(self, camera_id: str) -> Optional[CameraInstance]:
        """Get camera instance by ID"""
        return self.cameras.get(camera_id)
    
    def list_cameras(self) -> List[dict]:
        """List all registered cameras"""
        with self.registry_lock:
            return [camera.to_dict() for camera in self.cameras.values()]
    
    def start_camera(self, camera_id: str) -> bool:
        """Start a specific camera"""
        camera = self.get_camera(camera_id)
        if not camera:
            logger.error(f"Camera {camera_id} not found")
            return False
        return camera.start()
    
    def stop_camera(self, camera_id: str) -> bool:
        """Stop a specific camera"""
        camera = self.get_camera(camera_id)
        if not camera:
            logger.error(f"Camera {camera_id} not found")
            return False
        return camera.stop()
    
    def stop_all_cameras(self):
        """Stop all active cameras"""
        logger.info("Stopping all cameras...")
        with self.registry_lock:
            for camera in self.cameras.values():
                if camera.is_running:
                    camera.stop()
        logger.info("All cameras stopped")
    
    def get_active_cameras(self) -> List[str]:
        """Get list of active camera IDs"""
        with self.registry_lock:
            return [cid for cid, cam in self.cameras.items() if cam.status == CameraStatus.ACTIVE]
    
    def update_camera_settings(self, camera_id: str, settings: dict) -> bool:
        """Update camera settings"""
        camera = self.get_camera(camera_id)
        if not camera:
            return False
        
        camera.settings.update(settings)
        logger.info(f"Updated settings for camera {camera_id}")
        return True


# Global camera registry instance
camera_registry = CameraRegistry()
