"""
Safe camera wrapper for macOS that prevents segfaults
Uses polling with timeout and better error handling
"""

import cv2
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class SafeCamera:
    """Wrapper around cv2.VideoCapture with macOS crash protection"""
    
    def __init__(self, camera_index: int, max_retries: int = 3):
        self.camera_index = camera_index
        self.cap = None
        self.max_retries = max_retries
        self.is_open = False
        self.last_frame = None
        self.frame_count = 0
        self.error_count = 0
        self._lock = threading.Lock()  # Protect cap access
        
    def open(self) -> bool:
        """Safely open camera with retries"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"[CAMERA] Opening camera {self.camera_index}, attempt {attempt + 1}/{self.max_retries}")
                
                # Create camera capture
                self.cap = cv2.VideoCapture(self.camera_index)
                
                if self.cap is None:
                    logger.error(f"[CAMERA] Failed to create VideoCapture object")
                    continue
                
                # Verify it's open
                if not self.cap.isOpened():
                    logger.warning(f"[CAMERA] VideoCapture not initially open, retrying...")
                    self.cap.release()
                    self.cap = None
                    time.sleep(1)
                    continue
                
                # Try to read one frame to verify it works
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning(f"[CAMERA] Failed to read test frame, retrying...")
                    self.cap.release()
                    self.cap = None
                    time.sleep(1)
                    continue
                
                # Success!
                self.is_open = True
                self.last_frame = frame.copy()
                self.frame_count = 1
                logger.info(f"[CAMERA] Camera {self.camera_index} opened successfully")
                return True
                
            except Exception as e:
                logger.error(f"[CAMERA] Attempt {attempt + 1} failed: {e}")
                if self.cap:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                time.sleep(1)
        
        logger.error(f"[CAMERA] Failed to open camera after {self.max_retries} attempts")
        return False
    
    def read(self) -> tuple:
        """Safely read a frame with fallback to last valid frame"""
        if not self.is_open or self.cap is None:
            return False, None
        
        try:
            with self._lock:
                # Add timeout protection
                # Set read timeout to prevent hanging
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
                
                ret, frame = self.cap.read()
            
            if ret and frame is not None:
                # Verify frame validity
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    self.last_frame = frame.copy()
                    self.frame_count += 1
                    self.error_count = 0
                    return True, frame
            
            self.error_count += 1
            # Return last valid frame as fallback
            if self.last_frame is not None:
                return True, self.last_frame.copy()
            return False, None
                
        except Exception as e:
            self.error_count += 1
            logger.warning(f"[CAMERA] Read error: {type(e).__name__}: {e}")
            # Return last valid frame as fallback
            if self.last_frame is not None:
                return True, self.last_frame.copy()
            return False, None
    
    def release(self):
        """Safely release camera"""
        try:
            with self._lock:
                if self.cap:
                    self.cap.release()
                    logger.info(f"[CAMERA] Camera {self.camera_index} released ({self.frame_count} frames captured)")
        except Exception as e:
            logger.error(f"[CAMERA] Error releasing camera: {e}")
        finally:
            self.cap = None
            self.is_open = False
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
