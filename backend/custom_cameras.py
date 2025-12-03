import logging
from pathlib import Path
from camera_manager import camera_registry, CameraType
import random

logger = logging.getLogger(__name__)

def register_custom_cameras(root_dir: Path):
    """Register custom cameras with specific configurations"""
    
    # === TOP VIEW CAMERA ===
    topview_video_path = root_dir / "videos" / "topView.mp4"
    if topview_video_path.exists():
        try:
            if not camera_registry.get_camera("topview_cam"):
                camera_registry.register_camera(
                    name="Top View",
                    camera_type=CameraType.FILE,
                    source=str(topview_video_path),
                    location={"lat": 18.5204, "lng": 73.8567},
                    settings={
                        "resolution": "1280x720", 
                        "fps": 30, 
                        "loop": True,
                        "detection_mode": "topview"  # Use custom topview detection
                    },
                    camera_id="topview_cam"
                )
                logger.info(f"✅ Registered 'Top View' camera")
                camera_registry.start_camera("topview_cam")
        except Exception as e:
            logger.error(f"Failed to register 'Top View' camera: {e}")

    # === HOUSE CAMERA ===
    house_video_path = root_dir / "videos" / "House.mp4"
    if house_video_path.exists():
        try:
            if not camera_registry.get_camera("house_cam"):
                camera_registry.register_camera(
                    name="House",
                    camera_type=CameraType.FILE,
                    source=str(house_video_path),
                    location={"lat": 18.5204, "lng": 73.8567},
                    settings={
                        "resolution": "1280x720", 
                        "fps": 30, 
                        "loop": True,
                        "detection_mode": "house_medicine"  # Person + Gun detection
                    },
                    camera_id="house_cam"
                )
                logger.info(f"✅ Registered 'House' camera")
                camera_registry.start_camera("house_cam")
        except Exception as e:
            logger.error(f"Failed to register 'House' camera: {e}")

    # === MEDICINE CAMERA ===
    medicine_video_path = root_dir / "videos" / "Medicine.mp4"
    if medicine_video_path.exists():
        try:
            if not camera_registry.get_camera("medicine_cam"):
                camera_registry.register_camera(
                    name="Medicine",
                    camera_type=CameraType.FILE,
                    source=str(medicine_video_path),
                    location={"lat": 18.5204, "lng": 73.8567},
                    settings={
                        "resolution": "1280x720", 
                        "fps": 30, 
                        "loop": True,
                        "detection_mode": "medicine_only"  # Firearm detection only, no person
                    },
                    camera_id="medicine_cam"
                )
                logger.info(f"✅ Registered 'Medicine' camera")
                camera_registry.start_camera("medicine_cam")
        except Exception as e:
            logger.error(f"Failed to register 'Medicine' camera: {e}")

    # === GUN 1 CAMERA ===
    gun1_video_path = root_dir / "videos" / "Gun1.mp4"
    if gun1_video_path.exists():
        try:
            if not camera_registry.get_camera("gun_cam_1"):
                lat = 18.5204 + random.uniform(-0.005, 0.005)
                lng = 73.8567 + random.uniform(-0.005, 0.005)
                
                camera_registry.register_camera(
                    name="Gun 1",
                    camera_type=CameraType.FILE,
                    source=str(gun1_video_path),
                    location={"lat": lat, "lng": lng},
                    settings={
                        "resolution": "1280x720", 
                        "fps": 30, 
                        "loop": True,
                        "detection_mode": "gun_only"  # Gun detection only
                    },
                    camera_id="gun_cam_1"
                )
                logger.info(f"✅ Registered 'Gun 1' camera")
                camera_registry.start_camera("gun_cam_1")
        except Exception as e:
            logger.error(f"Failed to register 'Gun 1' camera: {e}")

    # === GUN CAMERA ===
    gun_video_path = root_dir / "videos" / "Gun.mp4"
    if gun_video_path.exists():
        try:
            if not camera_registry.get_camera("gun_cam"):
                camera_registry.register_camera(
                    name="Gun",
                    camera_type=CameraType.FILE,
                    source=str(gun_video_path),
                    location={"lat": 18.5204, "lng": 73.8567},
                    settings={
                        "resolution": "1280x720", 
                        "fps": 30, 
                        "loop": True,
                        "detection_mode": "gun_only"  # Gun detection only
                    },
                    camera_id="gun_cam"
                )
                logger.info(f"✅ Registered 'Gun' camera")
                camera_registry.start_camera("gun_cam")
        except Exception as e:
            logger.error(f"Failed to register 'Gun' camera: {e}")

    # === BEST (1) CAMERA ===
    best1_video_path = root_dir / "videos" / "best (1).mp4"
    if best1_video_path.exists():
        try:
            if not camera_registry.get_camera("best1_cam"):
                camera_registry.register_camera(
                    name="Best (1) Cam",
                    camera_type=CameraType.FILE,
                    source=str(best1_video_path),
                    location={"lat": 18.5204, "lng": 73.8567},
                    settings={
                        "resolution": "1280x720", 
                        "fps": 30, 
                        "loop": True,
                        "detection_mode": "gun_only"  # Gun detection only
                    },
                    camera_id="best1_cam"
                )
                logger.info(f"✅ Registered 'Best (1)' camera")
                camera_registry.start_camera("best1_cam")
        except Exception as e:
            logger.error(f"Failed to register 'Best (1)' camera: {e}")

