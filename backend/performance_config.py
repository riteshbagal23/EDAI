"""
Performance Configuration for Weapon Detection System
Adjust these settings to control CPU/GPU usage and reduce laptop heating
"""

# ============================================================================
# PERFORMANCE MODE
# ============================================================================
# Choose one: "low_power", "balanced", "high_performance"
PERFORMANCE_MODE = "balanced"  # Changed to balanced for better performance

# ============================================================================
# PERFORMANCE PROFILES
# ============================================================================

PROFILES = {
    "low_power": {
        "fps": 10,
        "resolution": "640x480",
        "enable_people_counting": False,
        "enable_thermal_detection": False,
        "process_every_n_frames": 2,  # Process every 2nd frame
        "weapon_confidence": 0.60,  # Higher threshold = fewer false positives
        "jpeg_quality": 75,
        "description": "Minimal CPU/GPU usage - recommended for laptops"
    },
    "balanced": {
        "fps": 20,  # Increased from 15 to 20 for smoother video
        "resolution": "1280x720",
        "enable_people_counting": True,
        "enable_thermal_detection": False,
        "process_every_n_frames": 2,  # Process every 2nd frame to reduce lag
        "weapon_confidence": 0.35,  # Lowered for better gun detection
        "jpeg_quality": 85,  # Increased quality
        "description": "Balanced performance and accuracy"
    },
    "high_performance": {
        "fps": 30,
        "resolution": "1280x720",
        "enable_people_counting": True,
        "enable_thermal_detection": True,
        "process_every_n_frames": 1,
        "weapon_confidence": 0.50,  # Increased from 0.30 to reduce false positives
        "jpeg_quality": 85,
        "description": "Maximum accuracy - requires good GPU"
    }
}

# ============================================================================
# CUSTOM SETTINGS (Override profile settings)
# ============================================================================
# Uncomment and modify to override profile settings

# FPS = 10  # Frames per second
# RESOLUTION = "640x480"  # Camera resolution (WIDTHxHEIGHT)
# ENABLE_PEOPLE_COUNTING = False  # Enable/disable people counting model
# ENABLE_THERMAL_DETECTION = False  # Enable/disable thermal gun detection
# PROCESS_EVERY_N_FRAMES = 2  # Process every Nth frame (1 = all frames)
# WEAPON_CONFIDENCE = 0.40  # Confidence threshold for weapon detection
# JPEG_QUALITY = 75  # JPEG compression quality (1-100)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config():
    """Get current configuration based on performance mode"""
    if PERFORMANCE_MODE not in PROFILES:
        raise ValueError(f"Invalid PERFORMANCE_MODE: {PERFORMANCE_MODE}. Choose from: {list(PROFILES.keys())}")
    
    config = PROFILES[PERFORMANCE_MODE].copy()
    
    # Override with custom settings if defined
    if 'FPS' in globals():
        config['fps'] = FPS
    if 'RESOLUTION' in globals():
        config['resolution'] = RESOLUTION
    if 'ENABLE_PEOPLE_COUNTING' in globals():
        config['enable_people_counting'] = ENABLE_PEOPLE_COUNTING
    if 'ENABLE_THERMAL_DETECTION' in globals():
        config['enable_thermal_detection'] = ENABLE_THERMAL_DETECTION
    if 'PROCESS_EVERY_N_FRAMES' in globals():
        config['process_every_n_frames'] = PROCESS_EVERY_N_FRAMES
    if 'WEAPON_CONFIDENCE' in globals():
        config['weapon_confidence'] = WEAPON_CONFIDENCE
    if 'JPEG_QUALITY' in globals():
        config['jpeg_quality'] = JPEG_QUALITY
    
    return config

def get_resolution_tuple(resolution_str):
    """Convert resolution string to (width, height) tuple"""
    width, height = map(int, resolution_str.split('x'))
    return width, height

def get_frame_delay(fps):
    """Calculate delay between frames in seconds"""
    return 1.0 / fps

# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    config = get_config()
    print(f"Performance Mode: {PERFORMANCE_MODE}")
    print(f"Description: {config['description']}")
    print(f"Settings:")
    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")
