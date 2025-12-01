# Performance Optimization Guide

## Problem
Your laptop is heating up because the weapon detection system is running multiple AI models simultaneously:
- **YOLOv8 weapon detection** (guns/knives)
- **YOLOv8m people counting** (52MB model)
- **2x Roboflow thermal gun APIs** (external API calls)

All running at **30 FPS** on every frame = very high CPU/GPU usage.

## Quick Fixes (Immediate Relief)

### 1. Reduce Frame Rate
Lower FPS from 30 to 10-15 FPS:

**File: `backend/server.py`**
- Line 610: Change `time.sleep(0.03)` to `time.sleep(0.1)` (10 FPS)
- Line 210: Change `time.sleep(0.03)` to `time.sleep(0.1)` (10 FPS)

### 2. Disable People Counting (Saves ~40% CPU)
People counting runs a separate YOLOv8m model on every frame.

**File: `backend/server.py`**
- Line 419-437: Comment out the entire people detection block in `detect_weapons_and_people()`

### 3. Disable Thermal Gun Detection (Saves ~30% CPU)
Thermal detection makes 2 external API calls per frame.

**File: `backend/server.py`**
- Line 672-707: Comment out thermal gun detection in `process_detection()`
- Or set a flag to disable it for live streaming

### 4. Lower Resolution
Reduce camera resolution from 1280x720 to 640x480:

**File: `backend/server.py`**
- Line 129: Change to `self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`
- Line 130: Change to `self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)`

### 5. Skip Frames
Process every 2nd or 3rd frame instead of every frame:

Add frame skipping logic in `generate_frames()` function.

## Recommended Configuration

### Low Power Mode (Recommended for Laptops)
```python
FPS = 10  # Instead of 30
RESOLUTION = "640x480"  # Instead of 1280x720
ENABLE_PEOPLE_COUNTING = False
ENABLE_THERMAL_DETECTION = False
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
```

### Balanced Mode
```python
FPS = 15
RESOLUTION = "1280x720"
ENABLE_PEOPLE_COUNTING = True
ENABLE_THERMAL_DETECTION = False
PROCESS_EVERY_N_FRAMES = 1
```

### High Performance Mode (Desktop/GPU)
```python
FPS = 30
RESOLUTION = "1280x720"
ENABLE_PEOPLE_COUNTING = True
ENABLE_THERMAL_DETECTION = True
PROCESS_EVERY_N_FRAMES = 1
```

## Implementation Steps

I can create an optimized version with:
1. **Configuration file** to easily switch between modes
2. **Frame skipping** to reduce processing load
3. **Conditional model loading** (disable unused models)
4. **Lower default FPS and resolution**

## Expected Results

| Optimization | CPU/GPU Reduction |
|--------------|-------------------|
| Reduce FPS (30→10) | ~60% |
| Disable people counting | ~40% |
| Disable thermal detection | ~30% |
| Lower resolution | ~25% |
| Skip frames (process every 2nd) | ~50% |

**Combined**: Applying all optimizations can reduce CPU/GPU usage by **70-80%**.

## Would you like me to:
1. ✅ Create an optimized configuration system
2. ✅ Implement frame skipping
3. ✅ Add performance modes (Low/Balanced/High)
4. ✅ Set default to "Low Power Mode"
