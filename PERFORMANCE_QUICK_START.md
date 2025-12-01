# Performance Optimization - Quick Start Guide

## 🔥 Problem Solved
Your laptop was heating up because the system was running:
- **YOLOv8 weapon detection** at 30 FPS
- **YOLOv8m people counting** (52MB model) at 30 FPS
- **2x Roboflow thermal gun APIs** on every frame

## ✅ Solution Implemented

A performance configuration system with three modes:

### 1. **Low Power Mode** (Default - Recommended for Laptops)
- **FPS**: 10 (instead of 30)
- **Resolution**: 640x480 (instead of 1280x720)
- **People Counting**: Disabled
- **Thermal Detection**: Disabled
- **Frame Processing**: Every 2nd frame
- **Expected CPU/GPU Reduction**: ~70-80%

### 2. **Balanced Mode**
- **FPS**: 15
- **Resolution**: 1280x720
- **People Counting**: Enabled
- **Thermal Detection**: Disabled
- **Frame Processing**: Every frame
- **Expected CPU/GPU Reduction**: ~40-50%

### 3. **High Performance Mode**
- **FPS**: 30
- **Resolution**: 1280x720
- **People Counting**: Enabled
- **Thermal Detection**: Enabled
- **Frame Processing**: Every frame
- **Use Case**: Desktop with good GPU

## 🚀 How to Change Performance Mode

**File**: `backend/performance_config.py`

Change line 7:
```python
PERFORMANCE_MODE = "low_power"  # Change to: "balanced" or "high_performance"
```

Then restart the server:
```bash
cd backend
python3 server.py
```

## 🎛️ Custom Configuration

You can also create custom settings by uncommenting and modifying lines in `performance_config.py`:

```python
# Custom settings (override profile)
FPS = 12  # Your custom FPS
RESOLUTION = "800x600"  # Your custom resolution
ENABLE_PEOPLE_COUNTING = False  # Enable/disable
ENABLE_THERMAL_DETECTION = False  # Enable/disable
PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame
WEAPON_CONFIDENCE = 0.35  # Detection confidence threshold
JPEG_QUALITY = 80  # Image quality (1-100)
```

## 📊 Expected Results

| Setting | CPU/GPU Impact |
|---------|----------------|
| Low Power Mode | **70-80% reduction** |
| Balanced Mode | **40-50% reduction** |
| High Performance | Original performance |

## 🔍 What Changed

### Files Modified:
1. **`backend/performance_config.py`** (NEW) - Configuration system
2. **`backend/server.py`** - Integrated performance settings
3. **`backend/camera_manager.py`** - Uses performance defaults

### Key Optimizations:
- ✅ Configurable FPS and resolution
- ✅ Conditional model loading (people counting disabled by default)
- ✅ Conditional thermal detection (disabled by default)
- ✅ Frame skipping (process every 2nd frame by default)
- ✅ Adjustable JPEG compression quality

## 🧪 Testing

Start the server and check the logs:
```bash
cd backend
python3 server.py
```

You should see:
```
🔧 Performance Mode: Minimal CPU/GPU usage - recommended for laptops
   FPS: 10, Resolution: 640x480
   People Counting: False, Thermal Detection: False
   Process Every N Frames: 2
ℹ️ People counting disabled (performance optimization)
```

## 💡 Tips

1. **Start with Low Power Mode** - Your laptop should stay cool
2. **Monitor CPU usage** - Use Activity Monitor (Mac) or Task Manager (Windows)
3. **Gradually increase** - If performance is good, try Balanced mode
4. **For production** - Use High Performance mode only on servers with good GPUs

## 🆘 Troubleshooting

**Still heating up?**
- Ensure `PERFORMANCE_MODE = "low_power"` in `performance_config.py`
- Restart the server completely
- Check that people counting is disabled in logs

**Need better detection accuracy?**
- Switch to "balanced" mode
- Or customize: `WEAPON_CONFIDENCE = 0.25` (lower = more sensitive)

**Want even lower CPU usage?**
- Set `PROCESS_EVERY_N_FRAMES = 3` (process every 3rd frame)
- Set `FPS = 5` (5 frames per second)
- Set `RESOLUTION = "320x240"` (very low resolution)
