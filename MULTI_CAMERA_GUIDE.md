# Multi-Camera Support - Quick Start Guide

## Overview

SecureView Alert now supports multiple camera sources! Monitor multiple locations simultaneously with independent detection and statistics.

## Supported Camera Types

- **Webcam** - Built-in or USB cameras (index: 0, 1, 2, etc.)
- **RTSP Stream** - Network cameras with RTSP protocol
- **IP Camera** - HTTP-based IP cameras
- **Video File** - Pre-recorded videos for testing

## Quick Start

### 1. Start the Backend

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Frontend

```bash
cd frontend
npm start
```

### 3. Access Camera Management

Navigate to: **http://localhost:3000/cameras**

## Adding Your First Camera

1. Click **"Add Camera"** button
2. Fill in the details:
   - **Name**: "Front Entrance" (or any descriptive name)
   - **Type**: Select "Webcam"
   - **Source**: Enter `0` (for first webcam)
   - **Location**: Enter your latitude and longitude
3. Click **"Add Camera"**
4. Click **"Start"** to begin monitoring

## Managing Multiple Cameras

### Add Multiple Cameras

Repeat the process above for each camera source:

```
Camera 1: Webcam (source: 0) - Front Entrance
Camera 2: Webcam (source: 1) - Back Door
Camera 3: RTSP (source: rtsp://...) - Parking Lot
Camera 4: File (source: /path/to/test.mp4) - Test Feed
```

### Monitor Cameras

- **Grid View**: See all cameras in responsive grid
- **Live Previews**: Thumbnail feeds for active cameras
- **Full Screen**: Click maximize icon for full view
- **Statistics**: View detections per camera

### Control Cameras

- **Start**: Green play button
- **Stop**: Red stop button
- **Delete**: Trash icon (with confirmation)

## API Endpoints

### Camera Management

```bash
# List all cameras
GET /api/cameras

# Register new camera
POST /api/cameras
{
  "name": "Front Entrance",
  "type": "webcam",
  "source": "0",
  "location": {"lat": 18.5204, "lng": 73.8567},
  "settings": {"resolution": "1280x720", "fps": 30}
}

# Start camera
POST /api/cameras/{camera_id}/start

# Stop camera
POST /api/cameras/{camera_id}/stop

# Get camera status
GET /api/cameras/{camera_id}/status

# Get video feed
GET /api/cameras/{camera_id}/video_feed

# Delete camera
DELETE /api/cameras/{camera_id}
```

## Camera Sources

### Webcam
```
Type: webcam
Source: 0, 1, 2, ... (camera index)
```

### RTSP Stream
```
Type: rtsp
Source: rtsp://username:password@192.168.1.100:554/stream1
```

### IP Camera
```
Type: ip
Source: http://192.168.1.100:8080/video
```

### Video File
```
Type: file
Source: /Users/username/videos/test.mp4
```

## Dashboard Integration

The Dashboard now shows:
- **Total Cameras**: Count of all registered cameras
- **Active Cameras**: Currently running cameras
- **Camera Names**: Source camera in recent detections

## Features

✅ **Independent Monitoring** - Each camera runs in its own thread
✅ **Individual Statistics** - Separate detection counts per camera
✅ **Live Previews** - Real-time thumbnail feeds
✅ **Status Indicators** - Color-coded camera states
✅ **Error Handling** - Automatic reconnection on failures
✅ **Backward Compatible** - Existing Live Monitor unchanged

## Performance

- **Recommended**: 2-4 concurrent cameras
- **Maximum**: 6-8 cameras (depending on hardware)
- **Resolution**: 1280x720 recommended
- **FPS**: 30 FPS per camera

## Troubleshooting

### Camera Won't Start

**Error**: "Failed to open camera source"

**Solutions**:
- Check camera index (try 0, 1, 2)
- Verify camera is not in use by another app
- Check camera permissions (macOS: System Preferences > Security & Privacy > Camera)
- For RTSP: Verify URL and credentials

### Video Feed Not Showing

**Solutions**:
- Ensure camera status is "active" (green badge)
- Refresh the page
- Check browser console for errors
- Verify backend is running on port 8000

### Low Frame Rate

**Solutions**:
- Reduce number of active cameras
- Lower resolution in camera settings
- Reduce FPS to 15-20
- Check CPU usage

## Example: Multi-Camera Setup

```python
# Using Python API client
import requests

API = "http://localhost:8000/api"

# Add front entrance camera
requests.post(f"{API}/cameras", json={
    "name": "Front Entrance",
    "type": "webcam",
    "source": "0",
    "location": {"lat": 18.5204, "lng": 73.8567}
})

# Add back door camera
requests.post(f"{API}/cameras", json={
    "name": "Back Door",
    "type": "webcam",
    "source": "1",
    "location": {"lat": 18.5210, "lng": 73.8570}
})

# Start both cameras
cameras = requests.get(f"{API}/cameras").json()["cameras"]
for cam in cameras:
    requests.post(f"{API}/cameras/{cam['id']}/start")
```

## Next Steps

1. ✅ Add your cameras via the Cameras page
2. ✅ Start monitoring multiple locations
3. ✅ View aggregate statistics on Dashboard
4. ✅ Check detections with camera source labels
5. 🔄 (Optional) Integrate with Map View for location tracking

## Support

For issues or questions:
- Check the [walkthrough.md](file:///Users/ritesh/.gemini/antigravity/brain/58abc7cb-b52d-44de-a8f3-79dbf920d7a2/walkthrough.md) for detailed documentation
- Review backend logs for errors
- Ensure MongoDB is running
- Verify all dependencies are installed

---

**Status**: ✅ Multi-Camera Support Fully Implemented
**Version**: 2.0
**Last Updated**: November 25, 2025
