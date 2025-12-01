# Backend - FastAPI Detection Server

Real-time weapon and people detection system using FastAPI and YOLOv8.

## Quick Start

### 1. Install Dependencies

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup Environment

Create `.env` file in `backend/` directory:

```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=detection_db
UPLOAD_DIR=./uploads
DETECTIONS_DIR=./detections
PORT=8000
CORS_ORIGINS=*
```

### 3. Run Server

```bash
source venv/bin/activate
uvicorn server:app --host 127.0.0.1 --port 8000
```

For development with auto-reload:
```bash
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

## API Endpoints

### Camera Control
- `POST /start_camera` - Start camera stream
- `POST /stop_camera` - Stop specific camera
- `POST /stop_all_cameras` - Stop all cameras
- `GET /video_feed` - MJPEG stream

### Detection
- `GET /api/get_detections` - Get current detections
- `GET /api/detections` - Get detection history
- `GET /api/cameras` - List all cameras

### Upload
- `POST /upload` - Upload and process images

### Blockchain
- `GET /api/blockchain` - Get blockchain records
- `POST /verify` - Verify detection hash

## Models

- `best.pt` - Weapon detection (pistol, knife)
- `yolov8m.pt` - People detection (80 classes)

## Key Features

- **Fallback Mode**: Prevents crashes on macOS by using test images
- **Thread-Safe**: SafeCamera wrapper with locking
- **MJPEG Streaming**: Real-time video with detection overlays
- **MongoDB**: Persistent storage for detections
- **CORS Enabled**: Cross-origin requests from React

## Troubleshooting

### Port already in use
```bash
lsof -i :8000
kill -9 <PID>
```

### MongoDB connection error
```bash
brew services start mongodb-community
```

### Camera issues on macOS
System automatically uses fallback mode - app remains stable
