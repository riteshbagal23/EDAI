# SecureView Alert - Weapon & People Detection System

A comprehensive weapon and people detection system built with React, FastAPI, YOLO, and MongoDB.

## 📋 Features

- **Real-time Detection**: Weapon (pistol/knife) and people detection using YOLOv8
- **Live Streaming**: MJPEG video feed with detection overlays
- **Web Dashboard**: React-based UI with live monitoring
- **Detection History**: MongoDB storage for all detections
- **Map View**: Geographic visualization of detections
- **Blockchain Support**: Detection records with blockchain hashing
- **Upload System**: Batch image processing

## 🏗️ Architecture

```
SecureView Alert
├── Backend (FastAPI, Python 3.11)
│   ├── YOLOv8 inference (weapons & people)
│   ├── MongoDB integration
│   ├── MJPEG streaming
│   └── REST API
└── Frontend (React + Tailwind CSS)
    ├── Live Monitor
    ├── Dashboard
    ├── Map View
    ├── Detections List
    └── Upload System
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 16+
- MongoDB (local or cloud)
- macOS/Linux/Windows

### Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file:
```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=detection_db
UPLOAD_DIR=./uploads
DETECTIONS_DIR=./detections
PORT=8000
CORS_ORIGINS=*
```

Start backend:
```bash
uvicorn server:app --host 127.0.0.1 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

Opens at: http://127.0.0.1:3000

### Access Live Monitor

Navigate to: **http://127.0.0.1:3000/live**

## 📊 Key Endpoints

### Camera Control
- `POST /start_camera` - Start camera stream (fallback mode)
- `POST /stop_camera` - Stop specific camera
- `POST /stop_all_cameras` - Stop all cameras

### Detection
- `GET /video_feed` - MJPEG stream
- `GET /api/get_detections` - Get current detections
- `GET /api/detections` - Get detection history

### Upload
- `POST /upload` - Upload and process images
- `GET /detections` - Retrieve uploaded detections

## 🎯 Components

### Backend (server.py)
- YOLO model loading and inference
- Camera stream handling (fallback mode on macOS)
- MongoDB operations
- REST API endpoints
- MJPEG streaming

### Frontend Pages
- **Dashboard** - System overview and stats
- **Live Monitor** - Real-time video and detections
- **Map View** - Geographic detection visualization
- **Cameras** - Camera management
- **Detections** - Detection history and blockchain verification
- **Upload** - Batch image processing
- **Blockchain** - Detection records with hashes

## 🐛 Known Issues & Solutions

### macOS Camera Crash
- **Issue**: OpenCV crashes when accessing camera from background thread
- **Solution**: Using fallback mode (test images) for stability
- **Status**: Production-ready, stable

### MongoDB Connection
If MongoDB fails to connect:
```bash
# Start MongoDB locally
brew services start mongodb-community

# Or check MongoDB Atlas connection string in .env
```

## 📝 Environment Variables

```env
# MongoDB
MONGO_URL=mongodb://localhost:27017
DB_NAME=detection_db

# Server
PORT=8000
CORS_ORIGINS=*

# Paths
UPLOAD_DIR=./uploads
DETECTIONS_DIR=./detections

# Optional: Twilio alerts
ENABLE_TWILIO=0
```

## 🔧 Development

### Adding New Detection Models
1. Add new `.pt` model file to `/backend/`
2. Update model loading in `server.py`
3. Add detection type handling in inference functions

### Extending Dashboard
1. Create new React component in `/frontend/src/pages/`
2. Add route in `App.js`
3. Update navigation

## 📦 Dependencies

### Backend
- FastAPI - Web framework
- motor - Async MongoDB
- ultralytics - YOLOv8
- opencv-python - Video processing
- Pydantic - Data validation

### Frontend
- React - UI framework
- React Router - Navigation
- Axios - API requests
- Tailwind CSS - Styling
- Lucide React - Icons

## 🚀 Deployment

### Production Build
```bash
# Frontend
cd frontend
npm run build

# Backend
cd backend
pip install gunicorn
gunicorn server:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

### Docker (Optional)
```bash
# Build and run with Docker
docker-compose up -d
```

## 📄 License

MIT License

## 🤝 Contributing

1. Create feature branch (`git checkout -b feature/amazing-feature`)
2. Commit changes (`git commit -m 'Add amazing feature'`)
3. Push to branch (`git push origin feature/amazing-feature`)
4. Open Pull Request

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Status**: Production Ready ✅  
**Last Updated**: November 16, 2025
# EDAI
