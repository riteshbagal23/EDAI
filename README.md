# SecureView Alert - AI-Powered Threat Detection System

## Overview
SecureView Alert is an advanced real-time threat detection system that uses multiple AI models to identify weapons, violence, and suspicious activities. The system provides instant alerts and stores tamper-proof evidence using IPFS technology.

## Core Features

### 🔫 Multi-Model Weapon Detection
- **Gun Detection Fusion**: Combines 4 YOLO models for maximum accuracy
  - `thermal.pt` (10% confidence) - Concealed weapon detection
  - `best.pt` (20% confidence) - Primary gun/knife detection
  - `best (1).pt` (30% confidence) - Verification model
  - `best (8).pt` (50% confidence) - High-confidence detection
- **Knife Detection**: 80% confidence threshold to reduce false positives
- **Thermal Gun Detection**: Detects weapons through thermal imaging

### 👁️ Additional Detection Capabilities
- **Violence Detection**: Real-time violence classification
- **People Counting**: Track number of people in frame
- **Top-View Detection**: Aerial/drone view human detection (`best (9).pt` at 30%)
- **Thermal Human Detection**: Heat signature-based person tracking

### 📡 Live Monitoring
- Real-time webcam feed processing
- Dual-stage gun verification system
- Instant threat alerts
- Violence detection overlay

### 📤 Video Upload & Analysis
- Support for image and video uploads
- Multiple detection modes (gun-fusion, thermal, topview, violence)
- Batch processing capability

---

## 🌐 IPFS Integration (Decentralized Evidence Storage)

### What is IPFS?
IPFS (InterPlanetary File System) is a distributed, peer-to-peer file storage system that makes detection evidence tamper-proof and permanently verifiable.

### How IPFS Works in This Project

#### 1. **Evidence Storage**
When a threat is detected:
- Detection image is uploaded to IPFS
- IPFS returns a unique **CID** (Content Identifier) - a cryptographic hash
- CID is stored in MongoDB along with detection metadata
- Original image is distributed across IPFS network nodes

#### 2. **Tamper-Proof Verification**
- **Content Addressing**: Each file's CID is based on its content
- **Immutability**: If anyone modifies the image, the CID changes completely
- **Verification**: Download image from IPFS using stored CID and verify it matches

#### 3. **Permanent & Distributed Storage**
- Files aren't stored on a single server - they're distributed across IPFS nodes
- No single point of failure
- Evidence remains accessible even if your server goes down
- Global accessibility for authorized parties

#### 4. **Chain of Custody**
```
Detection Event → Image Captured → Upload to IPFS → Receive CID → Store in Database
                                                                    ↓
Later Verification: Retrieve CID → Download from IPFS → Verify Content Integrity
```

### Benefits Over Traditional Storage

| Feature | Traditional Storage | IPFS Storage |
|---------|-------------------|--------------|
| **Tamper Detection** | ❌ Files can be modified | ✅ CID changes if modified |
| **Verification** | ❌ Requires blockchain | ✅ Built-in content addressing |
| **Availability** | ❌ Single server | ✅ Distributed network |
| **Censorship Resistance** | ❌ Can be taken down | ✅ Permanent once pinned |
| **Trust** | ❌ Trust the server | ✅ Cryptographic proof |

### Use Cases

1. **Legal Evidence**: Court-admissible proof that evidence hasn't been tampered
2. **Security Audits**: Third parties can verify detection authenticity
3. **Insurance Claims**: Immutable proof of security incidents
4. **Compliance**: Meet regulatory requirements for evidence integrity
5. **Multi-Site Deployments**: All locations share same evidence network

### API Endpoints (IPFS-Enabled)

- `POST /api/ipfs/upload` - Upload detection image to IPFS
- `GET /api/ipfs/verify/{cid}` - Verify image integrity
- `GET /api/ipfs/retrieve/{cid}` - Download image from IPFS
- `GET /api/detections` - List all detections with IPFS CIDs

---

## Detection Models

### Gun Fusion Detection
| Model | File | Confidence | Purpose |
|-------|------|------------|---------|
| Thermal | `thermal.pt` | 10% | Concealed weapons |
| Primary | `best.pt` | 20% | Main detection |
| Verification | `best (1).pt` | 30% | Cross-check |
| High-Conf | `best (8).pt` | 50% | Final validation |

### Specialized Models
- **Violence**: `best (2).pt` - Violence classification
- **Top-View**: `best (9).pt` (30%) - Aerial human detection
- **Thermal Human**: `thermalhuman.pt` - Heat signature detection
- **People Counting**: `yolov8n.pt` - General person detection

---

## Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **AI/ML**: YOLO (Ultralytics), OpenCV
- **Database**: MongoDB (Motor async driver)
- **Storage**: IPFS (ipfshttpclient)
- **Alerts**: Twilio (SMS & Voice)
- **Video Processing**: OpenCV, FFmpeg

### Frontend
- **Framework**: React
- **UI Components**: Custom components + Lucide icons
- **Maps**: Google Maps integration
- **Styling**: Custom CSS
- **HTTP Client**: Axios

---

## Installation

### Prerequisites
```bash
- Python 3.12+
- Node.js 16+
- MongoDB
- IPFS Desktop or IPFS daemon
```

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables
Create `.env` file in backend directory:
```env
MONGO_URL=mongodb://localhost:27017
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_number
ALERT_PHONE_NUMBER=your_alert_number
IPFS_API=/ip4/127.0.0.1/tcp/5001  # Local IPFS node
```

### IPFS Setup
```bash
# Install IPFS Desktop from https://ipfs.io
# Or use IPFS CLI:
ipfs init
ipfs daemon
```

### Frontend Setup
```bash
cd frontend
npm install
```

Create `.env` in frontend directory:
```env
REACT_APP_BACKEND_URL=http://localhost:8000
```

---

## Running the Application

### Start IPFS
```bash
ipfs daemon
```

### Start Backend
```bash
cd backend
uvicorn server:app --host 127.0.0.1 --port 8000
```

### Start Frontend
```bash
cd frontend
npm start
```

Access at: `http://localhost:3000`

---

## API Endpoints

### Detection
- `POST /api/test-upload` - Upload image/video for analysis
- `GET /api/detections` - List all detections
- `GET /api/detection-stats` - Get detection statistics

### IPFS
- `POST /api/ipfs/upload` - Upload to IPFS
- `GET /api/ipfs/verify/{cid}` - Verify IPFS content
- `GET /api/ipfs/{cid}` - Retrieve from IPFS

### Maps
- Detection cards include "View on Google Maps" links
- Interactive map view with color-coded markers

---

## Confidence Thresholds

| Detection Type | Confidence | Rationale |
|---------------|------------|-----------|
| Knife | 80% | High threshold to reduce false positives |
| Pistol (Stage 1) | 50% | Initial detection |
| Pistol (Stage 2) | 60% | Verification required |
| Thermal Gun | 10% | Sensitive to concealed threats |
| Top-View | 30% | Aerial detection sensitivity |
| Violence | Classified | Binary classification |

---

## Features

### Maps Integration
- **Detection Cards**: "View on Google Maps" link for each detection
- **Interactive Map**: Toggle between List/Map view
- **Color-Coded Markers**:
  - 🔴 Red: Pistol
  - 🟠 Orange: Knife
  - 🟣 Magenta: Thermal Gun
  - 🟡 Yellow: Top-View Person

### Alert System
- SMS alerts via Twilio
- Voice call notifications
- Cooldown period to prevent alert spam

---

## Project Structure

```
myproject/
├── backend/
│   ├── server.py                 # Main FastAPI application
│   ├── requirements.txt          # Python dependencies
│   ├── .env                      # Environment variables
│   ├── uploads/                  # Temporary upload storage
│   ├── detections/               # Detection images
│   └── models/                   # YOLO model files (.pt)
│
├── frontend/
│   ├── src/
│   │   ├── pages/               # React pages
│   │   │   ├── Detections.js    # Main admin page with maps
│   │   │   ├── LiveMonitoring.js
│   │   │   ├── Upload.js
│   │   │   └── ...
│   │   ├── components/          # Reusable components
│   │   │   └── DetectionMapView.js
│   │   └── App.js               # Main React app
│   ├── package.json             # Node dependencies
│   └── .env                     # Frontend env vars
│
└── README.md                     # This file
```

---

## Future Enhancements

- [ ] IPFS Pinning Service integration (Pinata, Web3.Storage)
- [ ] Multi-camera support
- [ ] Real-time dashboard with WebSocket updates
- [ ] Analytics and reporting
- [ ] Mobile app integration
- [ ] Edge device deployment

---

## License
Proprietary - All Rights Reserved

## Support
For issues or questions, contact the development team.

---

**Built with IPFS for tamper-proof, decentralized evidence storage** 🌐
