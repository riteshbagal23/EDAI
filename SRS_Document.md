# Software Requirements Specification (SRS)
## SecureView Alert - AI-Powered Threat Detection System

**Version 1.0**  
**Date:** January 21, 2026  
**Prepared by:** Development Team  

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 [Purpose](#11-purpose)
   - 1.2 [Scope](#12-scope)
   - 1.3 [Definitions, Acronyms, and Abbreviations](#13-definitions-acronyms-and-abbreviations)
   - 1.4 [References](#14-references)
   - 1.5 [Overview](#15-overview)

2. [Overall Description](#2-overall-description)
   - 2.1 [Product Perspective](#21-product-perspective)
   - 2.2 [Product Functions](#22-product-functions)
   - 2.3 [User Characteristics](#23-user-characteristics)
   - 2.4 [Constraints](#24-constraints)
   - 2.5 [Assumptions and Dependencies](#25-assumptions-and-dependencies)

3. [Specific Requirements](#3-specific-requirements)
   - 3.1 [Functional Requirements](#31-functional-requirements)
   - 3.2 [External Interface Requirements](#32-external-interface-requirements)
   - 3.3 [Performance Requirements](#33-performance-requirements)
   - 3.4 [Design Constraints](#34-design-constraints)
   - 3.5 [Software System Attributes](#35-software-system-attributes)

4. [Appendices](#4-appendices)

---

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) document provides a complete description of all functions and specifications for the SecureView Alert system. The document is intended for:

- **Development Team**: To understand and implement the system requirements
- **Stakeholders**: To validate that the system meets organizational security needs
- **Quality Assurance Team**: To develop test cases and validation criteria
- **System Administrators**: To understand deployment and operational requirements
- **End Users**: To understand system capabilities and limitations

### 1.2 Scope

**Product Name:** SecureView Alert - AI-Powered Threat Detection System

**Product Objectives:**

SecureView Alert is an advanced real-time threat detection and monitoring system designed to enhance security through AI-powered surveillance. The system provides:

- **Real-time Threat Detection**: Identifies weapons (guns, knives), violence, and suspicious activities using multiple YOLO-based AI models
- **Multi-Camera Monitoring**: Supports multiple camera feeds including standard RGB and thermal imaging
- **Instant Alerting**: Provides immediate notifications via SMS and voice calls through Twilio integration
- **Tamper-Proof Evidence Storage**: Utilizes IPFS (InterPlanetary File System) for decentralized, immutable storage of detection evidence
- **Administrative Dashboard**: Web-based interface for monitoring, analytics, and verification
- **Batch Processing**: Supports upload and analysis of pre-recorded video footage

**Key Benefits:**

- Enhanced security through AI-powered threat detection
- Reduced response time with instant alerts
- Permanent, verifiable evidence storage
- Comprehensive analytics and reporting
- Scalable multi-camera deployment

### 1.3 Definitions, Acronyms, and Abbreviations

| Term | Definition |
|------|------------|
| **AI** | Artificial Intelligence |
| **API** | Application Programming Interface |
| **CID** | Content Identifier (IPFS unique hash) |
| **CORS** | Cross-Origin Resource Sharing |
| **CNN** | Convolutional Neural Network |
| **FPS** | Frames Per Second |
| **GUI** | Graphical User Interface |
| **HTTP/HTTPS** | HyperText Transfer Protocol (Secure) |
| **IPFS** | InterPlanetary File System - Distributed peer-to-peer file storage |
| **ML** | Machine Learning |
| **MongoDB** | NoSQL Document Database |
| **REST** | Representational State Transfer |
| **SRS** | Software Requirements Specification |
| **UI/UX** | User Interface / User Experience |
| **YOLO** | You Only Look Once - Real-time object detection algorithm |
| **WebRTC** | Web Real-Time Communication |

### 1.4 References

- IEEE Standard 830-1998: IEEE Recommended Practice for Software Requirements Specifications
- YOLOv8 Documentation: Ultralytics YOLO
- IPFS Documentation: InterPlanetary File System Protocol
- Twilio API Documentation: SMS and Voice API
- Flask Documentation: Python Web Framework
- React Documentation: JavaScript UI Library
- MongoDB Documentation: NoSQL Database

### 1.5 Overview

This SRS document is organized according to IEEE 830-1998 standard. Section 2 provides an overall description of the SecureView Alert system, including product perspective, functions, user characteristics, and constraints. Section 3 details specific functional and non-functional requirements. Section 4 contains supplementary information in appendices.

---

## 2. Overall Description

### 2.1 Product Perspective

SecureView Alert is a standalone, self-contained system designed for deployment in security-critical environments such as:

- Educational institutions (schools, universities)
- Commercial establishments (malls, stores, offices)
- Public spaces (airports, train stations, parks)
- Government facilities
- Private properties

**System Context:**

```
┌─────────────────────────────────────────────────────────────┐
│                    SecureView Alert System                   │
│  ┌────────────────────────────┐  ┌─────────────────────────┐│
│  │      Backend Server         │  │   Frontend Dashboard    ││
│  │   (Python/Flask/FastAPI)    │  │      (React.js)         ││
│  │                             │  │                         ││
│  │ - AI Model Processing       │  │ - Live Monitoring       ││
│  │ - Camera Management         │  │ - Analytics View        ││
│  │ - IPFS Integration          │  │ - Detection Review      ││
│  │ - Alert System              │  │ - Admin Verification    ││
│  └────────────────────────────┘  └─────────────────────────┘│
│         │          │          │                              │
└─────────┼──────────┼──────────┼──────────────────────────────┘
          │          │          │
    ┌─────┘          │          └────────┐
    ▼                ▼                   ▼
┌─────────┐    ┌──────────┐       ┌──────────┐
│ Camera  │    │  IPFS    │       │ Twilio   │
│ Devices │    │  Network │       │ Alerts   │
└─────────┘    └──────────┘       └──────────┘
```

**System Interfaces:**

1. **Camera Input**: IP cameras, USB webcams, thermal cameras
2. **IPFS Network**: Distributed file storage for evidence
3. **Twilio API**: SMS and voice alert delivery
4. **MongoDB**: Detection metadata and system data
5. **Web Browser**: User interface access

### 2.2 Product Functions

The major functions of SecureView Alert include:

**F1. Multi-Model Weapon Detection**
- Fusion of 4 YOLO models for gun detection with varying confidence thresholds
- Knife detection with 80% confidence threshold
- Thermal imaging-based concealed weapon detection

**F2. Violence Detection**
- Real-time violence classification using specialized AI models
- Activity analysis and threat assessment

**F3. People Detection and Counting**
- Accurate person counting in frame
- Thermal signature-based human tracking
- Top-view/aerial detection for overhead cameras

**F4. Live Camera Monitoring**
- Real-time video feed processing
- Multi-camera simultaneous monitoring
- Dual-stage threat verification

**F5. Video Upload and Analysis**
- Batch processing of uploaded images/videos
- Multiple detection mode support
- Historical footage analysis

**F6. IPFS Evidence Storage**
- Decentralized storage of detection images
- Tamper-proof evidence with cryptographic hashing (CID)
- Permanent record retention

**F7. Alert Management**
- Instant SMS notifications for threats
- Voice call alerts for critical detections
- Configurable alert thresholds and cooldown periods

**F8. Administrative Dashboard**
- Real-time monitoring interface
- Detection history and analytics
- Camera management
- Verification workflow for detected threats

**F9. Analytics and Reporting**
- Detection statistics and trends
- Camera performance metrics
- Heat maps and geographic visualization

### 2.3 User Characteristics

**User Class 1: Security Personnel**
- **Technical Expertise**: Moderate (familiar with security systems)
- **Primary Tasks**: Monitor live feeds, respond to alerts, verify detections
- **Frequency of Use**: Continuous during shifts
- **Special Needs**: Mobile access, clear alert notifications

**User Class 2: System Administrators**
- **Technical Expertise**: High (IT/Security professionals)
- **Primary Tasks**: Configure cameras, manage system settings, maintain system health
- **Frequency of Use**: Regular (daily/weekly for maintenance)
- **Special Needs**: Advanced configuration options, system logs

**User Class 3: Management/Supervisors**
- **Technical Expertise**: Low to Moderate
- **Primary Tasks**: Review analytics, generate reports, assess security posture
- **Frequency of Use**: Periodic (weekly/monthly reviews)
- **Special Needs**: High-level dashboards, export capabilities

**User Class 4: Law Enforcement (Evidence Retrieval)**
- **Technical Expertise**: Low
- **Primary Tasks**: Access verified detection evidence via IPFS
- **Frequency of Use**: Incident-based
- **Special Needs**: Evidence verification, chain of custody

### 2.4 Constraints

**C1. Regulatory Constraints**
- Must comply with privacy laws and surveillance regulations
- Data retention policies must be configurable per jurisdiction
- GDPR/privacy law compliance for facial data

**C2. Hardware Constraints**
- Requires GPU for optimal AI model performance (NVIDIA CUDA-compatible recommended)
- Minimum 8GB RAM for simultaneous multi-model processing
- Network bandwidth for multiple camera streams

**C3. Software Constraints**
- Backend: Python 3.8+, Flask/FastAPI framework
- Frontend: Modern web browsers (Chrome, Firefox, Safari, Edge)
- Database: MongoDB 4.4+
- IPFS node connectivity required

**C4. Interface Constraints**
- Must support standard camera protocols (RTSP, HTTP, USB)
- REST API for external system integration
- Web-based interface (no native mobile app in v1.0)

**C5. Performance Constraints**
- Real-time processing: 15-30 FPS per camera stream
- Alert delivery: < 5 seconds from detection to notification
- Maximum 30-second alert cooldown period

**C6. Security Constraints**
- Encrypted communication (HTTPS/TLS)
- Authentication and authorization required
- Secure API key management for Twilio

### 2.5 Assumptions and Dependencies

**Assumptions:**

A1. Cameras are properly installed with clear field of view  
A2. Network connectivity is stable and reliable  
A3. IPFS network nodes are accessible  
A4. Twilio account has sufficient credits for alerts  
A5. GPU hardware is available for production deployment  
A6. System administrators have basic IT knowledge  
A7. Lighting conditions are adequate for RGB cameras  

**Dependencies:**

D1. **YOLO Model Dependencies**: Ultralytics library and pre-trained models  
D2. **IPFS Dependencies**: Running IPFS daemon or access to IPFS pinning service  
D3. **Twilio Dependencies**: Active Twilio account with API credentials  
D4. **Database Dependencies**: MongoDB server availability  
D5. **Python Libraries**: OpenCV, Flask, PyTorch, and other Python packages  
D6. **Network Dependencies**: Internet connectivity for IPFS and Twilio  
D7. **Browser Dependencies**: HTML5 video support for live streaming  

---

## 3. Specific Requirements

### 3.1 Functional Requirements

#### 3.1.1 Weapon Detection Module

**FR1.1: Multi-Model Gun Detection Fusion**

- **Description**: The system shall detect firearms using a fusion of four YOLO models with different confidence thresholds
- **Input**: Video frame (1280x720 minimum resolution)
- **Processing**:
  - Model 1 (`thermal.pt`): 10% confidence threshold for concealed weapons
  - Model 2 (`best.pt`): 20% confidence threshold for primary detection
  - Model 3 (`best (1).pt`): 30% confidence threshold for verification
  - Model 4 (`best (8).pt`): 50% confidence threshold for high-confidence detection
- **Output**: Gun detection event with confidence score, bounding box coordinates, timestamp
- **Priority**: Critical
- **Dependencies**: YOLOv8 models, GPU processing

**FR1.2: Knife Detection**

- **Description**: The system shall detect knives and bladed weapons
- **Input**: Video frame
- **Processing**: YOLO model inference with 80% confidence threshold
- **Output**: Knife detection event with metadata
- **Priority**: High
- **Dependencies**: Trained knife detection model

**FR1.3: Thermal Weapon Detection**

- **Description**: The system shall detect concealed weapons using thermal imaging
- **Input**: Thermal camera feed
- **Processing**: Thermal-specific YOLO model (`thermal.pt`, `thermalhuman.pt`)
- **Output**: Thermal weapon detection event
- **Priority**: High
- **Dependencies**: Thermal camera hardware, thermal-trained models

#### 3.1.2 Violence Detection Module

**FR2.1: Real-time Violence Classification**

- **Description**: The system shall identify violent activities in video streams
- **Input**: Video frame sequence
- **Processing**: Violence detection model inference
- **Output**: Violence detection event with severity classification
- **Priority**: High
- **Dependencies**: Violence detection model

**FR2.2: Activity Analysis**

- **Description**: The system shall analyze human activities for threat assessment
- **Input**: Person bounding boxes and temporal data
- **Processing**: Activity classification algorithm
- **Output**: Activity classification (normal, suspicious, violent)
- **Priority**: Medium

#### 3.1.3 People Detection Module

**FR3.1: Person Counting**

- **Description**: The system shall accurately count the number of people in frame
- **Input**: Video frame
- **Processing**: YOLOv8m model for person detection
- **Output**: Person count, individual bounding boxes
- **Priority**: Medium
- **Dependencies**: YOLOv8m model

**FR3.2: Thermal Human Detection**

- **Description**: The system shall detect humans via heat signatures
- **Input**: Thermal camera feed
- **Processing**: Thermal human detection model
- **Output**: Human detection with thermal signature
- **Priority**: Medium
- **Dependencies**: Thermal camera, thermal human model

**FR3.3: Top-View Detection**

- **Description**: The system shall detect people from overhead/aerial views
- **Input**: Top-view camera feed
- **Processing**: Top-view specific model (`best (9).pt`) at 30% confidence
- **Output**: Person detection from aerial perspective
- **Priority**: Low
- **Dependencies**: Top-view trained model

#### 3.1.4 Camera Management Module

**FR4.1: Camera Registration**

- **Description**: The system shall allow administrators to add and configure cameras
- **Input**: Camera name, stream URL, location, detection modes
- **Processing**: Validate camera connectivity, store configuration
- **Output**: Camera registered in system
- **Priority**: High

**FR4.2: Multi-Camera Monitoring**

- **Description**: The system shall support simultaneous monitoring of multiple cameras
- **Input**: Multiple camera streams
- **Processing**: Parallel processing of video feeds
- **Output**: Aggregated detection results across cameras
- **Priority**: High
- **Constraints**: Limited by available GPU/CPU resources

**FR4.3: Camera Status Monitoring**

- **Description**: The system shall monitor camera health and connectivity
- **Input**: Camera stream status
- **Processing**: Periodic health checks
- **Output**: Camera online/offline status, error alerts
- **Priority**: Medium

#### 3.1.5 IPFS Evidence Storage Module

**FR5.1: Detection Image Upload to IPFS**

- **Description**: The system shall upload detection images to IPFS network
- **Input**: Detection image (JPEG/PNG)
- **Processing**: Upload to IPFS, receive CID
- **Output**: IPFS CID (Content Identifier)
- **Priority**: High
- **Dependencies**: IPFS daemon running

**FR5.2: CID Storage in Database**

- **Description**: The system shall store IPFS CIDs with detection metadata
- **Input**: CID, detection details, timestamp
- **Processing**: Database insert operation
- **Output**: Detection record with IPFS reference
- **Priority**: High
- **Dependencies**: MongoDB connection

**FR5.3: Evidence Retrieval**

- **Description**: The system shall retrieve detection images from IPFS using CID
- **Input**: IPFS CID
- **Processing**: IPFS gateway retrieval
- **Output**: Detection image
- **Priority**: High
- **Dependencies**: IPFS network connectivity

**FR5.4: Batch IPFS Upload**

- **Description**: The system shall support batch upload of historical detections to IPFS
- **Input**: List of detection records without CIDs
- **Processing**: Iterate and upload images to IPFS
- **Output**: Updated detection records with CIDs
- **Priority**: Medium

#### 3.1.6 Alert Management Module

**FR6.1: SMS Alert Delivery**

- **Description**: The system shall send SMS alerts for threat detections
- **Input**: Detection event, recipient phone number(s)
- **Processing**: Format alert message, call Twilio API
- **Output**: SMS sent confirmation
- **Priority**: Critical
- **Dependencies**: Twilio account, API credentials

**FR6.2: Voice Call Alerts**

- **Description**: The system shall initiate voice calls for critical threats
- **Input**: Critical detection event, recipient phone number(s)
- **Processing**: Generate voice message, call Twilio API
- **Output**: Voice call initiated
- **Priority**: High
- **Dependencies**: Twilio voice API

**FR6.3: Alert Cooldown**

- **Description**: The system shall implement cooldown period between alerts
- **Input**: Alert cooldown configuration (default 30 seconds)
- **Processing**: Track last alert time, suppress duplicate alerts
- **Output**: Alert throttling
- **Priority**: Medium
- **Rationale**: Prevent alert fatigue

**FR6.4: Alert Configuration**

- **Description**: The system shall allow customization of alert settings
- **Input**: Alert thresholds, recipient lists, cooldown periods
- **Processing**: Validate and store configuration
- **Output**: Updated alert settings
- **Priority**: Medium

#### 3.1.7 Video Upload and Processing Module

**FR7.1: File Upload**

- **Description**: The system shall accept image and video file uploads
- **Input**: Image files (JPEG, PNG) or video files (MP4, AVI)
- **Processing**: Validate file type and size, store temporarily
- **Output**: Upload confirmation, file ID
- **Priority**: High
- **Constraints**: Maximum file size 100MB

**FR7.2: Detection Mode Selection**

- **Description**: The system shall support multiple detection modes for uploads
- **Input**: Detection mode selection (gun-fusion, thermal, topview, violence)
- **Processing**: Route to appropriate detection pipeline
- **Output**: Detection results based on selected mode
- **Priority**: High

**FR7.3: Batch Processing**

- **Description**: The system shall process multiple uploaded files
- **Input**: Multiple image/video files
- **Processing**: Queue-based processing
- **Output**: Aggregated detection results
- **Priority**: Medium

#### 3.1.8 Dashboard and Visualization Module

**FR8.1: Live Feed Display**

- **Description**: The system shall display live camera feeds with detection overlays
- **Input**: Camera streams
- **Processing**: Render video with bounding boxes and labels
- **Output**: Live monitoring interface
- **Priority**: High

**FR8.2: Detection History**

- **Description**: The system shall display historical detection records
- **Input**: Time range, filter criteria
- **Processing**: Query database, retrieve detections
- **Output**: Paginated detection list with thumbnails
- **Priority**: High

**FR8.3: Analytics Dashboard**

- **Description**: The system shall display detection statistics and trends
- **Input**: Date range
- **Processing**: Aggregate detection data, generate visualizations
- **Output**: Charts and graphs (detections over time, by type, by camera)
- **Priority**: Medium

**FR8.4: Map View**

- **Description**: The system shall display detections on geographic map
- **Input**: Camera locations, recent detections
- **Processing**: Geocoding, map rendering
- **Output**: Interactive map with detection markers
- **Priority**: Low

**FR8.5: Admin Verification Interface**

- **Description**: The system shall provide interface for verifying detections
- **Input**: Detection record
- **Processing**: Display detection details and image
- **Output**: Verification status (verified, false positive, pending)
- **Priority**: High
- **Rationale**: Reduce false positives, maintain accuracy

#### 3.1.9 User Authentication and Authorization

**FR9.1: User Login**

- **Description**: The system shall authenticate users
- **Input**: Username/email, password
- **Processing**: Credential verification
- **Output**: Authentication token, user session
- **Priority**: High

**FR9.2: Role-Based Access Control**

- **Description**: The system shall enforce role-based permissions
- **Input**: User role (admin, operator, viewer)
- **Processing**: Permission checking
- **Output**: Access granted/denied to features
- **Priority**: High
- **Roles**:
  - Admin: Full system access
  - Operator: Monitor and verify detections
  - Viewer: Read-only access to dashboards

**FR9.3: Session Management**

- **Description**: The system shall manage user sessions
- **Input**: User login
- **Processing**: Token generation, session timeout
- **Output**: Active session or logout
- **Priority**: Medium

### 3.2 External Interface Requirements

#### 3.2.1 User Interfaces

**UI1: Dashboard Interface**
- **Layout**: Responsive web design (desktop and tablet optimized)
- **Components**:
  - Navigation sidebar with menu items
  - Main content area for page-specific views
  - Real-time notification badge
  - Alert status indicator
- **Technology**: React.js, Tailwind CSS, Framer Motion
- **Accessibility**: WCAG 2.1 Level AA compliance

**UI2: Live Monitoring Interface**
- **Layout**: Grid view for multiple camera feeds
- **Components**:
  - Video player with detection overlays
  - Camera selector dropdown
  - Detection statistics panel
  - Alert history ticker
- **Features**: Full-screen mode, snapshot capture

**UI3: Detection Management Interface**
- **Layout**: Table/card view with filters
- **Components**:
  - Detection list with thumbnails
  - Filter by date, type, camera, status
  - Pagination controls
  - Detail modal for detection review
  - Verification buttons

**UI4: Upload Interface**
- **Layout**: Drag-and-drop upload area
- **Components**:
  - File selector
  - Detection mode selector
  - Upload progress indicator
  - Results display

**UI5: Analytics Interface**
- **Layout**: Dashboard with charts
- **Components**:
  - Date range picker
  - Detection trend charts
  - Camera performance metrics
  - Export functionality

**UI6: Camera Management Interface**
- **Layout**: List/grid view of cameras
- **Components**:
  - Add camera form
  - Camera status indicators
  - Edit/delete actions
  - Test connection button

**UI7: Settings Interface**
- **Layout**: Tabbed settings panel
- **Components**:
  - Alert configuration
  - Detection thresholds
  - User management
  - System preferences

#### 3.2.2 Hardware Interfaces

**HI1: Camera Interface**
- **Supported Protocols**: RTSP, HTTP/HTTPS, USB (V4L2)
- **Video Formats**: H.264, H.265, MJPEG
- **Resolution**: Minimum 640x480, recommended 1280x720 or higher
- **Frame Rate**: 15-30 FPS
- **Camera Types**:
  - RGB/Standard cameras
  - Thermal imaging cameras (FLIR, Seek, etc.)
  - IP network cameras
  - USB webcams

**HI2: GPU Hardware**
- **Interface**: CUDA-compatible NVIDIA GPU
- **Minimum**: NVIDIA GTX 1060 or equivalent (6GB VRAM)
- **Recommended**: NVIDIA RTX 3060 or higher (12GB+ VRAM)
- **Purpose**: Accelerated AI model inference

#### 3.2.3 Software Interfaces

**SI1: IPFS Interface**
- **Component**: IPFS Daemon / HTTP API
- **Version**: IPFS 0.10+
- **Communication**: HTTP REST API (default port 5001)
- **Operations**:
  - `POST /api/v0/add`: Upload files
  - `GET /ipfs/{CID}`: Retrieve files
  - `POST /api/v0/pin/add`: Pin content

**SI2: Twilio API**
- **Component**: Twilio SMS and Voice APIs
- **Version**: Twilio REST API v2010-04-01
- **Communication**: HTTPS REST API
- **Authentication**: Account SID and Auth Token
- **Operations**:
  - Send SMS messages
  - Initiate voice calls
  - TwiML voice scripts

**SI3: MongoDB Database**
- **Component**: MongoDB Server
- **Version**: MongoDB 4.4+
- **Communication**: MongoDB Wire Protocol (default port 27017)
- **Collections**:
  - `detections`: Detection records
  - `cameras`: Camera configurations
  - `users`: User accounts
  - `verifications`: Admin verifications
  - `alerts`: Alert history

**SI4: Python Backend Libraries**
- **Flask/FastAPI**: Web framework (REST API)
- **OpenCV**: Video processing (cv2)
- **Ultralytics**: YOLO model interface
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical processing
- **Pillow**: Image manipulation

**SI5: Frontend Libraries**
- **React**: UI framework (v18+)
- **Axios**: HTTP client
- **React Router**: Navigation
- **Framer Motion**: Animations
- **Lucide React**: Icons
- **Tailwind CSS**: Styling

#### 3.2.4 Communication Interfaces

**CI1: HTTP/HTTPS API**
- **Protocol**: HTTP/1.1, HTTPS (TLS 1.2+)
- **Format**: RESTful JSON API
- **Port**: 8000 (backend), 3000 (frontend development)
- **Authentication**: JWT tokens / API keys
- **Endpoints**: See API specification in Appendix A

**CI2: WebSocket (Live Streaming)**
- **Protocol**: WebSocket (ws/wss)
- **Purpose**: Real-time video streaming
- **Format**: Binary frames or base64 encoded JPEG
- **Fallback**: HTTP polling if WebSocket unavailable

**CI3: CORS Configuration**
- **Purpose**: Allow frontend-backend communication
- **Allowed Origins**: Configured domains
- **Allowed Methods**: GET, POST, PUT, DELETE, OPTIONS
- **Credentials**: Enabled for authenticated requests

### 3.3 Performance Requirements

**PR1: Detection Processing Speed**
- **Requirement**: Process video frames at minimum 15 FPS per camera
- **Target**: 30 FPS for single camera, 15 FPS for 4 concurrent cameras
- **Measurement**: Average frames processed per second over 60-second interval
- **Rationale**: Real-time threat detection requires immediate processing

**PR2: Alert Delivery Time**
- **Requirement**: Alert delivery within 5 seconds of threat detection
- **Measurement**: Time from detection event to SMS/call initiation
- **Rationale**: Critical for rapid security response

**PR3: System Response Time**
- **Requirement**: API response time < 200ms for 95% of requests
- **Measurement**: Server response time from request to response
- **Exceptions**: Video upload/processing requests may take longer

**PR4: Concurrent Camera Support**
- **Requirement**: Support minimum 4 simultaneous camera streams
- **Target**: 8-16 cameras with appropriate hardware
- **Dependency**: GPU memory and processing power

**PR5: Database Query Performance**
- **Requirement**: Detection history queries return results within 1 second
- **Measurement**: Database query execution time
- **Strategy**: Indexed queries, pagination

**PR6: IPFS Upload Performance**
- **Requirement**: Upload detection images to IPFS within 10 seconds
- **Measurement**: Time from image capture to CID receipt
- **Dependency**: IPFS network speed, file size

**PR7: Video Upload Processing**
- **Requirement**: Process uploaded videos at minimum 1x speed
- **Example**: 60-second video processed in ≤ 60 seconds
- **Dependency**: GPU availability, video resolution

**PR8: System Uptime**
- **Requirement**: 99.5% uptime during operational hours
- **Measurement**: System availability over monthly period
- **Downtime**: Scheduled maintenance excluded

**PR9: Memory Usage**
- **Requirement**: Backend process memory < 8GB with 4 cameras
- **Measurement**: Peak memory consumption
- **Rationale**: Support deployment on standard servers

**PR10: Storage Requirements**
- **Database**: Estimate 100 KB per detection record
- **Local Storage**: Temporary video storage, logs
- **IPFS**: Evidence images stored on distributed network

### 3.4 Design Constraints

**DC1: Programming Languages**
- **Backend**: Python 3.8 or higher
- **Frontend**: JavaScript (ES6+), JSX
- **Rationale**: Ecosystem support for AI/ML libraries

**DC2: Framework Constraints**
- **Backend**: Flask or FastAPI (Python web frameworks)
- **Frontend**: React.js
- **Rationale**: Industry standard, extensive libraries

**DC3: Database Constraint**
- **Database**: MongoDB (NoSQL document store)
- **Rationale**: Flexible schema for detection metadata

**DC4: AI Model Format**
- **Format**: YOLO (Ultralytics framework)
- **Versions**: YOLOv8 models (.pt format)
- **Rationale**: State-of-art real-time object detection

**DC5: Video Processing Library**
- **Library**: OpenCV (cv2)
- **Rationale**: Industry standard for computer vision

**DC6: Deployment Platform**
- **OS**: Linux (Ubuntu 20.04+), macOS (development)
- **Containerization**: Docker support recommended
- **Rationale**: Production stability

**DC7: Browser Compatibility**
- **Supported**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Rationale**: Modern web standards (HTML5 video, WebSocket)

**DC8: Network Architecture**
- **Architecture**: Client-server (REST API)
- **Communication**: JSON over HTTP/HTTPS
- **Rationale**: Stateless, scalable architecture

### 3.5 Software System Attributes

#### 3.5.1 Reliability

**REL1: Error Handling**
- The system shall gracefully handle camera disconnections without crashing
- Failed IPFS uploads shall be retried up to 3 times
- Database connection failures shall be logged and retried

**REL2: Data Integrity**
- Detection records shall include checksums for verification
- IPFS CIDs provide cryptographic integrity verification
- Database transactions shall be atomic

**REL3: Fault Tolerance**
- Camera failure shall not affect other cameras
- Individual model failures shall fall back to other models
- Alert delivery failures shall be logged and retried

**REL4: Recovery**
- System shall automatically reconnect to cameras after network recovery
- Unsent alerts shall be queued and delivered upon service restoration
- System state shall be recoverable after unexpected shutdown

#### 3.5.2 Availability

**AVL1: System Availability**
- Target: 99.5% uptime during operational hours
- Scheduled maintenance windows allowed
- Redundant deployment recommended for critical installations

**AVL2: Service Degradation**
- System shall continue operating with reduced camera count if some fail
- Alert delivery via SMS shall continue if voice calls fail
- Local storage fallback if IPFS is unavailable

#### 3.5.3 Security

**SEC1: Authentication**
- User authentication required for dashboard access
- JWT token-based session management
- Password complexity requirements enforced

**SEC2: Authorization**
- Role-based access control (Admin, Operator, Viewer)
- API endpoints protected by authentication middleware
- Camera access restricted by user permissions

**SEC3: Data Encryption**
- HTTPS/TLS for all web communications
- API credentials stored securely (environment variables, secrets management)
- Database connections encrypted

**SEC4: Privacy**
- Personal data anonymization where required
- Configurable data retention policies
- GDPR compliance support (data export, deletion)

**SEC5: Audit Logging**
- User actions logged (login, verification, configuration changes)
- Detection events logged with timestamps
- Alert delivery logged for accountability

#### 3.5.4 Maintainability

**MNT1: Code Organization**
- Modular architecture with clear separation of concerns
- Backend routes, services, and models separated
- Frontend components, pages, and utilities organized

**MNT2: Configuration Management**
- External configuration files for settings
- Environment variables for credentials
- No hard-coded credentials or endpoints

**MNT3: Logging**
- Comprehensive logging at INFO, WARNING, ERROR levels
- Structured log format for parsing
- Configurable log levels and destinations

**MNT4: Documentation**
- Code comments for complex logic
- API documentation (OpenAPI/Swagger)
- User manual and admin guide

**MNT5: Version Control**
- Git-based version control
- Semantic versioning for releases
- Change log maintenance

#### 3.5.5 Portability

**PORT1: Platform Independence**
- Backend: Cross-platform Python application
- Frontend: Browser-based (no platform-specific code)
- Database: MongoDB available on all platforms

**PORT2: Containerization**
- Docker support for consistent deployment
- Docker Compose for multi-service orchestration
- Portable across cloud providers

**PORT3: Configuration Portability**
- Environment-based configuration
- No hardcoded paths or platform-specific code
- Relative paths where possible

#### 3.5.6 Scalability

**SCALE1: Horizontal Scaling**
- Backend API can be load-balanced across multiple servers
- Camera processing can be distributed
- MongoDB supports sharding for large datasets

**SCALE2: Vertical Scaling**
- Support for GPU upgrades to increase camera capacity
- Memory scalable with increased camera count
- CPU cores utilized for parallel processing

**SCALE3: Data Scaling**
- IPFS provides distributed storage (no single-server limit)
- Database indexes for fast querying at scale
- Pagination for large result sets

#### 3.5.7 Usability

**USE1: User Interface Design**
- Intuitive navigation with clear menu structure
- Responsive design for desktop and tablet
- Consistent visual language and branding

**USE2: Learnability**
- User manual with screenshots
- Tooltips and help text in interface
- Logical workflow for common tasks

**USE3: Accessibility**
- Keyboard navigation support
- Screen reader compatibility
- Color contrast for readability

**USE4: Error Messages**
- Clear, actionable error messages
- User-friendly language (avoid technical jargon)
- Suggestions for resolution

---

## 4. Appendices

### Appendix A: API Endpoint Specification

**Base URL**: `http://localhost:8000/api`

#### Detection Endpoints

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/detections` | Get detection list | Query: limit, offset, camera_id | Array of detection objects |
| GET | `/detections/{id}` | Get detection by ID | Path: id | Detection object |
| POST | `/detections/verify` | Verify detection | Body: {id, status} | Updated detection |
| DELETE | `/detections/{id}` | Delete detection | Path: id | Success message |

#### Camera Endpoints

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/cameras` | List all cameras | None | Array of camera objects |
| POST | `/cameras` | Add new camera | Body: camera config | Created camera object |
| PUT | `/cameras/{id}` | Update camera | Path: id, Body: updates | Updated camera |
| DELETE | `/cameras/{id}` | Remove camera | Path: id | Success message |
| GET | `/cameras/{id}/status` | Check camera status | Path: id | Status object |

#### Upload Endpoints

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| POST | `/upload/image` | Upload image for detection | Body: multipart/form-data | Detection results |
| POST | `/upload/video` | Upload video for detection | Body: multipart/form-data | Detection results |
| POST | `/upload/batch` | Batch upload multiple files | Body: multipart/form-data | Array of results |

#### Analytics Endpoints

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/analytics/stats` | Get detection statistics | Query: start_date, end_date | Statistics object |
| GET | `/analytics/trends` | Get detection trends | Query: period | Trend data |
| GET | `/analytics/heatmap` | Get detection heatmap | Query: camera_id, date | Heatmap data |

#### Verification Endpoints

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/verification-stats` | Get verification statistics | None | {pending, verified, rejected} |
| GET | `/verifications/pending` | Get pending verifications | Query: limit | Array of detections |

#### Live Streaming Endpoints

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/stream/{camera_id}` | Get camera video stream | Path: camera_id | Video stream (MJPEG) |
| WS | `/ws/live/{camera_id}` | WebSocket live feed | Path: camera_id | WebSocket frames |

### Appendix B: Database Schema

#### Detections Collection

```json
{
  "_id": "ObjectId",
  "timestamp": "ISODate",
  "camera_id": "string",
  "camera_name": "string",
  "detection_type": "string", // "gun", "knife", "violence", "person"
  "confidence": "number", // 0.0 to 1.0
  "bounding_box": {
    "x1": "number",
    "y1": "number",
    "x2": "number",
    "y2": "number"
  },
  "ipfs_cid": "string", // IPFS Content Identifier
  "image_path": "string", // Local backup path
  "verification_status": "string", // "pending", "verified", "false_positive"
  "verified_by": "string", // User ID
  "verified_at": "ISODate",
  "alert_sent": "boolean",
  "model_used": "string", // Model filename
  "metadata": {
    "people_count": "number",
    "location": {
      "latitude": "number",
      "longitude": "number"
    }
  }
}
```

#### Cameras Collection

```json
{
  "_id": "ObjectId",
  "name": "string",
  "stream_url": "string",
  "camera_type": "string", // "rgb", "thermal", "topview"
  "location": {
    "name": "string",
    "latitude": "number",
    "longitude": "number"
  },
  "detection_modes": ["string"], // ["gun", "knife", "violence", "people"]
  "status": "string", // "online", "offline", "error"
  "last_seen": "ISODate",
  "created_at": "ISODate",
  "settings": {
    "resolution": "string",
    "fps": "number",
    "alerts_enabled": "boolean"
  }
}
```

#### Users Collection

```json
{
  "_id": "ObjectId",
  "username": "string",
  "email": "string",
  "password_hash": "string",
  "role": "string", // "admin", "operator", "viewer"
  "created_at": "ISODate",
  "last_login": "ISODate",
  "phone": "string",
  "preferences": {
    "alert_notifications": "boolean",
    "email_notifications": "boolean"
  }
}
```

### Appendix C: Model Specifications

#### Gun Detection Models

| Model File | Purpose | Confidence Threshold | Classes |
|------------|---------|---------------------|---------|
| `thermal.pt` | Concealed weapon detection | 10% | Thermal gun signatures |
| `best.pt` | Primary gun/knife detection | 20% | Gun, knife |
| `best (1).pt` | Verification model | 30% | Gun, pistol |
| `best (8).pt` | High-confidence detection | 50% | Gun |

#### Other Models

| Model File | Purpose | Confidence Threshold | Classes |
|------------|---------|---------------------|---------|
| `yolov8m.pt` | People detection | 40% | Person |
| `best (9).pt` | Top-view detection | 30% | Person (aerial) |
| `thermalhuman.pt` | Thermal human detection | 40% | Person (thermal) |
| Violence model | Violence classification | Custom | Violence activities |

### Appendix D: Deployment Architecture

**Single-Server Deployment:**

```
┌─────────────────────────────────────────────┐
│         Server (Ubuntu 20.04 LTS)           │
│                                             │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │   Nginx      │      │   Frontend      │ │
│  │  (Reverse    │─────▶│   (React Build) │ │
│  │   Proxy)     │      └─────────────────┘ │
│  └──────────────┘                          │
│         │                                   │
│         │                                   │
│  ┌──────▼───────┐      ┌─────────────────┐ │
│  │   Backend    │─────▶│    MongoDB      │ │
│  │  (Python)    │      │   Database      │ │
│  └──────────────┘      └─────────────────┘ │
│         │                                   │
│         │                                   │
│  ┌──────▼───────┐      ┌─────────────────┐ │
│  │ IPFS Daemon  │      │  GPU (NVIDIA)   │ │
│  │              │      │  CUDA Toolkit   │ │
│  └──────────────┘      └─────────────────┘ │
└─────────────────────────────────────────────┘
         │                      │
         │                      │
    ┌────▼────┐            ┌────▼────┐
    │  IPFS   │            │ Camera  │
    │ Network │            │ Network │
    └─────────┘            └─────────┘
```

**Docker Deployment:**

```yaml
# docker-compose.yml structure
services:
  - backend (Python)
  - frontend (Nginx + React build)
  - mongodb
  - ipfs
```

### Appendix E: Alert Message Templates

**SMS Alert Template:**

```
🚨 SECURITY ALERT
Type: [GUN/KNIFE/VIOLENCE]
Camera: [Camera Name]
Location: [Location]
Time: [Timestamp]
Confidence: [XX%]
Evidence: ipfs://[CID]
Verify: [Dashboard URL]
```

**Voice Call Script:**

```
"Security alert. [Gun/Knife/Violence] detected at [Location] 
on camera [Camera Name] at [Time]. Please check the 
monitoring dashboard immediately for verification."
```

### Appendix F: System Requirements Summary

**Minimum System Requirements:**

- **CPU**: Intel Core i5 or AMD Ryzen 5 (4+ cores)
- **RAM**: 8 GB
- **GPU**: NVIDIA GTX 1060 (6GB VRAM) or equivalent
- **Storage**: 100 GB SSD
- **OS**: Ubuntu 20.04 LTS or macOS 11+
- **Network**: 100 Mbps internet connection
- **Python**: 3.8 or higher
- **Node.js**: 14.x or higher

**Recommended System Requirements:**

- **CPU**: Intel Core i7/i9 or AMD Ryzen 7/9 (8+ cores)
- **RAM**: 16-32 GB
- **GPU**: NVIDIA RTX 3060/3070 (12GB+ VRAM)
- **Storage**: 500 GB NVMe SSD
- **OS**: Ubuntu 22.04 LTS
- **Network**: 1 Gbps internet connection

**Software Dependencies:**

- Python packages: `requirements.txt`
- Node packages: `package.json`
- System: CUDA Toolkit 11.x, cuDNN 8.x

### Appendix G: Glossary of Technical Terms

- **Bounding Box**: Rectangular region identifying detected object in image
- **CID (Content Identifier)**: Unique cryptographic hash in IPFS identifying content
- **Confidence Score**: Probability (0-1) that detection is correct
- **Cooldown Period**: Minimum time between repeated alerts
- **False Positive**: Incorrect detection (alert when no threat exists)
- **Inference**: Process of running AI model on input data to get predictions
- **IPFS Gateway**: HTTP interface for accessing IPFS content
- **Pinning**: Keeping IPFS content permanently available
- **RTSP**: Real-Time Streaming Protocol for video streams
- **Thermal Imaging**: Infrared camera detecting heat signatures
- **Top-View**: Overhead/aerial camera perspective
- **YOLO**: "You Only Look Once" - Fast object detection algorithm

---

## Document Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | 2026-01-21 | Development Team | Initial SRS document creation |

---

## Approval Signatures

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Project Manager | _____________ | _____________ | _______ |
| Lead Developer | _____________ | _____________ | _______ |
| Quality Assurance | _____________ | _____________ | _______ |
| Stakeholder | _____________ | _____________ | _______ |

---

**END OF DOCUMENT**
