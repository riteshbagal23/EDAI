from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import threading
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize models
logger.info("Loading weapon detection model...")
weapon_model = YOLO('best.pt')  # Gun and knife detection model
logger.info("Loading people detection model...")
people_model = YOLO('yolov8m.pt')  # People detection model

# Global variables
camera = None
is_running = False
detection_stats = {
    'guns': 0,
    'knives': 0,
    'people': 0,
    'alert': False,
    'alert_message': '',
    'last_alert_time': None
}
frame_lock = threading.Lock()
current_frame = None
alert_cooldown = 30  # seconds between alerts


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
    def __del__(self):
        if self.video.isOpened():
            self.video.release()
    
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        return frame


def detect_weapons_and_people(frame):
    """Detect weapons and count people in frame"""
    global detection_stats
    
    guns_detected = 0
    knives_detected = 0
    people_detected = 0
    
    # Detect weapons (guns and knives) with moderate threshold
    weapon_results = weapon_model(frame, conf=0.40, verbose=False)

    # Process weapon detections (guns / knives)
    for result in weapon_results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class name
            class_name = weapon_model.names[cls].lower()

            # Draw bounding box for weapons
            if 'pistol' in class_name or 'gun' in class_name:
                guns_detected += 1
                color = (0, 0, 255)  # Red for guns
                label = f'GUN {conf:.2f}'
                logger.info(f"🔴 GUN DETECTED with confidence {conf:.2f}")
            elif 'knife' in class_name:
                knives_detected += 1
                color = (0, 165, 255)  # Orange for knives
                label = f'KNIFE {conf:.2f}'
                logger.info(f"🟠 KNIFE DETECTED with confidence {conf:.2f}")
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Always run people detection (count only, no boxes)
    people_results = people_model(frame, conf=0.5, verbose=False)
    for result in people_results:
        boxes = result.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            cls = int(box.cls[0])
            class_name = people_model.names[cls].lower()

            # Only count 'person' class - NO BOXES DRAWN
            if class_name == 'person':
                people_detected += 1
                # Removed: bounding box drawing for people
    
    # Update detection stats
    detection_stats['guns'] = guns_detected
    detection_stats['knives'] = knives_detected
    detection_stats['people'] = people_detected
    
    # Trigger alert if weapon detected
    if guns_detected > 0 or knives_detected > 0:
        trigger_alert(guns_detected, knives_detected, people_detected)
    else:
        detection_stats['alert'] = False
        detection_stats['alert_message'] = ''
    
    # Add stats overlay to frame
    overlay_height = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, overlay_height), (255, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, f'Guns: {guns_detected}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f'Knives: {knives_detected}', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    cv2.putText(frame, f'People: {people_detected}', (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, (frame.shape[1] - 350, frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame


def trigger_alert(guns, knives, people):
    """Trigger alert if weapon detected"""
    global detection_stats
    
    current_time = time.time()
    
    # Check if we're in cooldown period
    if detection_stats['last_alert_time'] is not None:
        if current_time - detection_stats['last_alert_time'] < alert_cooldown:
            return
    
    # Create alert message
    weapons = []
    if guns > 0:
        weapons.append(f"{guns} gun(s)")
    if knives > 0:
        weapons.append(f"{knives} knife(s)")
    
    alert_message = f"THREAT DETECTED: {' and '.join(weapons)} detected"
    if people > 0:
        alert_message += f" with {people} person(s) in frame"
    
    detection_stats['alert'] = True
    detection_stats['alert_message'] = alert_message
    detection_stats['last_alert_time'] = current_time
    
    logger.warning(f"⚠️ ALERT: {alert_message}")


def generate_frames():
    """Generate frames for video streaming"""
    global current_frame, is_running
    
    while is_running:
        if camera is None:
            time.sleep(0.1)
            continue
            
        frame = camera.get_frame()
        if frame is None:
            continue
        
        # Perform detection
        frame = detect_weapons_and_people(frame)
        
        # Store current frame
        with frame_lock:
            current_frame = frame.copy()
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/start', methods=['POST'])
def start():
    """Start the detection system"""
    global camera, is_running
    
    if not is_running:
        is_running = True
        camera = VideoCamera()
        logger.info("✅ Detection system started")
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})


@app.route('/stop', methods=['POST'])
def stop():
    """Stop the detection system"""
    global camera, is_running
    
    if is_running:
        is_running = False
        if camera is not None:
            del camera
            camera = None
        
        # Reset stats
        detection_stats.update({
            'guns': 0,
            'knives': 0,
            'people': 0,
            'alert': False,
            'alert_message': ''
        })
        logger.info("⏹️ Detection system stopped")
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not_running'})


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    """Get current detection status"""
    return jsonify(detection_stats)


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload for detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read file into numpy array
        import io
        file_bytes = io.BytesIO(file.read())
        nparr = np.frombuffer(file_bytes.getvalue(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        logger.info(f"📤 Processing uploaded file: {file.filename}")
        
        # Perform detection on uploaded file
        frame = detect_weapons_and_people(frame)
        
        # Return detection results
        return jsonify({
            'status': 'ok',
            'detections': {
                'guns': detection_stats['guns'],
                'knives': detection_stats['knives'],
                'people': detection_stats['people']
            }
        })
    except Exception as e:
        logger.error(f"❌ Error in upload: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os
    os.makedirs('templates', exist_ok=True)
    
    print("="*60)
    print("🔒 SECURITY DETECTION SYSTEM")
    print("="*60)
    print("Models loaded:")
    print("  ✅ Weapon Detection: best.pt")
    print("  ✅ People Detection: yolov8m.pt")
    print("="*60)
    print("🌐 Open http://localhost:5002 in your browser")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
