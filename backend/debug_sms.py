import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from twilio.rest import Client

# Load env
load_dotenv()

def test_sms_content():
    print("🧪 Testing Twilio SMS with full content...")
    
    sid = os.environ.get('TWILIO_ACCOUNT_SID')
    token = os.environ.get('TWILIO_AUTH_TOKEN')
    phone = os.environ.get('TWILIO_PHONE_NUMBER')
    contacts = os.environ.get('EMERGENCY_CONTACTS', '').split(',')
    
    if not (sid and token and phone and contacts):
        print("❌ Missing configuration")
        return

    client = Client(sid, token)
    
    # Simulate the exact message body from the app
    detection_type = "TEST_GUN"
    confidence = 0.95
    camera_name = "Test Camera"
    timestamp = datetime.now().isoformat()
    lat = 0
    lng = 0
    maps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"
    
    message_body = (
        f"🚨 SECURITY ALERT 🚨\n"
        f"Type: {detection_type.upper()}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Camera: {camera_name}\n"
        f"Time: {timestamp}\n"
        f"Location: {maps_link}\n"
        f"Please verify immediately."
    )
    
    print(f"📝 Message Body:\n{message_body}\n")
    
    for contact in contacts:
        contact = contact.strip()
        if not contact: continue
        
        print(f"📤 Sending to {contact}...")
        try:
            message = client.messages.create(
                body=message_body,
                from_=phone,
                to=contact
            )
            print(f"✅ SMS queued! SID: {message.sid}")
            print(f"   Status: {message.status}")
            print(f"   Error Code: {message.error_code}")
            print(f"   Error Message: {message.error_message}")
        except Exception as e:
            print(f"❌ Failed to send SMS: {e}")

if __name__ == "__main__":
    test_sms_content()
