import os
from twilio.rest import Client
from dotenv import load_dotenv
from pathlib import Path

# Load env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
EMERGENCY_CONTACTS = os.environ.get('EMERGENCY_CONTACTS', '').split(',')

print(f"SID: {TWILIO_ACCOUNT_SID}")
print(f"Token: {'*' * 5 if TWILIO_AUTH_TOKEN else 'None'}")
print(f"From: {TWILIO_PHONE_NUMBER}")
print(f"Contacts: {EMERGENCY_CONTACTS}")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    print("❌ Missing credentials")
    exit(1)

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

for contact in EMERGENCY_CONTACTS:
    contact = contact.strip()
    if not contact:
        continue
    
    print(f"\nTesting contact: {contact}")
    
    # Test SMS
    try:
        print("Attempting to send SMS...")
        # Simulate a real alert message
        maps_link = "https://www.google.com/maps/search/?api=1&query=28.6139,77.2090" # Example: New Delhi
        message_body = (
            f"🚨 TEST ALERT 🚨\n"
            f"Type: WEAPON DETECTED\n"
            f"Camera: Test Camera\n"
            f"Location: {maps_link}\n"
            f"This is a test of the SecureView alert system."
        )
        
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=contact
        )
        print(f"✅ SMS Sent! SID: {message.sid}")
    except Exception as e:
        print(f"❌ SMS Failed: {e}")

    # Test Call
    try:
        print("Attempting to make Call...")
        call = client.calls.create(
            twiml='<Response><Say>This is a test security alert from Secure View. A weapon has been detected. Please check your text messages for location details.</Say></Response>',
            to=contact,
            from_=TWILIO_PHONE_NUMBER
        )
        print(f"✅ Call Initiated! SID: {call.sid}")
    except Exception as e:
        print(f"❌ Call Failed: {e}")
