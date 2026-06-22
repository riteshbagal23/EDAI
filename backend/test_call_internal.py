import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load env
load_dotenv()

# Import the actual function
try:
    from utils.alerts import send_twilio_alert
    print("✅ Successfully imported send_twilio_alert")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

def test_internal_call():
    print("🧪 Testing send_twilio_alert function directly...")
    
    # Mock data
    test_data = {
        'detection_type': 'TEST_INTERNAL_CALL',
        'confidence': 0.99,
        'camera_name': 'Debug Script',
        'timestamp': datetime.now().isoformat(),
        'location': {'lat': 0, 'lng': 0}
    }
    
    try:
        result = send_twilio_alert(test_data)
        if result:
            print("✅ Function returned True (Success)")
        else:
            print("❌ Function returned False (Failure)")
            
    except Exception as e:
        print(f"❌ Exception during execution: {e}")

if __name__ == "__main__":
    test_internal_call()
