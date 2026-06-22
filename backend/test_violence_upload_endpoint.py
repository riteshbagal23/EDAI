import requests
import cv2
import numpy as np
import os

# Configuration
API_URL = "http://localhost:8000/api/test-upload"
IMAGE_PATH = "dummy_violence.jpg"

def create_dummy_image():
    # Create a black image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Add some text
    cv2.putText(img, "TEST VIOLENCE", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imwrite(IMAGE_PATH, img)
    print(f"Created dummy image at {IMAGE_PATH}")

def test_upload():
    if not os.path.exists(IMAGE_PATH):
        create_dummy_image()

    print(f"Sending request to {API_URL}...")
    
    with open(IMAGE_PATH, 'rb') as f:
        files = {'file': f}
        data = {'model_type': 'violence', 'camera_id': 'test_script'}
        
        try:
            response = requests.post(API_URL, files=files, data=data)
            
            print(f"Status Code: {response.status_code}")
            print("Response JSON:")
            print(response.json())
            
            if response.status_code == 200:
                print("\n✅ Test Passed!")
            else:
                print("\n❌ Test Failed!")
                
        except requests.exceptions.ConnectionError:
            print("\n❌ Connection Error: Is the server running on localhost:8000?")
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_upload()
