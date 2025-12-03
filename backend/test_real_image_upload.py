import sys
sys.path.append('/Users/ritesh/myproject/backend')

from ipfs_manager import IPFSManager
import os

# Test with an actual detection image
test_image = "/Users/ritesh/myproject/backend/detections/violence_e22ad11b-77d1-4631-9c6c-62ccc470ddb9.jpg"

if not os.path.exists(test_image):
    print(f"❌ Test image not found: {test_image}")
    exit(1)

print(f"📸 Testing IPFS upload with: {test_image}")
print(f"File size: {os.path.getsize(test_image)} bytes")

ipfs = IPFSManager()
result = ipfs.upload_to_ipfs(test_image, metadata={"type": "violence_detection", "test": True})

if result and 'ipfs_hash' in result:
    print("\n✅ UPLOAD SUCCESSFUL!")
    print(f"IPFS Hash: {result['ipfs_hash']}")
    print(f"SHA256: {result['sha256_hash']}")
    print(f"URL: {result['url']}")
    print(f"\n🌐 View image at: {result['url']}")
else:
    print(f"\n❌ UPLOAD FAILED: {result}")
