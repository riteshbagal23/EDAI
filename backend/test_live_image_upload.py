#!/usr/bin/env python3
import sys
sys.path.append('/Users/ritesh/myproject/backend')

from ipfs_manager import IPFSManager

# Test uploading an existing live detection image
test_file = "/Users/ritesh/myproject/backend/detections/live_violence_fbcc8963-8597-4100-bcc0-ca59a363ce19.jpg"

print(f"Testing upload of: {test_file}")

ipfs = IPFSManager()
result = ipfs.upload_to_ipfs(test_file, metadata={"test": "direct_upload"})

if result and 'ipfs_hash' in result:
    print(f"\n✅ Upload successful!")
    print(f"IPFS Hash: {result['ipfs_hash']}")
    print(f"URL: {result['url']}")
    print(f"\n🔗 Test this link in your browser:")
    print(f"   {result['url']}")
else:
    print(f"\n❌ Upload failed: {result}")
