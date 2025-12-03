#!/usr/bin/env python3
"""
Upload the most recent dual verification images to IPFS
"""
import sys
import os
from pathlib import Path
from glob import glob

sys.path.append('/Users/ritesh/myproject/backend')
from ipfs_manager import IPFSManager

DETECTIONS_DIR = Path("/Users/ritesh/myproject/backend/detections")

def upload_recent_dual_images():
    """Upload recent dual_annotated images to IPFS"""
    ipfs = IPFSManager()
    
    # Find all dual_annotated images
    pattern = str(DETECTIONS_DIR / "dual_annotated_*.jpg")
    files = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
    
    print(f"📋 Found {len(files)} dual verification images")
    
    # Upload the 5 most recent
    for i, file_path in enumerate(files[:5], 1):
        file_name = Path(file_path).name
        print(f"\n[{i}/5] 📤 Uploading: {file_name}")
        
        try:
            result = ipfs.upload_to_ipfs(
                file_path,
                metadata={
                    "type": "admin_verification",
                    "detection_type": "pistol",
                    "source": "dual_model_verification"
                }
            )
            
            if result and 'ipfs_hash' in result:
                print(f"✅ IPFS Hash: {result['ipfs_hash']}")
                print(f"🔗 URL: {result['url']}")
                print(f"🔐 SHA256: {result['sha256_hash']}")
            else:
                print(f"❌ Upload failed: {result}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ Uploaded 5 most recent dual verification images to IPFS")
    print(f"{'='*60}")

if __name__ == "__main__":
    upload_recent_dual_images()
