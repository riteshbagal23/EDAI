#!/usr/bin/env python3
"""
Upload existing admin verification images to IPFS
"""
import sys
import os
import asyncio
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from dotenv import load_dotenv

# Add backend to path
sys.path.append('/Users/ritesh/myproject/backend')
from ipfs_manager import IPFSManager

load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(MONGO_URI)
db = client.secureview_db

# Directories
DETECTIONS_DIR = Path("/Users/ritesh/myproject/backend/detections")

async def upload_verification_images():
    """Find admin verification images and upload to IPFS"""
    ipfs = IPFSManager()
    
    # Get all pending verifications from DB
    verifications = await db.pending_verifications.find({}).to_list(1000)
    
    print(f"📋 Found {len(verifications)} verification records")
    
    uploaded_count = 0
    skipped_count = 0
    
    for verification in verifications:
        # Check if already has IPFS data
        if 'ipfs_data' in verification and verification['ipfs_data']:
            print(f"⏭️  Skipping {verification['id']} - already has IPFS data")
            skipped_count += 1
            continue
        
        # Get the annotated image path
        annotated_path = verification.get('annotated_image_path')
        if not annotated_path:
            print(f"⚠️  No annotated image for {verification['id']}")
            continue
        
        # Convert to full path
        full_path = DETECTIONS_DIR / annotated_path.replace('/detections/', '')
        
        if not full_path.exists():
            print(f"❌ File not found: {full_path}")
            continue
        
        print(f"\n📤 Uploading: {full_path.name}")
        
        # Upload to IPFS
        try:
            result = ipfs.upload_to_ipfs(
                str(full_path),
                metadata={
                    "verification_id": verification['id'],
                    "detection_type": verification.get('detection_type', 'unknown'),
                    "camera": verification.get('camera_name', 'unknown'),
                    "timestamp": verification.get('timestamp', datetime.now().isoformat())
                }
            )
            
            if result and 'ipfs_hash' in result:
                # Update verification record with IPFS data
                await db.pending_verifications.update_one(
                    {"id": verification['id']},
                    {"$set": {"ipfs_data": result}}
                )
                
                print(f"✅ Uploaded: {result['ipfs_hash']}")
                print(f"🔗 URL: {result['url']}")
                uploaded_count += 1
            else:
                print(f"❌ Upload failed: {result}")
        
        except Exception as e:
            print(f"❌ Error uploading {full_path.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ Uploaded: {uploaded_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"📊 Total: {len(verifications)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(upload_verification_images())
