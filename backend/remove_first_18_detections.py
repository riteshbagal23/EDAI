#!/usr/bin/env python3
"""
Script to remove the first 18 detections from the database
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "SecureView"

async def remove_first_18_detections():
    """Remove the first 18 detections from the database"""
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
    
    try:
        # Get total count before deletion
        total_before = await db.detections.count_documents({})
        print(f"📊 Total detections before: {total_before}")
        
        # Find the first 18 detections sorted by timestamp
        detections = await db.detections.find({}).sort("timestamp", 1).limit(18).to_list(18)
        
        if not detections:
            print("ℹ️ No detections found in database")
            return
        
        print(f"\n🔍 Found {len(detections)} detections to remove:")
        for i, det in enumerate(detections, 1):
            det_type = det.get('detection_type', 'unknown')
            timestamp = det.get('timestamp', 'unknown')
            det_id = det.get('id', det.get('_id'))
            print(f"  {i}. {det_type} - {timestamp} (ID: {det_id})")
        
        # Delete these detections
        ids_to_delete = [det['_id'] for det in detections]
        result = await db.detections.delete_many({"_id": {"$in": ids_to_delete}})
        
        print(f"\n✅ Deleted {result.deleted_count} detections")
        
        # Get total count after deletion
        total_after = await db.detections.count_documents({})
        print(f"📊 Total detections after: {total_after}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(remove_first_18_detections())
