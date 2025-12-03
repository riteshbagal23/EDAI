#!/usr/bin/env python3
"""
Cleanup script to remove incomplete detection records from MongoDB
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

async def cleanup_detections():
    """Remove detection records missing required fields"""
    
    # Connect to MongoDB
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
    client = AsyncIOMotorClient(mongo_uri)
    db = client.weapon_detection
    
    print("🔍 Checking for incomplete detection records...")
    
    # Required fields for valid detections
    required_fields = ['camera_id', 'camera_name', 'detection_type', 'confidence', 'image_path', 'location']
    
    # Find records missing any required field
    query = {
        "$or": [
            {field: {"$exists": False}} for field in required_fields
        ] + [
            {field: None} for field in required_fields
        ]
    }
    
    # Count incomplete records
    count = await db.detections.count_documents(query)
    print(f"📊 Found {count} incomplete detection record(s)")
    
    if count > 0:
        # Show some examples
        cursor = db.detections.find(query).limit(5)
        print("\n📋 Sample incomplete records:")
        async for doc in cursor:
            print(f"  - ID: {doc.get('id', 'unknown')}, Missing fields:", end=" ")
            missing = [field for field in required_fields if field not in doc or doc.get(field) is None]
            print(", ".join(missing))
        
        # Ask for confirmation
        print(f"\n⚠️  About to delete {count} incomplete record(s)...")
        confirm = input("Continue? (yes/no): ")
        
        if confirm.lower() in ['yes', 'y']:
            result = await db.detections.delete_many(query)
            print(f"✅ Deleted {result.deleted_count} incomplete detection record(s)")
        else:
            print("❌ Cleanup cancelled")
    else:
        print("✅ No incomplete records found! Database is clean.")
    
    # Show final count
    total = await db.detections.count_documents({})
    print(f"\n📊 Total detection records remaining: {total}")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(cleanup_detections())
