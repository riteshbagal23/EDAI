"""
Alert utilities for sending notifications via Twilio.

This module handles SMS and voice call alerts for security detections.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Optional, List
from twilio.rest import Client

logger = logging.getLogger(__name__)

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
EMERGENCY_CONTACTS = os.environ.get('EMERGENCY_CONTACTS', '').split(',')

# Initialize Twilio client
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("✅ Twilio Client initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize Twilio Client: {e}")


async def send_twilio_alert(detection_data: Dict) -> bool:
    """
    Send Twilio SMS alert and Voice Call for verified detections.
    
    Args:
        detection_data: Dict containing:
            - detection_type: Type of threat detected
            - confidence: Detection confidence (0-1)
            - camera_name: Name of camera
            - timestamp: Detection timestamp
            - location: Dict with 'lat' and 'lng'
    
    Returns:
        True if alert sent successfully, False otherwise
    """
    if not twilio_client:
        logger.debug("Twilio client not initialized, skipping alert")
        return False

    try:
        detection_type = detection_data.get('detection_type', 'Unknown Threat')
        confidence = detection_data.get('confidence', 0.0)
        camera_name = detection_data.get('camera_name', 'Unknown Camera')
        timestamp = detection_data.get('timestamp', datetime.now().isoformat())
        location = detection_data.get('location', {})
        
        # Construct Google Maps Link
        lat = location.get('lat', 0)
        lng = location.get('lng', 0)
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
        
        success_count = 0
        
        for contact in EMERGENCY_CONTACTS:
            contact = contact.strip()
            if not contact:
                continue
            
            # 1. Send SMS
            try:
                message = twilio_client.messages.create(
                    body=message_body,
                    from_=TWILIO_PHONE_NUMBER,
                    to=contact
                )
                logger.info(f"📨 Twilio SMS sent to {contact}: {message.sid}")
                success_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to send SMS to {contact}: {e}")
            
            # 2. Make Voice Call
            try:
                call = twilio_client.calls.create(
                    twiml=f'<Response><Say>Security Alert. {detection_type} detected at {camera_name}. Please check your messages for location details.</Say></Response>',
                    to=contact,
                    from_=TWILIO_PHONE_NUMBER
                )
                logger.info(f"📞 Twilio Call initiated to {contact}: {call.sid}")
            except Exception as e:
                logger.error(f"❌ Failed to initiate call to {contact}: {e}")
        
        return success_count > 0
            
    except Exception as e:
        logger.error(f"❌ Failed to process Twilio alert: {e}")
        return False


def get_emergency_contacts() -> List[str]:
    """Get list of configured emergency contacts."""
    return [c.strip() for c in EMERGENCY_CONTACTS if c.strip()]


def is_twilio_configured() -> bool:
    """Check if Twilio is properly configured."""
    return twilio_client is not None
