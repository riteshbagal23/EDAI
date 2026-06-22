#!/bin/bash
SID="YOUR_TWILIO_SID"
TOKEN="YOUR_TWILIO_TOKEN"
FROM="YOUR_TWILIO_PHONE"
TO="YOUR_RECIPIENT_PHONE"

# URL encoded body with emojis and link
BODY="🚨 SECURITY ALERT 🚨%0AType: TEST_GUN%0AConfidence: 95.0%%0ACamera: Test Camera%0ATime: 2025-01-01T12:00:00%0ALocation: https://www.google.com/maps/search/?api=1&query=0,0%0APlease verify immediately."

echo "Testing Twilio SMS with complex content..."
curl -X POST https://api.twilio.com/2010-04-01/Accounts/$SID/Messages.json \
--data-urlencode "Body=$BODY" \
--data-urlencode "From=$FROM" \
--data-urlencode "To=$TO" \
-u $SID:$TOKEN
