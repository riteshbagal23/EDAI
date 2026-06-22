#!/bin/bash
SID="YOUR_TWILIO_SID"
TOKEN="YOUR_TWILIO_TOKEN"
FROM="YOUR_TWILIO_PHONE"
TO="YOUR_RECIPIENT_PHONE"

# TwiML for voice message
TWIML="<Response><Say>This is a test call from Secure View Debugger.</Say></Response>"

echo "Testing Twilio Voice Call..."
curl -X POST https://api.twilio.com/2010-04-01/Accounts/$SID/Calls.json \
--data-urlencode "Twiml=$TWIML" \
--data-urlencode "From=$FROM" \
--data-urlencode "To=$TO" \
-u $SID:$TOKEN
