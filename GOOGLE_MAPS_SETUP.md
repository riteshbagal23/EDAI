# Google Maps API Integration Guide

## Overview
This guide explains how to integrate Google Maps API into your SecureView Alert application to display hospitals and police stations. You can use your **Gemini API key** with Google Maps API since they're both part of the Google Cloud Platform.

## Prerequisites
- Google Cloud Platform account
- Gemini API key (can be used for Google Maps)

---

## Step 1: Enable Google Maps JavaScript API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (the same one where you have your Gemini API key)
3. Navigate to **APIs & Services** > **Library**
4. Search for "Maps JavaScript API"
5. Click **Enable**

> **Note**: Your existing Gemini API key should work for Google Maps if both APIs are enabled in the same project.

---

## Step 2: Install Required Packages

```bash
cd frontend
npm install @react-google-maps/api
```

---

## Step 3: Add API Key to Environment

Add your Google Maps API key to `frontend/.env`:

```bash
REACT_APP_GOOGLE_MAPS_API_KEY=your_api_key_here
REACT_APP_BACKEND_URL=http://localhost:8000
```

> **Important**: Replace `your_api_key_here` with your actual API key

---

## Step 4: Update MapView Component

Here's the updated `MapView.js` using Google Maps:

```javascript
import React, { useEffect, useState, useCallback } from "react";
import axios from "axios";
import { GoogleMap, LoadScript, Marker, Circle, InfoWindow } from "@react-google-maps/api";
import { Card, CardContent } from "@/components/ui/card";
import { Shield, AlertTriangle, MapPin, Navigation } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;
const GOOGLE_MAPS_API_KEY = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;

const mapContainerStyle = {
  width: "100%",
  height: "600px",
};

const defaultCenter = {
  lat: 18.5204,
  lng: 73.8567,
};

const MapView = () => {
  const [map, setMap] = useState(null);
  const [center, setCenter] = useState(defaultCenter);
  const [stats, setStats] = useState({ police: 0, detections: 0, hospitals: 0, fire: 0 });
  const [locationStatus, setLocationStatus] = useState("Locating...");
  const [markers, setMarkers] = useState({
    police: [],
    hospitals: [],
    fire: [],
    detections: [],
  });
  const [selectedMarker, setSelectedMarker] = useState(null);

  const onLoad = useCallback((map) => {
    setMap(map);
  }, []);

  useEffect(() => {
    const initMapData = async () => {
      let lat = 18.5204;
      let lng = 73.8567;

      // Get user location
      try {
        const position = await new Promise((resolve, reject) => {
          navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 5000 });
        });
        lat = position.coords.latitude;
        lng = position.coords.longitude;
        setCenter({ lat, lng });
        setLocationStatus("Location Found");
      } catch (error) {
        console.warn("Location access denied, using default:", error);
        setLocationStatus("Using Default Location");
      }

      // Fetch emergency services and detections
      const updateData = async () => {
        try {
          const [detectionsRes, contextRes] = await Promise.all([
            axios.get(`${API}/detections`),
            axios.get(`${API}/emergency-context`, { params: { lat, lng } }),
          ]);

          const detections = detectionsRes.data || [];
          const context = contextRes.data || {};

          setMarkers({
            police: context.police_stations || [],
            hospitals: context.hospitals || [],
            fire: context.fire_stations || [],
            detections: detections,
          });

          setStats({
            police: (context.police_stations || []).length,
            detections: detections.length,
            hospitals: (context.hospitals || []).length,
            fire: (context.fire_stations || []).length,
          });
        } catch (err) {
          console.error("Map data fetch error:", err);
        }
      };

      updateData();
      const interval = setInterval(updateData, 5000);
      return () => clearInterval(interval);
    };

    initMapData();
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
            Map View
          </h1>
          <div className="flex items-center gap-2 mt-2">
            <Navigation className="w-4 h-4 text-green-600" />
            <p className="text-gray-600 text-sm">{locationStatus}</p>
          </div>
        </div>
      </div>

      <Card className="bg-white border-gray-200 shadow-sm">
        <CardContent className="p-4">
          <div className="flex flex-wrap gap-6 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
              <span className="text-gray-700">Police ({stats.police})</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-cyan-500 rounded-full"></div>
              <span className="text-gray-700">Hospitals ({stats.hospitals})</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-amber-500 rounded-full"></div>
              <span className="text-gray-700">Fire ({stats.fire})</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-red-500 rounded-full"></div>
              <span className="text-gray-700">Detections ({stats.detections})</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-white border-gray-200 shadow-sm overflow-hidden">
        <CardContent className="p-0">
          <LoadScript googleMapsApiKey={GOOGLE_MAPS_API_KEY}>
            <GoogleMap
              mapContainerStyle={mapContainerStyle}
              center={center}
              zoom={14}
              onLoad={onLoad}
            >
              {/* User Location */}
              <Marker
                position={center}
                icon={{
                  path: window.google.maps.SymbolPath.CIRCLE,
                  scale: 10,
                  fillColor: "#22c55e",
                  fillOpacity: 1,
                  strokeColor: "#fff",
                  strokeWeight: 2,
                }}
              />
              <Circle
                center={center}
                radius={500}
                options={{
                  fillColor: "#22c55e",
                  fillOpacity: 0.1,
                  strokeColor: "#22c55e",
                  strokeWeight: 1,
                }}
              />

              {/* Police Stations */}
              {markers.police.map((station, idx) => (
                <Marker
                  key={`police-${idx}`}
                  position={{ lat: station.lat, lng: station.lng }}
                  icon={{
                    url: "data:image/svg+xml;base64," + btoa(`
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="40" height="40">
                        <path fill="#3b82f6" d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>
                      </svg>
                    `),
                  }}
                  onClick={() => setSelectedMarker({ type: "police", data: station })}
                />
              ))}

              {/* Hospitals */}
              {markers.hospitals.map((hospital, idx) => (
                <Marker
                  key={`hospital-${idx}`}
                  position={{ lat: hospital.lat, lng: hospital.lng }}
                  icon={{
                    url: "data:image/svg+xml;base64," + btoa(`
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="40" height="40">
                        <path fill="#06b6d4" d="M19 3H5c-1.1 0-1.99.9-1.99 2L3 19c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-1 11h-4v4h-4v-4H6v-4h4V6h4v4h4v4z"/>
                      </svg>
                    `),
                  }}
                  onClick={() => setSelectedMarker({ type: "hospital", data: hospital })}
                />
              ))}

              {/* Fire Stations */}
              {markers.fire.map((station, idx) => (
                <Marker
                  key={`fire-${idx}`}
                  position={{ lat: station.lat, lng: station.lng }}
                  icon={{
                    url: "data:image/svg+xml;base64," + btoa(`
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="40" height="40">
                        <path fill="#f59e0b" d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/>
                      </svg>
                    `),
                  }}
                  onClick={() => setSelectedMarker({ type: "fire", data: station })}
                />
              ))}

              {/* Detections */}
              {markers.detections.map((detection, idx) => (
                <React.Fragment key={`detection-${idx}`}>
                  <Marker
                    position={{ lat: detection.location.lat, lng: detection.location.lng }}
                    icon={{
                      url: "data:image/svg+xml;base64," + btoa(`
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="40" height="40">
                          <path fill="#ef4444" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                        </svg>
                      `),
                    }}
                    onClick={() => setSelectedMarker({ type: "detection", data: detection })}
                  />
                  <Circle
                    center={{ lat: detection.location.lat, lng: detection.location.lng }}
                    radius={200}
                    options={{
                      fillColor: "#ef4444",
                      fillOpacity: 0.1,
                      strokeColor: "#ef4444",
                      strokeWeight: 1,
                    }}
                  />
                </React.Fragment>
              ))}

              {/* Info Window */}
              {selectedMarker && (
                <InfoWindow
                  position={
                    selectedMarker.type === "detection"
                      ? { lat: selectedMarker.data.location.lat, lng: selectedMarker.data.location.lng }
                      : { lat: selectedMarker.data.lat, lng: selectedMarker.data.lng }
                  }
                  onCloseClick={() => setSelectedMarker(null)}
                >
                  <div className="p-2">
                    {selectedMarker.type === "detection" ? (
                      <>
                        <h3 className="font-bold text-red-600 capitalize">
                          {selectedMarker.data.detection_type} Detected
                        </h3>
                        <p>Confidence: {(selectedMarker.data.confidence * 100).toFixed(0)}%</p>
                        <p className="text-xs text-gray-500">
                          {new Date(selectedMarker.data.timestamp).toLocaleTimeString()}
                        </p>
                      </>
                    ) : (
                      <>
                        <h3 className="font-bold">{selectedMarker.data.name}</h3>
                        <p className="text-sm">📞 {selectedMarker.data.phone}</p>
                        <p className="text-xs text-gray-600">
                          {selectedMarker.data.distance_km < 1
                            ? `${selectedMarker.data.distance_m}m away`
                            : `${selectedMarker.data.distance_km}km away`}
                        </p>
                        <a
                          href={`https://www.google.com/maps/dir/?api=1&destination=${selectedMarker.data.lat},${selectedMarker.data.lng}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-block mt-2 bg-blue-500 text-white px-3 py-1 rounded text-xs font-semibold hover:bg-blue-600"
                        >
                          📍 Get Directions
                        </a>
                      </>
                    )}
                  </div>
                </InfoWindow>
              )}
            </GoogleMap>
          </LoadScript>
        </CardContent>
      </Card>
    </div>
  );
};

export default MapView;
```

---

## Step 5: API Key Security (Important!)

### For Development:
The `.env` file is fine for local development.

### For Production:
1. **Restrict your API key** in Google Cloud Console:
   - Go to **APIs & Services** > **Credentials**
   - Click on your API key
   - Under "Application restrictions", select "HTTP referrers"
   - Add your domain (e.g., `yourdomain.com/*`)

2. **Enable only required APIs**:
   - Maps JavaScript API
   - Places API (if using place search)
   - Directions API (for routing)

---

## Alternative: Keep Using OpenStreetMap (Free)

If you prefer to keep using the current free OpenStreetMap solution (Leaflet), you don't need a Google Maps API key at all. The current implementation already works well and displays hospitals and police stations.

**Current setup advantages:**
- ✅ Completely free
- ✅ No API key required
- ✅ Already working
- ✅ Shows all emergency services with directions

**Google Maps advantages:**
- ✅ Better satellite imagery
- ✅ Street View integration
- ✅ More detailed POI data
- ✅ Better mobile experience

---

## Troubleshooting

### "This page can't load Google Maps correctly"
- Check if your API key is correct in `.env`
- Verify Maps JavaScript API is enabled in Google Cloud Console
- Check browser console for specific error messages

### Markers not showing
- Verify backend is returning correct lat/lng data
- Check browser console for errors
- Ensure API responses have the correct structure

### Directions not working
- Make sure the link format is correct
- Test the Google Maps directions URL manually

---

## Next Steps

1. Add your Google Maps API key to `frontend/.env`
2. Install the required package: `npm install @react-google-maps/api`
3. Replace the current `MapView.js` with the Google Maps version above
4. Restart your frontend server

**Or** keep the current OpenStreetMap implementation - it's already working great!
