import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Card, CardContent } from "@/components/ui/card";
import { Shield, AlertTriangle, MapPin, Navigation } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

// Fix Leaflet default icon
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require("leaflet/dist/images/marker-icon-2x.png"),
  iconUrl: require("leaflet/dist/images/marker-icon.png"),
  shadowUrl: require("leaflet/dist/images/marker-shadow.png"),
});

const createIcon = (svg, color) => new L.Icon({
  iconUrl: "data:image/svg+xml;base64," + btoa(svg),
  iconSize: [40, 40],
  iconAnchor: [20, 40],
  popupAnchor: [0, -40],
  className: "drop-shadow-lg"
});

const policeIcon = createIcon(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#3b82f6" stroke="white" stroke-width="2"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/><path d="M12 7a2 2 0 1 0 0 4 2 2 0 0 0 0-4zm0 6c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" fill="white"/></svg>`);
const detectionIcon = createIcon(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#ef4444" stroke="white" stroke-width="2"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>`);
const hospitalIcon = createIcon(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#06b6d4" stroke="white" stroke-width="2"><path d="M19 3H5c-1.1 0-1.99.9-1.99 2L3 19c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-1 11h-4v4h-4v-4H6v-4h4V6h4v4h4v4z"/></svg>`);
const fireIcon = createIcon(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#f59e0b" stroke="white" stroke-width="2"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zM7 9c0-2.76 2.24-5 5-5s5 2.24 5 5c0 2.88-2.88 7.19-5 9.88C9.92 16.21 7 11.85 7 9z"/><circle cx="12" cy="9" r="2.5" fill="white"/></svg>`);
const userIcon = createIcon(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#22c55e" stroke="white" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4" fill="white"/></svg>`);

const MapView = () => {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const [stats, setStats] = useState({ police: 0, detections: 0, hospitals: 0, fire: 0 });
  const [locationStatus, setLocationStatus] = useState("Locating...");

  useEffect(() => {
    // Initialize Map
    if (!mapInstanceRef.current && mapContainerRef.current) {
      const map = L.map(mapContainerRef.current).setView([18.5204, 73.8567], 13); // Default Pune

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      }).addTo(map);

      mapInstanceRef.current = map;
    }

    // Get User Location and Fetch Data
    const initMapData = async () => {
      if (!mapInstanceRef.current) return;

      let lat = 18.5204;
      let lng = 73.8567;

      // Try to get real location
      try {
        const position = await new Promise((resolve, reject) => {
          navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 5000 });
        });
        lat = position.coords.latitude;
        lng = position.coords.longitude;
        setLocationStatus("Location Found");

        // Update map view to user location
        mapInstanceRef.current.setView([lat, lng], 14);

        // Add user marker
        L.marker([lat, lng], { icon: userIcon })
          .bindPopup("<b>You are here</b>")
          .addTo(mapInstanceRef.current);

        L.circle([lat, lng], {
          color: '#22c55e',
          fillColor: '#22c55e',
          fillOpacity: 0.1,
          radius: 500
        }).addTo(mapInstanceRef.current);

      } catch (error) {
        console.warn("Location access denied or failed, using default:", error);
        setLocationStatus("Using Default Location");
      }

      // Fetch Data Function
      const updateData = async () => {
        try {
          const map = mapInstanceRef.current;
          if (!map) return;

          // Fetch Data First (to avoid flickering)
          const [detectionsRes, contextRes] = await Promise.all([
            axios.get(`${API}/detections`),
            axios.get(`${API}/emergency-context`, { params: { lat, lng } })
          ]);

          const detections = detectionsRes.data || [];
          const context = contextRes.data || {};
          const police = context.police_stations || [];
          const hospitals = context.hospitals || [];
          const fire = context.fire_stations || [];

          // Now clear and update markers
          markersRef.current.forEach(marker => marker.remove());
          markersRef.current = [];

          detections.forEach(d => {
            const marker = L.marker([d.location.lat, d.location.lng], { icon: detectionIcon })
              .bindPopup(`
                <div class="text-sm font-sans">
                  <strong class="text-red-600 capitalize">${d.detection_type} Detected</strong><br/>
                  Confidence: ${(d.confidence * 100).toFixed(0)}%<br/>
                  <span class="text-gray-500 text-xs">${new Date(d.timestamp).toLocaleTimeString()}</span>
                </div>
              `)
              .addTo(map);

            const circle = L.circle([d.location.lat, d.location.lng], {
              color: 'red',
              fillColor: '#f03',
              fillOpacity: 0.1,
              radius: 200
            }).addTo(map);

            markersRef.current.push(marker, circle);
          });

          police.forEach(s => {
            const distanceText = s.distance_km < 1
              ? `${s.distance_m}m away`
              : `${s.distance_km}km away`;

            const marker = L.marker([s.lat, s.lng], { icon: policeIcon })
              .bindPopup(`
                <div class="font-sans">
                  <div class="font-bold text-blue-600 mb-1">${s.name}</div>
                  <div class="text-xs text-gray-600 mb-2">
                    <span class="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded">📍 ${distanceText}</span>
                  </div>
                  <div class="text-xs text-gray-700 mb-1">📞 ${s.phone}</div>
                  <a 
                    href="https://www.google.com/maps/dir/?api=1&destination=${s.lat},${s.lng}" 
                    target="_blank" 
                    class="inline-block mt-2 bg-blue-500 text-white px-3 py-1 rounded text-xs font-semibold hover:bg-blue-600"
                  >
                    📍 Get Directions
                  </a>
                </div>
              `)
              .addTo(map);
            markersRef.current.push(marker);
          });

          hospitals.forEach(h => {
            const distanceText = h.distance_km < 1
              ? `${h.distance_m}m away`
              : `${h.distance_km}km away`;

            const marker = L.marker([h.lat, h.lng], { icon: hospitalIcon })
              .bindPopup(`
                <div class="font-sans">
                  <div class="font-bold text-cyan-600 mb-1">${h.name}</div>
                  <div class="text-xs text-gray-600 mb-2">
                    <span class="inline-block bg-cyan-100 text-cyan-800 px-2 py-1 rounded">📍 ${distanceText}</span>
                  </div>
                  <div class="text-xs text-gray-700 mb-1">📞 ${h.phone}</div>
                  <a 
                    href="https://www.google.com/maps/dir/?api=1&destination=${h.lat},${h.lng}" 
                    target="_blank" 
                    class="inline-block mt-2 bg-cyan-500 text-white px-3 py-1 rounded text-xs font-semibold hover:bg-cyan-600"
                  >
                    📍 Get Directions
                  </a>
                </div>
              `)
              .addTo(map);
            markersRef.current.push(marker);
          });

          fire.forEach(f => {
            const distanceText = f.distance_km < 1
              ? `${f.distance_m}m away`
              : `${f.distance_km}km away`;

            const marker = L.marker([f.lat, f.lng], { icon: fireIcon })
              .bindPopup(`
                <div class="font-sans">
                  <div class="font-bold text-amber-600 mb-1">${f.name}</div>
                  <div class="text-xs text-gray-600 mb-2">
                    <span class="inline-block bg-amber-100 text-amber-800 px-2 py-1 rounded">📍 ${distanceText}</span>
                  </div>
                  <div class="text-xs text-gray-700 mb-1">📞 ${f.phone}</div>
                  <a 
                    href="https://www.google.com/maps/dir/?api=1&destination=${f.lat},${f.lng}" 
                    target="_blank" 
                    class="inline-block mt-2 bg-amber-500 text-white px-3 py-1 rounded text-xs font-semibold hover:bg-amber-600"
                  >
                    📍 Get Directions
                  </a>
                </div>
              `)
              .addTo(map);
            markersRef.current.push(marker);
          });

          setStats({
            police: police.length,
            detections: detections.length,
            hospitals: hospitals.length,
            fire: fire.length
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

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
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
          <div ref={mapContainerRef} style={{ height: "600px", width: "100%" }} />
        </CardContent>
      </Card>
    </div>
  );
};

export default MapView;