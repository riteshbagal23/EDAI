import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Card, CardContent } from "@/components/ui/card";
import { Shield, AlertTriangle, MapPin, Navigation, Radio } from "lucide-react";

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
const multispecialtyIcon = createIcon(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#a855f7" stroke="white" stroke-width="2"><path d="M19 3H5c-1.1 0-1.99.9-1.99 2L3 19c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-1 11h-4v4h-4v-4H6v-4h4V6h4v4h4v4z"/><circle cx="12" cy="12" r="3" fill="white"/><path d="M12 8v8M8 12h8" stroke="#a855f7" stroke-width="1.5"/></svg>`);
const fireIcon = createIcon(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#f59e0b" stroke="white" stroke-width="2"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zM7 9c0-2.76 2.24-5 5-5s5 2.24 5 5c0 2.88-2.88 7.19-5 9.88C9.92 16.21 7 11.85 7 9z"/><circle cx="12" cy="9" r="2.5" fill="white"/></svg>`);
const userIcon = createIcon(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#22c55e" stroke="white" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4" fill="white"/></svg>`);

const MapView = () => {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const [stats, setStats] = useState({ police: 0, detections: 0, hospitals: 0, multispecialty: 0, fire: 0 });
  const [locationStatus, setLocationStatus] = useState("Locating...");
  const [radius, setRadius] = useState(10); // Default 10km radius
  const [totalFound, setTotalFound] = useState({ police: 0, hospitals: 0, multispecialty: 0, fire: 0 });
  const [userLocation, setUserLocation] = useState({ lat: 18.5204, lng: 73.8567 });

  // Initialize Map (runs once on mount)
  useEffect(() => {
    if (!mapInstanceRef.current && mapContainerRef.current) {
      const map = L.map(mapContainerRef.current).setView([18.5204, 73.8567], 13); // Default Pune

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      }).addTo(map);

      mapInstanceRef.current = map;

      // Get User Location
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const lat = position.coords.latitude;
          const lng = position.coords.longitude;
          setUserLocation({ lat, lng });
          setLocationStatus("Location Found");

          // Update map view to user location
          map.setView([lat, lng], 14);

          // Add user marker
          L.marker([lat, lng], { icon: userIcon })
            .bindPopup("<b>You are here</b>")
            .addTo(map);

          L.circle([lat, lng], {
            color: '#22c55e',
            fillColor: '#22c55e',
            fillOpacity: 0.1,
            radius: 500
          }).addTo(map);
        },
        (error) => {
          console.warn("Location access denied or failed, using default:", error);
          setLocationStatus("Using Default Location");
        },
        { timeout: 5000 }
      );
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []); // Run only once on mount

  // Fetch and update data when radius or userLocation changes
  useEffect(() => {
    const updateData = async () => {
      try {
        const map = mapInstanceRef.current;
        if (!map) return;

        const { lat, lng } = userLocation;

        // Fetch Data First (to avoid flickering)
        const [detectionsRes, contextRes] = await Promise.all([
          axios.get(`${API}/detections`),
          axios.get(`${API}/emergency-context`, { params: { lat, lng, radius } })
        ]);

        const detections = detectionsRes.data || [];
        const context = contextRes.data || {};
        const police = context.police_stations || [];
        const hospitals = context.hospitals || [];
        const multispecialtyHospitals = context.multispecialty_hospitals || [];
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
                  <span class="inline-block bg-cyan-100 text-cyan-800 px-2 py-1 rounded">🏥 General Hospital</span>
                  <span class="inline-block bg-cyan-100 text-cyan-800 px-2 py-1 rounded ml-1">📍 ${distanceText}</span>
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

        multispecialtyHospitals.forEach(m => {
          const distanceText = m.distance_km < 1
            ? `${m.distance_m}m away`
            : `${m.distance_km}km away`;

          const specialtiesList = m.specialties ? m.specialties.join(', ') : 'Multiple Specialties';

          const marker = L.marker([m.lat, m.lng], { icon: multispecialtyIcon })
            .bindPopup(`
              <div class="font-sans">
                <div class="font-bold text-purple-600 mb-1">${m.name}</div>
                <div class="text-xs text-gray-600 mb-2">
                  <span class="inline-block bg-purple-100 text-purple-800 px-2 py-1 rounded">⭐ Multi-Specialty</span>
                  <span class="inline-block bg-purple-100 text-purple-800 px-2 py-1 rounded ml-1">📍 ${distanceText}</span>
                </div>
                <div class="text-xs text-gray-700 mb-2">📞 ${m.phone}</div>
                <div class="text-xs text-gray-600 mb-2">
                  <strong>Specialties:</strong> ${specialtiesList}
                </div>
                <a 
                  href="https://www.google.com/maps/dir/?api=1&destination=${m.lat},${m.lng}" 
                  target="_blank" 
                  class="inline-block mt-2 bg-purple-500 text-white px-3 py-1 rounded text-xs font-semibold hover:bg-purple-600"
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
          multispecialty: multispecialtyHospitals.length,
          fire: fire.length
        });

        // Update total found counts
        if (context.total_found) {
          setTotalFound(context.total_found);
        }

      } catch (err) {
        console.error("Map data fetch error:", err);
      }
    };

    updateData();
    const intervalRef = setInterval(updateData, 5000);
    return () => clearInterval(intervalRef);
  }, [radius, userLocation]); // Re-fetch when radius or userLocation changes

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

      {/* Radius Control */}
      <Card className="bg-white border-gray-200 shadow-sm">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Radio className="w-5 h-5 text-blue-600" />
              <span className="font-semibold text-gray-700">Search Radius</span>
            </div>
            <span className="text-sm text-gray-600">{radius} km</span>
          </div>
          <input
            type="range"
            min="1"
            max="20"
            value={radius}
            onChange={(e) => setRadius(Number(e.target.value))}
            className="w-full mt-3 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>1 km</span>
            <span className="text-gray-700 font-medium">Showing nearest facilities within range</span>
            <span>20 km</span>
          </div>
          {totalFound && (
            <div className="mt-3 text-xs text-gray-600">
              Total found in {radius}km: Police ({totalFound.police}), Hospitals ({totalFound.hospitals}), Multi-Specialty ({totalFound.multispecialty}), Fire ({totalFound.fire})
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="bg-white border-gray-200 shadow-sm">
        <CardContent className="p-4">
          <div className="flex flex-wrap gap-6 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-blue-500 rounded-full"></div>
              <span className="text-gray-700">Police ({stats.police})</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-cyan-500 rounded-full"></div>
              <span className="text-gray-700">General Hospitals ({stats.hospitals})</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-purple-500 rounded-full"></div>
              <span className="text-gray-700">Multi-Specialty ({stats.multispecialty})</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-amber-500 rounded-full"></div>
              <span className="text-gray-700">Fire Stations ({stats.fire})</span>
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
    </div >
  );
};

export default MapView;