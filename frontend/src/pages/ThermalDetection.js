import React, { useState, useEffect } from "react";
import axios from "axios";
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Thermometer, MapPin, Image as ImageIcon, Video as VideoIcon } from "lucide-react";
import { toast } from "sonner";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

// Fix leaflet default icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require("leaflet/dist/images/marker-icon-2x.png"),
  iconUrl: require("leaflet/dist/images/marker-icon.png"),
  shadowUrl: require("leaflet/dist/images/marker-shadow.png"),
});

const defaultCenter = [18.5204, 73.8567];

const ThermalDetection = () => {
  const [userLocation, setUserLocation] = useState(null);
  const [selectedLocation, setSelectedLocation] = useState(null);
  const [locLoading, setLocLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState(null);

  useEffect(() => {
    if (!navigator.geolocation) {
      setUserLocation(null);
      return;
    }
    setLocLoading(true);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const loc = [pos.coords.latitude, pos.coords.longitude];
        setUserLocation(loc);
        setSelectedLocation((prev) => prev || loc);
        setLocLoading(false);
      },
      () => {
        setLocLoading(false);
      },
      { enableHighAccuracy: true }
    );
  }, []);

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f || null);
    setResults(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      toast.error("Please select a file");
      return;
    }
    if (!selectedLocation) {
      toast.error("Please click on the map to choose a location");
      return;
    }

    try {
      setUploading(true);
      const formData = new FormData();
      formData.append("file", file);

      const [lat, lng] = selectedLocation;
      const params = new URLSearchParams({
        lat: String(lat),
        lng: String(lng),
        camera_name: "Thermal Map", // for backend context
      });

      const response = await axios.post(
        `${API}/detect-thermal-guns?${params.toString()}`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResults(response.data);

      if (response.data.detections && response.data.detections.length > 0) {
        toast.error(`${response.data.thermal_guns_count} thermal gun(s) detected!`);
      } else {
        toast.success("Scan complete - No thermal guns detected");
      }
    } catch (err) {
      console.error("Error running thermal detection:", err);
      toast.error("Thermal detection failed");
    } finally {
      setUploading(false);
    }
  };

  const LocationPicker = () => {
    useMapEvents({
      click(e) {
        setSelectedLocation([e.latlng.lat, e.latlng.lng]);
      },
    });
    return null;
  };

  const center = selectedLocation || userLocation || defaultCenter;

  const selectedLat = selectedLocation ? selectedLocation[0].toFixed(4) : "-";
  const selectedLng = selectedLocation ? selectedLocation[1].toFixed(4) : "-";

  return (
    <div className="space-y-6" data-testid="thermal-detection-page">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            Thermal Detection
          </h1>
          <p className="text-slate-400 mt-2">
            Click on the map to choose a camera location and run thermal gun detection using Roboflow.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Map + location info */}
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-slate-100">
              <MapPin className="h-5 w-5 text-green-400" />
              <span>Choose Detection Location</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-sm text-slate-300 flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">Selected Coordinates</p>
                <p className="font-mono">
                  {selectedLat}, {selectedLng}
                </p>
              </div>
              <div className="text-xs text-slate-400">
                {locLoading
                  ? "Detecting your location..."
                  : userLocation
                  ? "You can move the marker by clicking on the map"
                  : "Geolocation unavailable - using default map center"}
              </div>
            </div>

            <div style={{ height: "420px" }} data-testid="thermal-map-container">
              <MapContainer center={center} zoom={13} style={{ height: "100%", width: "100%" }}>
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <LocationPicker />
                {selectedLocation && (
                  <Marker position={selectedLocation}>
                    <Popup>
                      <div className="text-sm text-slate-800">
                        <strong>Detection Location</strong>
                        <p className="text-xs">
                          {selectedLat}, {selectedLng}
                        </p>
                      </div>
                    </Popup>
                  </Marker>
                )}
                {userLocation && !selectedLocation && (
                  <Marker position={userLocation}>
                    <Popup>
                      <div className="text-sm text-slate-800">
                        <strong>Your Location</strong>
                        <p className="text-xs">
                          {userLocation[0].toFixed(4)}, {userLocation[1].toFixed(4)}
                        </p>
                      </div>
                    </Popup>
                  </Marker>
                )}
              </MapContainer>
            </div>
          </CardContent>
        </Card>

        {/* Upload + results */}
        <div className="space-y-6">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="flex items-center text-white">
                <Thermometer className="h-5 w-5 mr-2 text-purple-400" />
                Upload Thermal Image / Video
              </CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label htmlFor="thermal-file" className="block text-sm font-medium text-slate-300 mb-1">
                    File (Image or Video)
                  </label>
                  <Input
                    id="thermal-file"
                    type="file"
                    accept="image/*,video/*"
                    onChange={handleFileChange}
                    className="bg-slate-700 border-slate-600 text-white file:bg-slate-600 file:text-white file:border-0 file:mr-4 file:py-2 file:px-4"
                    required
                  />
                  {file && (
                    <p className="text-sm text-slate-400 mt-2 flex items-center">
                      {file.type.startsWith("video") ? (
                        <VideoIcon className="h-4 w-4 mr-2" />
                      ) : (
                        <ImageIcon className="h-4 w-4 mr-2" />
                      )}
                      {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                    </p>
                  )}
                </div>

                <Button
                  type="submit"
                  disabled={uploading}
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
                >
                  {uploading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Processing...
                    </>
                  ) : (
                    <>
                      <Thermometer className="h-4 w-4 mr-2" />
                      Run Thermal Detection
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader>
              <CardTitle className="flex items-center text-white">
                <Thermometer className="h-5 w-5 mr-2" />
                Thermal Detection Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!results ? (
                <div className="text-center py-12 text-slate-400">
                  <Thermometer className="h-16 w-16 mx-auto mb-4 opacity-50" />
                  <p>Upload a file to see thermal gun detection results</p>
                </div>
              ) : results.detections && results.detections.length === 0 ? (
                <div className="text-center py-12">
                  <div className="bg-green-500/10 border border-green-500/30 rounded-full w-20 h-20 mx-auto mb-4 flex items-center justify-center">
                    <svg className="h-10 w-10 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold text-green-400 mb-2">All Clear</h3>
                  <p className="text-slate-400">No thermal guns detected</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-2">
                    <h3
                      className="font-semibold text-purple-400 mb-1 flex items-center"
                      style={{ fontSize: "10px", lineHeight: "1" }}
                    >
                      <Thermometer className="h-2 w-2 mr-1" />
                      {results.thermal_guns_count} Thermal Gun(s) Detected
                    </h3>
                  </div>

                  {/* Annotated image */}
                  {(results.annotated_image || (results.detections && results.detections[0]?.image_path)) && (
                    <div className="bg-slate-900/50 border border-purple-700/50 rounded-lg p-2 mb-2">
                      <h4
                        className="font-semibold text-purple-300 mb-2"
                        style={{ fontSize: "10px", lineHeight: "1" }}
                      >
                        Annotated Detection Image
                      </h4>
                      <img
                        src={`${BACKEND_URL}${results.annotated_image || results.detections[0].image_path}?t=${Date.now()}`}
                        alt="Thermal Gun Detection with Boxes"
                        className="w-full rounded-lg border border-purple-700/30"
                      />
                    </div>
                  )}

                  {/* Individual detections */}
                  <div className="space-y-2 max-h-[400px] overflow-y-auto">
                    {results.detections?.map((det, idx) => (
                      <div
                        key={idx}
                        className="bg-slate-900/50 border border-purple-700/50 rounded-lg p-2 text-xs text-slate-200"
                      >
                        <div className="flex items-start justify-between">
                          <div>
                            <p className="font-semibold text-purple-300">Thermal Gun</p>
                            <p className="text-slate-400">
                              Confidence: {(det.confidence * 100).toFixed(1)}%
                            </p>
                            {det.model && (
                              <p className="text-slate-500">Model: {det.model}</p>
                            )}
                            {det.location && det.location.lat != null && det.location.lng != null && (
                              <p className="text-slate-500 mt-1">
                                Location: {det.location.lat.toFixed(4)}, {det.location.lng.toFixed(4)}
                              </p>
                            )}
                            {det.police_station && (
                              <p className="text-slate-500 mt-1">
                                Nearest Police: {det.police_station.name}
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ThermalDetection;
