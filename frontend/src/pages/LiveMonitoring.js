import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Video, AlertTriangle, Circle, Shield, Hospital, Navigation, Volume2, VolumeX, Maximize, Users, Activity } from "lucide-react";
import toast from "react-hot-toast";
import { motion } from "framer-motion";

// Use FastAPI backend on port 8000
const BACKEND_URL = "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

import { useMonitoring } from "../context/MonitoringContext";

const LiveMonitoring = () => {
  const {
    isMonitoring,
    stats,
    detections,
    startMonitoring,
    stopMonitoring,
    setVoiceEnabled: setGlobalVoiceEnabled,
    emergencyContext,
    locationStatus
  } = useMonitoring();

  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);

  // Sync local voice state with global context
  useEffect(() => {
    setGlobalVoiceEnabled(voiceEnabled);
  }, [voiceEnabled, setGlobalVoiceEnabled]);

  return (
    <div className="space-y-6" data-testid="live-monitoring-page">
      <div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent">
          Live Monitoring
        </h1>
        <p className="text-gray-600 mt-2">Real-time weapon detection via webcam</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Video Feed - Full Width */}
        <div className="lg:col-span-3">
          <Card className="bg-white border-gray-200 shadow-lg">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-gray-900 flex items-center">
                  <Video className="h-5 w-5 mr-2" />
                  Webcam Feed
                </CardTitle>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setVoiceEnabled(!voiceEnabled)}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                    title={voiceEnabled ? "Mute Voice Alerts" : "Enable Voice Alerts"}
                  >
                    {voiceEnabled ? <Volume2 className="h-5 w-5 text-gray-700" /> : <VolumeX className="h-5 w-5 text-red-500" />}
                  </button>
                  <button
                    onClick={() => setFullscreen(!fullscreen)}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                    title="Toggle Fullscreen"
                  >
                    <Maximize className="h-5 w-5 text-gray-700" />
                  </button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className={`relative bg-black rounded-lg overflow-hidden w-full ${fullscreen ? 'h-[80vh]' : 'h-[500px]'}`}>
                {isMonitoring ? (
                  <img
                    src={`${API}/video_feed`}
                    alt="Live Feed"
                    className="w-full h-full object-contain"
                    style={{ imageRendering: 'auto' }}
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <Video className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                      <p className="text-gray-400 text-lg">Click "Start Monitoring" to begin</p>
                      <p className="text-gray-500 text-sm mt-2">Webcam feed will appear here</p>
                    </div>
                  </div>
                )}
                {isMonitoring && (
                  <div className="absolute top-4 right-4 flex items-center space-x-2 bg-red-500/90 px-4 py-2 rounded-lg shadow-lg">
                    <Circle className="h-3 w-3 fill-white text-white animate-pulse" />
                    <span className="text-white text-sm font-semibold">LIVE</span>
                  </div>
                )}
              </div>

              <div className="flex gap-3">
                <Button
                  onClick={startMonitoring}
                  disabled={isMonitoring}
                  className="flex-1 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white disabled:opacity-50"
                >
                  <Video className="h-4 w-4 mr-2" />
                  {isMonitoring ? "Monitoring Active" : "Start Monitoring"}
                </Button>
                <Button
                  onClick={stopMonitoring}
                  disabled={!isMonitoring}
                  className="flex-1 bg-gradient-to-r from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700 text-white disabled:opacity-50"
                >
                  Stop Monitoring
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Stats Cards */}
          <div className="grid grid-cols-4 gap-4 mt-6">
            {[
              { label: "Frames", value: stats.framesProcessed, icon: Activity, color: "from-blue-500 to-cyan-500" },
              { label: "FPS", value: stats.fps, icon: Video, color: "from-purple-500 to-pink-500" },
              { label: "People", value: stats.peopleCount, icon: Users, color: "from-green-500 to-emerald-500" },
              { label: "Threats", value: stats.threatsDetected, icon: AlertTriangle, color: "from-red-500 to-orange-500" },
            ].map((stat, index) => {
              const Icon = stat.icon;
              return (
                <Card key={index} className="bg-white border-gray-200 hover:shadow-lg transition-shadow">
                  <CardContent className="pt-6">
                    <div className="text-center">
                      <div className={`bg-gradient-to-r ${stat.color} p-3 rounded-xl mx-auto w-fit mb-2`}>
                        <Icon className="h-6 w-6 text-white" />
                      </div>
                      <p className="text-3xl font-bold text-gray-900">{stat.value}</p>
                      <p className="text-sm text-gray-600 mt-1">{stat.label}</p>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Sidebar - Alerts & Emergency Services */}
        <div className="space-y-6">
          {/* Current Alerts */}
          <Card className="bg-white border-gray-200 shadow-lg">
            <CardHeader>
              <CardTitle className="text-gray-900 text-lg flex items-center">
                <AlertTriangle className="h-5 w-5 mr-2 text-red-500" />
                Current Alerts
              </CardTitle>
            </CardHeader>
            <CardContent>
              {detections.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Shield className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                  <p className="text-sm">No threats detected</p>
                  <p className="text-xs text-gray-400 mt-1">System monitoring</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {detections.map((det, idx) => (
                    <motion.div
                      key={idx}
                      initial={{ scale: 0.9 }}
                      animate={{ scale: 1 }}
                      className="bg-red-50 border border-red-200 rounded-lg p-4"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-semibold text-red-600 capitalize text-lg">
                            {det.detection_type}
                          </p>
                          <p className="text-sm text-gray-600">
                            Confidence: {(det.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                        <AlertTriangle className="h-6 w-6 text-red-500 animate-pulse" />
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Blockchain Verification */}
          <Card className="bg-white border-gray-200 shadow-lg">
            <CardHeader>
              <CardTitle className="text-gray-900 text-lg flex items-center">
                <Shield className="h-5 w-5 mr-2 text-purple-600" />
                Blockchain Verification
              </CardTitle>
            </CardHeader>
            <CardContent>
              {stats.latest_ipfs ? (
                <div className="space-y-3">
                  <div className="p-3 bg-purple-50 rounded-lg border border-purple-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-semibold text-purple-700 uppercase">
                        {stats.latest_ipfs.threat_type} EVIDENCE
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date(stats.latest_ipfs.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="text-xs text-gray-600 break-all font-mono bg-white p-2 rounded border border-purple-100 mb-2">
                      {stats.latest_ipfs.hash}
                    </div>
                    <a
                      href={stats.latest_ipfs.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-purple-600 hover:text-purple-800 underline flex items-center"
                    >
                      View on IPFS <Maximize className="h-3 w-3 ml-1" />
                    </a>
                  </div>
                </div>
              ) : (
                <div className="text-center py-4 text-gray-500">
                  <p className="text-sm">No recent uploads</p>
                  <p className="text-xs text-gray-400 mt-1">Waiting for high-confidence threats...</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Emergency Services */}
          <Card className="bg-white border-gray-200 shadow-lg">
            <CardHeader>
              <CardTitle className="text-gray-900 text-lg flex items-center justify-between">
                <div className="flex items-center">
                  <Navigation className="h-5 w-5 mr-2 text-blue-600" />
                  Emergency Services
                </div>
                <span className="text-xs text-gray-500 font-normal">{locationStatus}</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {!emergencyContext ? (
                <div className="text-center text-gray-500 text-sm py-4">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
                  Fetching location data...
                </div>
              ) : (
                <>
                  {/* Police */}
                  {emergencyContext.police_stations?.[0] && (
                    <div className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
                      <Shield className="h-5 w-5 text-blue-600 mt-1 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-blue-700">Nearest Police</p>
                        <p className="text-xs text-gray-700 truncate">{emergencyContext.police_stations[0].name}</p>
                        <p className="text-xs text-gray-600">{emergencyContext.police_stations[0].phone || 'N/A'}</p>
                        <p className="text-xs text-blue-600 mt-1">{emergencyContext.police_stations[0].distance?.toFixed(1)} km away</p>
                      </div>
                    </div>
                  )}

                  {/* Hospital */}
                  {emergencyContext.hospitals?.[0] && (
                    <div className="flex items-start space-x-3 p-3 bg-green-50 rounded-lg border border-green-200">
                      <Hospital className="h-5 w-5 text-green-600 mt-1 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-green-700">Nearest Hospital</p>
                        <p className="text-xs text-gray-700 truncate">{emergencyContext.hospitals[0].name}</p>
                        <p className="text-xs text-gray-600">{emergencyContext.hospitals[0].phone || 'N/A'}</p>
                        <p className="text-xs text-green-600 mt-1">{emergencyContext.hospitals[0].distance?.toFixed(1)} km away</p>
                      </div>
                    </div>
                  )}

                  {/* Multispeciality Hospital */}
                  {emergencyContext.hospitals?.[1] && (
                    <div className="flex items-start space-x-3 p-3 bg-cyan-50 rounded-lg border border-cyan-200">
                      <Hospital className="h-5 w-5 text-cyan-600 mt-1 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-cyan-700">Multispeciality Hospital</p>
                        <p className="text-xs text-gray-700 truncate">{emergencyContext.hospitals[1].name}</p>
                        <p className="text-xs text-gray-600">{emergencyContext.hospitals[1].phone || 'N/A'}</p>
                        <p className="text-xs text-cyan-600 mt-1">{emergencyContext.hospitals[1].distance?.toFixed(1)} km away</p>
                      </div>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default LiveMonitoring;
