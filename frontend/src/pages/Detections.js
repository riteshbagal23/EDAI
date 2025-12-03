import React, { useState, useEffect } from "react";
import axios from "axios";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle, MapPin, CheckCircle2, Shield, Map, List } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import DetectionMapView from "@/components/DetectionMapView";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Detections = () => {
  const [detections, setDetections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [verificationResult, setVerificationResult] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [viewMode, setViewMode] = useState('list'); // 'list' or 'map'

  const fetchDetections = async () => {
    try {
      const response = await axios.get(`${API}/detections`);
      setDetections(response.data);
      setLoading(false);
    } catch (error) {
      console.error("Error fetching detections:", error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDetections();
    const interval = setInterval(fetchDetections, 30000); // Reduced frequency to 30s
    return () => clearInterval(interval);
  }, []);

  const handleVerify = async (detection) => {
    setSelectedDetection(detection);
    setDialogOpen(true);

    try {
      const response = await axios.get(`${API}/blockchain/verify/${detection.id}`);
      setVerificationResult(response.data);
    } catch (error) {
      console.error("Error verifying blockchain:", error);
    }
  };

  return (
    <div className="space-y-6" data-testid="detections-page">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-red-400 to-pink-400 bg-clip-text text-transparent">
            Detection History
          </h1>
          <p className="text-slate-400 mt-2">All threat detections with blockchain verification</p>
        </div>
        <div className="flex items-center space-x-4">
          {/* View Toggle */}
          <div className="flex bg-slate-800/50 border border-slate-700 rounded-lg p-1">
            <button
              onClick={() => setViewMode('list')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-all ${viewMode === 'list'
                ? 'bg-blue-500 text-white'
                : 'text-slate-400 hover:text-white'
                }`}
            >
              <List className="h-4 w-4" />
              <span className="text-sm font-medium">List</span>
            </button>
            <button
              onClick={() => setViewMode('map')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-md transition-all ${viewMode === 'map'
                ? 'bg-blue-500 text-white'
                : 'text-slate-400 hover:text-white'
                }`}
            >
              <Map className="h-4 w-4" />
              <span className="text-sm font-medium">Map</span>
            </button>
          </div>
          {/* Detection Count */}
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-2">
            <p className="text-sm text-slate-300">
              Total Detections: <span className="font-bold text-red-400">{detections.length}</span>
            </p>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-pulse text-slate-400">Loading detections...</div>
        </div>
      ) : detections.length === 0 ? (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="py-12">
            <div className="text-center">
              <Shield className="h-20 w-20 mx-auto text-slate-600 mb-4" />
              <h3 className="text-xl font-semibold text-slate-300 mb-2">No detections yet</h3>
              <p className="text-slate-400">System is actively monitoring. Detections will appear here.</p>
            </div>
          </CardContent>
        </Card>
      ) : viewMode === 'map' ? (
        <DetectionMapView detections={detections} />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {detections.map((detection, index) => (
            <Card
              key={detection.id}
              data-testid={`detection-card-${index}`}
              className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-all detection-card"
            >
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="bg-gradient-to-br from-red-500 to-orange-500 p-3 rounded-lg">
                      <AlertTriangle className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-white capitalize">{detection.detection_type}</CardTitle>
                      <p className="text-sm text-slate-400 mt-1">
                        {(detection.confidence * 100).toFixed(1)}% confidence
                      </p>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {detection.image_path && (
                  <div className="relative group">
                    <img
                      src={`${BACKEND_URL}${detection.image_path}`}
                      alt="Detection"
                      className="w-full rounded-lg border border-slate-700"
                    />
                    <div className="absolute top-2 right-2 bg-red-500 text-white text-xs px-2 py-1 rounded-full font-semibold">
                      THREAT
                    </div>
                  </div>
                )}

                <div className="space-y-2 text-sm">
                  {/* Camera information removed */}
                  <div className="flex items-center space-x-2 text-slate-400">
                    <MapPin className="h-4 w-4" />
                    <span>
                      {detection.location.lat.toFixed(4)}, {detection.location.lng.toFixed(4)}
                    </span>
                  </div>
                  <div className="mt-2">
                    <a
                      href={`https://www.google.com/maps?q=${detection.location.lat},${detection.location.lng}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center space-x-1 text-xs text-blue-400 hover:text-blue-300 transition-colors"
                    >
                      <MapPin className="h-3 w-3" />
                      <span>View on Google Maps</span>
                    </a>
                  </div>
                  <div className="pt-2 border-t border-slate-700">
                    <p className="text-xs text-slate-500">
                      {new Date(detection.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>

                <div className="pt-2">
                  <Button
                    data-testid={`verify-button-${index}`}
                    onClick={() => handleVerify(detection)}
                    className="w-full bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600"
                  >
                    <Shield className="h-4 w-4 mr-2" />
                    Verify Blockchain
                  </Button>
                </div>

                {detection.alert_sent && (
                  <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-2 flex items-center space-x-2">
                    <CheckCircle2 className="h-4 w-4 text-green-400" />
                    <span className="text-xs text-green-400">Alert Sent</span>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="bg-slate-800 border-slate-700 text-white max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center">
              <Shield className="h-5 w-5 mr-2" />
              Blockchain Verification
            </DialogTitle>
          </DialogHeader>
          {selectedDetection && verificationResult && (
            <div className="space-y-4">
              <div className={`p-4 rounded-lg border ${verificationResult.verified
                ? 'bg-green-500/10 border-green-500/30'
                : 'bg-red-500/10 border-red-500/30'
                }`}>
                <div className="flex items-center space-x-3">
                  {verificationResult.verified ? (
                    <CheckCircle2 className="h-6 w-6 text-green-400" />
                  ) : (
                    <AlertTriangle className="h-6 w-6 text-red-400" />
                  )}
                  <div>
                    <h3 className={`font-semibold ${verificationResult.verified ? 'text-green-400' : 'text-red-400'
                      }`}>
                      {verificationResult.verified ? 'Verified' : 'Verification Failed'}
                    </h3>
                    <p className="text-sm text-slate-400">
                      {verificationResult.verified
                        ? 'Evidence integrity confirmed'
                        : 'Evidence may have been tampered'}
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <h4 className="text-sm font-semibold text-slate-300 mb-1">Detection Details</h4>
                  <div className="bg-slate-900/50 rounded-lg p-3 text-sm space-y-1">
                    <p className="text-slate-400">
                      <span className="text-slate-500">Type:</span>{' '}
                      <span className="text-white capitalize">{selectedDetection.detection_type}</span>
                    </p>
                    {/* Camera info removed from verification details */}
                    <p className="text-slate-400">
                      <span className="text-slate-500">Confidence:</span>{' '}
                      <span className="text-white">{(selectedDetection.confidence * 100).toFixed(1)}%</span>
                    </p>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-slate-300 mb-1">Stored Hash</h4>
                  <div className="bg-slate-900/50 rounded-lg p-3">
                    <code className="text-xs text-green-400 break-all">
                      {verificationResult.stored_hash}
                    </code>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-slate-300 mb-1">Calculated Hash</h4>
                  <div className="bg-slate-900/50 rounded-lg p-3">
                    <code className="text-xs text-blue-400 break-all">
                      {verificationResult.calculated_hash}
                    </code>
                  </div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Detections;