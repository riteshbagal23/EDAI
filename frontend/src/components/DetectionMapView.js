import React, { useState } from 'react';
import { MapPin, X } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

const DetectionMapView = ({ detections }) => {
    const [selectedDetection, setSelectedDetection] = useState(null);

    // Calculate bounds for all detections
    const calculateMapCenter = () => {
        if (detections.length === 0) return { lat: 0, lng: 0 };

        const avgLat = detections.reduce((sum, d) => sum + d.location.lat, 0) / detections.length;
        const avgLng = detections.reduce((sum, d) => sum + d.location.lng, 0) / detections.length;

        return { lat: avgLat, lng: avgLng };
    };

    const center = calculateMapCenter();

    // Color mapping for detection types
    const getMarkerColor = (type) => {
        const colors = {
            pistol: '#EF4444', // Red
            knife: '#F97316', // Orange
            thermal_gun: '#A855F7', // Magenta
            person: '#FBBF24', // Yellow
        };
        return colors[type] || '#6B7280'; // Gray default
    };

    const openInGoogleMaps = (lat, lng) => {
        window.open(`https://www.google.com/maps?q=${lat},${lng}`, '_blank');
    };

    return (
        <div className="space-y-4">
            {/* Map Header */}
            <div className="flex items-center justify-between bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <div className="flex items-center space-x-2">
                    <MapPin className="h-5 w-5 text-blue-400" />
                    <h2 className="text-xl font-semibold text-white">Detection Map</h2>
                </div>
                <div className="text-sm text-slate-400">
                    {detections.length} detection{detections.length !== 1 ? 's' : ''}
                </div>
            </div>

            {/* Simple Map Visualization */}
            <div className="relative bg-slate-900/50 border border-slate-700 rounded-lg overflow-hidden" style={{ height: '500px' }}>
                {/* Map Background with Grid */}
                <div className="absolute inset-0 opacity-20">
                    <div className="grid grid-cols-10 grid-rows-10 h-full">
                        {Array.from({ length: 100 }).map((_, i) => (
                            <div key={i} className="border border-slate-700" />
                        ))}
                    </div>
                </div>

                {/* Center Coordinates Display */}
                <div className="absolute top-4 left-4 bg-slate-800/90 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300">
                    Center: {center.lat.toFixed(4)}, {center.lng.toFixed(4)}
                </div>

                {/* Legend */}
                <div className="absolute top-4 right-4 bg-slate-800/90 border border-slate-700 rounded-lg p-3 space-y-2">
                    <div className="text-xs font-semibold text-slate-300 mb-2">Detection Types</div>
                    {['pistol', 'knife', 'thermal_gun', 'person'].map(type => (
                        <div key={type} className="flex items-center space-x-2 text-xs">
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getMarkerColor(type) }} />
                            <span className="text-slate-400 capitalize">{type.replace('_', ' ')}</span>
                        </div>
                    ))}
                </div>

                {/* Detection Markers */}
                <div className="absolute inset-0 flex items-center justify-center p-8">
                    <div className="relative w-full h-full">
                        {detections.map((detection, index) => {
                            // Normalize coordinates to fit in the container (simple visualization)
                            // This is a simplified approach - real maps would use actual projections
                            const latRange = detections.length > 0 ? Math.max(...detections.map(d => d.location.lat)) - Math.min(...detections.map(d => d.location.lat)) : 1;
                            const lngRange = detections.length > 0 ? Math.max(...detections.map(d => d.location.lng)) - Math.min(...detections.map(d => d.location.lng)) : 1;

                            const minLat = Math.min(...detections.map(d => d.location.lat));
                            const minLng = Math.min(...detections.map(d => d.location.lng));

                            const x = latRange === 0 ? 50 : ((detection.location.lat - minLat) / latRange) * 80 + 10;
                            const y = lngRange === 0 ? 50 : ((detection.location.lng - minLng) / lngRange) * 80 + 10;

                            return (
                                <div
                                    key={detection.id}
                                    className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer group"
                                    style={{ left: `${y}%`, top: `${x}%` }}
                                    onClick={() => setSelectedDetection(detection)}
                                >
                                    {/* Marker Pin */}
                                    <div className="relative">
                                        <div
                                            className="w-6 h-6 rounded-full border-2 border-white shadow-lg group-hover:scale-110 transition-transform"
                                            style={{ backgroundColor: getMarkerColor(detection.detection_type) }}
                                        />
                                        <div
                                            className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent"
                                            style={{ borderTopColor: getMarkerColor(detection.detection_type) }}
                                        />
                                    </div>

                                    {/* Hover Tooltip */}
                                    <div className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
                                        <div className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-xs text-white shadow-xl">
                                            {detection.detection_type.toUpperCase()}
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Selected Detection Info */}
                {selectedDetection && (
                    <div className="absolute bottom-4 left-4 right-4 max-w-md mx-auto">
                        <Card className="bg-slate-800/95 border-slate-700 backdrop-blur">
                            <CardContent className="p-4">
                                <div className="flex items-start justify-between mb-3">
                                    <div>
                                        <h3 className="text-white font-semibold capitalize">{selectedDetection.detection_type}</h3>
                                        <p className="text-sm text-slate-400 mt-1">
                                            {(selectedDetection.confidence * 100).toFixed(1)}% confidence
                                        </p>
                                    </div>
                                    <button
                                        onClick={() => setSelectedDetection(null)}
                                        className="text-slate-400 hover:text-white transition-colors"
                                    >
                                        <X className="h-4 w-4" />
                                    </button>
                                </div>

                                {selectedDetection.image_path && (
                                    <img
                                        src={`${BACKEND_URL}${selectedDetection.image_path}`}
                                        alt="Detection"
                                        className="w-full rounded-lg border border-slate-700 mb-3"
                                    />
                                )}

                                <div className="space-y-2 text-sm">
                                    <div className="flex items-center space-x-2 text-slate-400">
                                        <MapPin className="h-4 w-4" />
                                        <span>
                                            {selectedDetection.location.lat.toFixed(4)}, {selectedDetection.location.lng.toFixed(4)}
                                        </span>
                                    </div>
                                    <div className="text-xs text-slate-500">
                                        {new Date(selectedDetection.timestamp).toLocaleString()}
                                    </div>
                                </div>

                                <Button
                                    onClick={() => openInGoogleMaps(selectedDetection.location.lat, selectedDetection.location.lng)}
                                    className="w-full mt-3 bg-blue-500 hover:bg-blue-600"
                                    size="sm"
                                >
                                    <MapPin className="h-4 w-4 mr-2" />
                                    Open in Google Maps
                                </Button>
                            </CardContent>
                        </Card>
                    </div>
                )}
            </div>

            {/* Quick Actions */}
            <div className="flex justify-center space-x-4">
                <Button
                    onClick={() => openInGoogleMaps(center.lat, center.lng)}
                    variant="outline"
                    className="border-slate-700 text-slate-300 hover:bg-slate-800"
                >
                    <MapPin className="h-4 w-4 mr-2" />
                    View Area in Google Maps
                </Button>
            </div>
        </div>
    );
};

export default DetectionMapView;
