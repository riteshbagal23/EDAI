import React, { useState, useEffect } from "react";
import axios from "axios";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
    Camera,
    Plus,
    Play,
    Square,
    Trash2,
    Edit,
    Circle,
    AlertCircle,
    MapPin,
    Video,
    Maximize2
} from "lucide-react";
import { toast } from "sonner";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";

const BACKEND_URL = "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

const Cameras = () => {
    const [cameras, setCameras] = useState([]);
    const [loading, setLoading] = useState(true);
    const [isAddModalOpen, setIsAddModalOpen] = useState(false);
    const [selectedCamera, setSelectedCamera] = useState(null);
    const [isViewModalOpen, setIsViewModalOpen] = useState(false);

    // Form state for adding new camera
    const [newCamera, setNewCamera] = useState({
        name: "",
        type: "webcam",
        source: "0",
        location: { lat: 18.5204, lng: 73.8567 },
        settings: { resolution: "1280x720", fps: 30 }
    });

    // Fetch cameras on mount and periodically
    useEffect(() => {
        fetchCameras();
        const interval = setInterval(fetchCameras, 3000); // Refresh every 3 seconds
        return () => clearInterval(interval);
    }, []);

    const fetchCameras = async () => {
        try {
            const response = await axios.get(`${API}/cameras`);
            setCameras(response.data.cameras || []);
            setLoading(false);
        } catch (error) {
            console.error("Error fetching cameras:", error);
            toast.error("Failed to fetch cameras");
            setLoading(false);
        }
    };

    const handleAddCamera = async () => {
        try {
            const response = await axios.post(`${API}/cameras`, newCamera);
            toast.success(`Camera "${newCamera.name}" added successfully!`);
            setIsAddModalOpen(false);
            setNewCamera({
                name: "",
                type: "webcam",
                source: "0",
                location: { lat: 18.5204, lng: 73.8567 },
                settings: { resolution: "1280x720", fps: 30 }
            });
            fetchCameras();
        } catch (error) {
            console.error("Error adding camera:", error);
            toast.error(error.response?.data?.detail || "Failed to add camera");
        }
    };

    const handleStartCamera = async (cameraId, cameraName) => {
        try {
            await axios.post(`${API}/cameras/${cameraId}/start`);
            toast.success(`Camera "${cameraName}" started`);
            fetchCameras();
        } catch (error) {
            console.error("Error starting camera:", error);
            toast.error(error.response?.data?.detail || "Failed to start camera");
        }
    };

    const handleStopCamera = async (cameraId, cameraName) => {
        try {
            await axios.post(`${API}/cameras/${cameraId}/stop`);
            toast.info(`Camera "${cameraName}" stopped`);
            fetchCameras();
        } catch (error) {
            console.error("Error stopping camera:", error);
            toast.error("Failed to stop camera");
        }
    };

    const handleDeleteCamera = async (cameraId, cameraName) => {
        if (!window.confirm(`Are you sure you want to delete camera "${cameraName}"?`)) {
            return;
        }

        try {
            await axios.delete(`${API}/cameras/${cameraId}`);
            toast.success(`Camera "${cameraName}" deleted`);
            fetchCameras();
        } catch (error) {
            console.error("Error deleting camera:", error);
            toast.error("Failed to delete camera");
        }
    };

    const handleViewCamera = (camera) => {
        setSelectedCamera(camera);
        setIsViewModalOpen(true);
    };

    const getStatusColor = (status) => {
        switch (status) {
            case "active":
                return "text-green-400 bg-green-500/20 border-green-500/30";
            case "inactive":
                return "text-slate-400 bg-slate-500/20 border-slate-500/30";
            case "error":
                return "text-red-400 bg-red-500/20 border-red-500/30";
            case "starting":
                return "text-yellow-400 bg-yellow-500/20 border-yellow-500/30";
            default:
                return "text-slate-400 bg-slate-500/20 border-slate-500/30";
        }
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case "active":
                return <Circle className="h-3 w-3 fill-green-400 text-green-400 animate-pulse" />;
            case "error":
                return <AlertCircle className="h-3 w-3 text-red-400" />;
            default:
                return <Circle className="h-3 w-3 text-slate-400" />;
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="text-slate-400">Loading cameras...</div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-4xl font-bold bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
                        Camera Management
                    </h1>
                    <p className="text-slate-400 mt-2">
                        Manage and monitor multiple camera feeds
                    </p>
                </div>

                <Dialog open={isAddModalOpen} onOpenChange={setIsAddModalOpen}>
                    <DialogTrigger asChild>
                        <Button className="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700">
                            <Plus className="h-4 w-4 mr-2" />
                            Add Camera
                        </Button>
                    </DialogTrigger>
                    <DialogContent className="bg-slate-800 border-slate-700 text-white">
                        <DialogHeader>
                            <DialogTitle>Add New Camera</DialogTitle>
                            <DialogDescription className="text-slate-400">
                                Configure a new camera source for monitoring
                            </DialogDescription>
                        </DialogHeader>

                        <div className="space-y-4 mt-4">
                            <div>
                                <Label htmlFor="camera-name">Camera Name</Label>
                                <Input
                                    id="camera-name"
                                    placeholder="e.g., Front Entrance"
                                    value={newCamera.name}
                                    onChange={(e) => setNewCamera({ ...newCamera, name: e.target.value })}
                                    className="bg-slate-700 border-slate-600 text-white"
                                />
                            </div>

                            <div>
                                <Label htmlFor="camera-type">Camera Type</Label>
                                <Select
                                    value={newCamera.type}
                                    onValueChange={(value) => {
                                        setNewCamera({ ...newCamera, type: value });
                                        // Set default source based on type
                                        if (value === "webcam") setNewCamera(prev => ({ ...prev, source: "0" }));
                                        else if (value === "rtsp") setNewCamera(prev => ({ ...prev, source: "rtsp://" }));
                                        else if (value === "ip") setNewCamera(prev => ({ ...prev, source: "http://" }));
                                    }}
                                >
                                    <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent className="bg-slate-700 border-slate-600 text-white">
                                        <SelectItem value="webcam">Webcam</SelectItem>
                                        <SelectItem value="rtsp">RTSP Stream</SelectItem>
                                        <SelectItem value="ip">IP Camera</SelectItem>
                                        <SelectItem value="file">Video File</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>

                            <div>
                                <Label htmlFor="camera-source">
                                    Source {newCamera.type === "webcam" ? "(Camera Index)" : "(URL/Path)"}
                                </Label>
                                <Input
                                    id="camera-source"
                                    placeholder={
                                        newCamera.type === "webcam" ? "0" :
                                            newCamera.type === "rtsp" ? "rtsp://username:password@ip:port/stream" :
                                                newCamera.type === "ip" ? "http://ip:port/video" :
                                                    "/path/to/video.mp4"
                                    }
                                    value={newCamera.source}
                                    onChange={(e) => setNewCamera({ ...newCamera, source: e.target.value })}
                                    className="bg-slate-700 border-slate-600 text-white"
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <Label htmlFor="camera-lat">Latitude</Label>
                                    <Input
                                        id="camera-lat"
                                        type="number"
                                        step="0.0001"
                                        value={newCamera.location.lat}
                                        onChange={(e) => setNewCamera({
                                            ...newCamera,
                                            location: { ...newCamera.location, lat: parseFloat(e.target.value) }
                                        })}
                                        className="bg-slate-700 border-slate-600 text-white"
                                    />
                                </div>
                                <div>
                                    <Label htmlFor="camera-lng">Longitude</Label>
                                    <Input
                                        id="camera-lng"
                                        type="number"
                                        step="0.0001"
                                        value={newCamera.location.lng}
                                        onChange={(e) => setNewCamera({
                                            ...newCamera,
                                            location: { ...newCamera.location, lng: parseFloat(e.target.value) }
                                        })}
                                        className="bg-slate-700 border-slate-600 text-white"
                                    />
                                </div>
                            </div>

                            <Button
                                onClick={handleAddCamera}
                                disabled={!newCamera.name || !newCamera.source}
                                className="w-full bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700"
                            >
                                Add Camera
                            </Button>
                        </div>
                    </DialogContent>
                </Dialog>
            </div>

            {/* Camera Grid */}
            {cameras.length === 0 ? (
                <Card className="bg-slate-800/50 border-slate-700">
                    <CardContent className="flex flex-col items-center justify-center py-16">
                        <Camera className="h-16 w-16 text-slate-600 mb-4" />
                        <p className="text-slate-400 text-lg mb-2">No cameras registered</p>
                        <p className="text-slate-500 text-sm mb-4">Add your first camera to get started</p>
                        <Button
                            onClick={() => setIsAddModalOpen(true)}
                            className="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700"
                        >
                            <Plus className="h-4 w-4 mr-2" />
                            Add Camera
                        </Button>
                    </CardContent>
                </Card>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {cameras.map((camera) => (
                        <Card key={camera.id} className="bg-slate-800/50 border-slate-700 hover:border-slate-600 transition-colors">
                            <CardHeader>
                                <CardTitle className="flex items-center justify-between text-white">
                                    <div className="flex items-center space-x-2">
                                        <Camera className="h-5 w-5 text-orange-400" />
                                        <span className="truncate">{camera.name}</span>
                                    </div>
                                    <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs border ${getStatusColor(camera.status)}`}>
                                        {getStatusIcon(camera.status)}
                                        <span className="capitalize">{camera.status}</span>
                                    </div>
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                {/* Camera Preview */}
                                <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: "16/9" }}>
                                    {camera.status === "active" ? (
                                        <img
                                            src={`${API}/cameras/${camera.id}/video_feed?t=${Date.now()}`}
                                            alt={camera.name}
                                            className="w-full h-full object-cover"
                                            onError={(e) => {
                                                e.target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect fill='%23334155' width='100' height='100'/%3E%3C/svg%3E";
                                            }}
                                        />
                                    ) : (
                                        <div className="flex items-center justify-center h-full">
                                            <Video className="h-12 w-12 text-slate-600" />
                                        </div>
                                    )}
                                    <button
                                        onClick={() => handleViewCamera(camera)}
                                        className="absolute top-2 right-2 bg-black/60 hover:bg-black/80 text-white p-2 rounded-full transition-colors"
                                    >
                                        <Maximize2 className="h-4 w-4" />
                                    </button>
                                </div>

                                {/* Camera Info */}
                                <div className="space-y-2 text-sm">
                                    <div className="flex items-center text-slate-400">
                                        <span className="font-medium mr-2">Type:</span>
                                        <span className="capitalize">{camera.type}</span>
                                    </div>
                                    <div className="flex items-center text-slate-400">
                                        <MapPin className="h-3 w-3 mr-1" />
                                        <span>{camera.location.lat.toFixed(4)}, {camera.location.lng.toFixed(4)}</span>
                                    </div>
                                    {camera.stats && (
                                        <div className="flex items-center justify-between text-xs">
                                            <span className="text-slate-500">Frames: {camera.stats.frames_processed}</span>
                                            <span className="text-red-400">Threats: {camera.stats.guns + camera.stats.knives}</span>
                                        </div>
                                    )}
                                </div>

                                {/* Control Buttons */}
                                <div className="flex gap-2">
                                    {camera.status === "active" ? (
                                        <Button
                                            onClick={() => handleStopCamera(camera.id, camera.name)}
                                            className="flex-1 bg-gradient-to-r from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700"
                                            size="sm"
                                        >
                                            <Square className="h-3 w-3 mr-1" />
                                            Stop
                                        </Button>
                                    ) : (
                                        <Button
                                            onClick={() => handleStartCamera(camera.id, camera.name)}
                                            className="flex-1 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700"
                                            size="sm"
                                        >
                                            <Play className="h-3 w-3 mr-1" />
                                            Start
                                        </Button>
                                    )}
                                    <Button
                                        onClick={() => handleDeleteCamera(camera.id, camera.name)}
                                        variant="outline"
                                        className="border-red-500/30 text-red-400 hover:bg-red-500/10"
                                        size="sm"
                                    >
                                        <Trash2 className="h-3 w-3" />
                                    </Button>
                                </div>

                                {camera.error_message && (
                                    <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded p-2">
                                        {camera.error_message}
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    ))}
                </div>
            )}

            {/* Full View Modal */}
            <Dialog open={isViewModalOpen} onOpenChange={setIsViewModalOpen}>
                <DialogContent className="bg-slate-800 border-slate-700 text-white max-w-4xl">
                    {selectedCamera && (
                        <>
                            <DialogHeader>
                                <DialogTitle className="flex items-center space-x-2">
                                    <Camera className="h-5 w-5 text-orange-400" />
                                    <span>{selectedCamera.name}</span>
                                    <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs border ${getStatusColor(selectedCamera.status)}`}>
                                        {getStatusIcon(selectedCamera.status)}
                                        <span className="capitalize">{selectedCamera.status}</span>
                                    </div>
                                </DialogTitle>
                            </DialogHeader>
                            <div className="mt-4">
                                {selectedCamera.status === "active" ? (
                                    <img
                                        src={`${API}/cameras/${selectedCamera.id}/video_feed?t=${Date.now()}`}
                                        alt={selectedCamera.name}
                                        className="w-full rounded-lg"
                                    />
                                ) : (
                                    <div className="bg-black rounded-lg flex items-center justify-center" style={{ aspectRatio: "16/9" }}>
                                        <div className="text-center text-slate-400">
                                            <Video className="h-16 w-16 mx-auto mb-2" />
                                            <p>Camera is {selectedCamera.status}</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </>
                    )}
                </DialogContent>
            </Dialog>
        </div>
    );
};

export default Cameras;
