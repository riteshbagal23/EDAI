import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
    CheckCircle,
    XCircle,
    Clock,
    ZoomIn,
    MapPin,
    Camera,
    Loader2
} from 'lucide-react';
import { motion } from 'framer-motion';

import { useMonitoring } from "../context/MonitoringContext";

const API_BASE_URL = 'http://localhost:8000/api';

function AdminVerification() {
    const { verifications, verificationStats: stats, fetchVerifications } = useMonitoring();
    const [loading, setLoading] = useState(false); // Managed globally now
    const [selectedTab, setSelectedTab] = useState('pending');
    const [selectedVerification, setSelectedVerification] = useState(null);
    const [imageDialogOpen, setImageDialogOpen] = useState(false);
    const [selectedImage, setSelectedImage] = useState(null);
    const [processingId, setProcessingId] = useState(null);
    const [hasMore, setHasMore] = useState(false); // Pagination state
    const [notes, setNotes] = useState('');
    const [alert, setAlert] = useState(null);

    // Filter verifications based on tab locally since we have global data
    const filteredVerifications = verifications.filter(v =>
        selectedTab === 'all' ? true : v.status === selectedTab
    );

    // No local polling needed - handled by context

    const handleDecision = async (verificationId, decision) => {
        setProcessingId(verificationId);
        try {
            const formData = new FormData();
            formData.append('decision', decision);
            formData.append('admin_id', 'admin');
            formData.append('notes', notes);

            await axios.post(`${API_BASE_URL}/verify-detection/${verificationId}`, formData);

            setAlert({
                type: 'success',
                message: `Detection ${decision === 'confirm' ? 'confirmed' : 'rejected'} successfully!`
            });

            setNotes('');
            setSelectedVerification(null);
            fetchVerifications();
            fetchStats();

            setTimeout(() => setAlert(null), 3000);
        } catch (error) {
            console.error('Error processing decision:', error);
            setAlert({
                type: 'error',
                message: 'Failed to process decision. Please try again.'
            });
            setTimeout(() => setAlert(null), 3000);
        } finally {
            setProcessingId(null);
        }
    };

    const openImageDialog = (imagePath) => {
        setSelectedImage(`http://localhost:8000${imagePath}`);
        setImageDialogOpen(true);
    };

    const getStatusBadge = (status) => {
        const variants = {
            pending: 'bg-yellow-100 text-yellow-800 border-yellow-300',
            confirmed: 'bg-green-100 text-green-800 border-green-300',
            rejected: 'bg-red-100 text-red-800 border-red-300'
        };
        return variants[status] || 'bg-gray-100 text-gray-800';
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case 'pending': return <Clock className="h-4 w-4" />;
            case 'confirmed': return <CheckCircle className="h-4 w-4" />;
            case 'rejected': return <XCircle className="h-4 w-4" />;
            default: return null;
        }
    };

    const formatTimestamp = (timestamp) => {
        return new Date(timestamp).toLocaleString();
    };

    return (
        <div className="space-y-6">
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center justify-between"
            >
                <div>
                    <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                        🛡️ Admin Verification Dashboard
                    </h1>
                    <p className="text-gray-600 mt-2">Review and verify dual-model gun detections</p>
                </div>
            </motion.div>

            {alert && (
                <Alert className={alert.type === 'success' ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}>
                    <AlertDescription className={alert.type === 'success' ? 'text-green-800' : 'text-red-800'}>
                        {alert.message}
                    </AlertDescription>
                </Alert>
            )}

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="bg-gradient-to-br from-purple-500 to-indigo-600 text-white border-0">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-white text-lg">Pending</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-4xl font-bold">{stats.pending}</div>
                    </CardContent>
                </Card>
                <Card className="bg-gradient-to-br from-pink-500 to-rose-600 text-white border-0">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-white text-lg">Confirmed</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-4xl font-bold">{stats.confirmed}</div>
                    </CardContent>
                </Card>
                <Card className="bg-gradient-to-br from-blue-500 to-cyan-600 text-white border-0">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-white text-lg">Rejected</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-4xl font-bold">{stats.rejected}</div>
                    </CardContent>
                </Card>
                <Card className="bg-gradient-to-br from-orange-500 to-yellow-600 text-white border-0">
                    <CardHeader className="pb-2">
                        <CardTitle className="text-white text-lg">Total</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-4xl font-bold">{stats.total}</div>
                    </CardContent>
                </Card>
            </div>

            {/* Tabs */}
            <Tabs value={selectedTab} onValueChange={setSelectedTab}>
                <TabsList className="grid w-full grid-cols-4">
                    <TabsTrigger value="pending" className="relative">
                        Pending
                        {stats.pending > 0 && (
                            <Badge className="ml-2 bg-yellow-500">{stats.pending}</Badge>
                        )}
                    </TabsTrigger>
                    <TabsTrigger value="confirmed">Confirmed</TabsTrigger>
                    <TabsTrigger value="rejected">Rejected</TabsTrigger>
                    <TabsTrigger value="all">All</TabsTrigger>
                </TabsList>
            </Tabs>

            {/* Content - Rendered directly to avoid TabsContent visibility issues */}
            <div className="mt-6">
                {loading ? (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 className="h-12 w-12 animate-spin text-blue-500" />
                    </div>
                ) : verifications.length === 0 ? (
                    <Alert>
                        <AlertDescription>No verifications found for this status.</AlertDescription>
                    </Alert>
                ) : (
                    <div className="space-y-4">
                        {filteredVerifications.map((verification) => (
                            <Card key={verification.id} className={selectedVerification?.id === verification.id ? 'border-blue-500 border-2' : ''}>
                                <CardContent className="pt-6">
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        {/* Images */}
                                        <div>
                                            <h3 className="font-semibold mb-3">Detection Images</h3>
                                            <div className="grid grid-cols-3 gap-2">
                                                {[
                                                    { path: verification.annotated_image_path, label: 'Annotated' },
                                                    { path: verification.full_frame_path, label: 'Full Frame' },
                                                    { path: verification.cropped_region_path, label: 'Cropped' }
                                                ].map((img, idx) => (
                                                    <div key={idx} className="relative group cursor-pointer" onClick={() => openImageDialog(img.path)}>
                                                        <img
                                                            src={`http://localhost:8000${img.path}`}
                                                            alt={img.label}
                                                            className="w-full h-24 object-cover rounded-lg border-2 border-gray-200 group-hover:border-blue-500 transition-all"
                                                            onError={(e) => {
                                                                e.target.onerror = null;
                                                                e.target.src = 'https://via.placeholder.com/150?text=Image+Not+Found';
                                                            }}
                                                        />
                                                        <div className="absolute top-1 left-1">
                                                            <Badge className="text-xs">{img.label}</Badge>
                                                        </div>
                                                        <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                                            <ZoomIn className="h-4 w-4 text-white drop-shadow-lg" />
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Details */}
                                        <div>
                                            <div className="flex justify-between items-start mb-4">
                                                <h3 className="font-semibold">Detection Details</h3>
                                                <Badge className={`${getStatusBadge(verification.status)} flex items-center gap-1`}>
                                                    {getStatusIcon(verification.status)}
                                                    {verification.status.toUpperCase()}
                                                </Badge>
                                            </div>

                                            <div className="space-y-2 text-sm mb-4">
                                                <div className="flex items-center gap-2">
                                                    <Camera className="h-4 w-4 text-gray-500" />
                                                    <span className="text-gray-600">
                                                        {verification.camera_name} ({verification.camera_id})
                                                    </span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <Clock className="h-4 w-4 text-gray-500" />
                                                    <span className="text-gray-600">{formatTimestamp(verification.timestamp)}</span>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <MapPin className="h-4 w-4 text-gray-500" />
                                                    <span className="text-gray-600">
                                                        {verification.location?.lat?.toFixed(4) || 0}, {verification.location?.lng?.toFixed(4) || 0}
                                                    </span>
                                                </div>
                                            </div>

                                            <div className="bg-gray-50 p-3 rounded-lg mb-4">
                                                <h4 className="font-semibold text-sm mb-2">Model Confidence Scores</h4>
                                                <div className="space-y-2">
                                                    <div className="flex justify-between items-center">
                                                        <span className="text-sm">🔴 best.pt (Primary):</span>
                                                        <Badge className="bg-blue-100 text-blue-800">
                                                            {(verification.best_pt_confidence * 100).toFixed(1)}%
                                                        </Badge>
                                                    </div>
                                                    <div className="flex justify-between items-center">
                                                        <span className="text-sm">🟢 best (1).pt (Verify):</span>
                                                        <Badge className="bg-green-100 text-green-800">
                                                            {(verification.best1_pt_confidence * 100).toFixed(1)}%
                                                        </Badge>
                                                    </div>
                                                </div>
                                            </div>

                                            {verification.status === 'pending' && (
                                                <div className="space-y-3">
                                                    <Textarea
                                                        placeholder="Add notes (optional)"
                                                        value={selectedVerification?.id === verification.id ? notes : ''}
                                                        onChange={(e) => {
                                                            setNotes(e.target.value);
                                                            setSelectedVerification(verification);
                                                        }}
                                                        rows={2}
                                                    />
                                                    <div className="flex gap-2">
                                                        <Button
                                                            onClick={() => handleDecision(verification.id, 'confirm')}
                                                            disabled={processingId === verification.id}
                                                            className="flex-1 bg-green-600 hover:bg-green-700"
                                                        >
                                                            {processingId === verification.id ? (
                                                                <Loader2 className="h-4 w-4 animate-spin" />
                                                            ) : (
                                                                <>
                                                                    <CheckCircle className="h-4 w-4 mr-2" />
                                                                    CONFIRM
                                                                </>
                                                            )}
                                                        </Button>
                                                        <Button
                                                            onClick={() => handleDecision(verification.id, 'reject')}
                                                            disabled={processingId === verification.id}
                                                            className="flex-1 bg-red-600 hover:bg-red-700"
                                                            variant="destructive"
                                                        >
                                                            {processingId === verification.id ? (
                                                                <Loader2 className="h-4 w-4 animate-spin" />
                                                            ) : (
                                                                <>
                                                                    <XCircle className="h-4 w-4 mr-2" />
                                                                    FALSE ALARM
                                                                </>
                                                            )}
                                                        </Button>
                                                    </div>
                                                </div>
                                            )}

                                            {verification.admin_decision && (
                                                <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                                                    <h4 className="font-semibold text-sm mb-2">Admin Decision</h4>
                                                    <div className="space-y-1 text-sm">
                                                        <div><strong>Decision:</strong> {verification.admin_decision.decision.toUpperCase()}</div>
                                                        <div><strong>Admin:</strong> {verification.admin_decision.admin_id}</div>
                                                        <div><strong>Time:</strong> {formatTimestamp(verification.admin_decision.timestamp)}</div>
                                                        {verification.admin_decision.notes && (
                                                            <div><strong>Notes:</strong> {verification.admin_decision.notes}</div>
                                                        )}
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                )}

                {/* Load More Button */}
                {hasMore && !loading && (
                    <div className="flex justify-center mt-6 mb-8">
                        <Button
                            variant="outline"
                            onClick={() => fetchVerifications(true)}
                            className="w-full md:w-1/3 border-blue-200 text-blue-600 hover:bg-blue-50"
                        >
                            Load More Verifications
                        </Button>
                    </div>
                )}
            </div>

            {/* Image Zoom Dialog */}
            <Dialog open={imageDialogOpen} onOpenChange={setImageDialogOpen}>
                <DialogContent className="max-w-4xl">
                    <DialogHeader>
                        <DialogTitle>Image Viewer</DialogTitle>
                    </DialogHeader>
                    {selectedImage && (
                        <img
                            src={selectedImage}
                            alt="Zoomed"
                            className="w-full h-auto rounded-lg"
                        />
                    )}
                </DialogContent>
            </Dialog>
        </div>
    );
}

export default AdminVerification;
