import React, { useState, useCallback } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Upload as UploadIcon, X, CheckCircle, AlertCircle, Image as ImageIcon, FileImage, Shield, Crosshair, Users, Flame } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import toast from "react-hot-toast";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

const Upload = () => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [detectionMode, setDetectionMode] = useState('all'); // 'all', 'best.pt', 'best (1).pt', 'thermal', 'violence'
  const [droneConfidence, setDroneConfidence] = useState(0.50); // Default 50% for drone detection

  const detectionModes = [
    { id: 'all', name: 'All Models (Comprehensive)', icon: Shield, description: 'Run all detection models' },
    { id: 'gun-fusion', name: 'Gun Fusion Detection', icon: Crosshair, description: 'Fused thermal + best.pt + best(1).pt' },
    { id: 'drone', name: 'Drone People Detection', icon: Users, description: 'Aerial people detection' },
    { id: 'topview', name: 'Top-View Human Detection', icon: Users, description: 'Overhead human detection' },
    { id: 'thermal-human', name: 'Thermal Human Detection', icon: Flame, description: 'Thermal imaging humans' },
    { id: 'violence', name: 'Violence Detection', icon: AlertCircle, description: 'Scene violence classification' }
  ];

  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.map(file => ({
      file,
      preview: URL.createObjectURL(file),
      progress: 0,
      status: 'pending', // pending, uploading, success, error
      result: null
    }));
    setFiles(prev => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv']
    },
    multiple: true
  });

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const uploadFiles = async () => {
    setUploading(true);

    for (let i = 0; i < files.length; i++) {
      if (files[i].status !== 'pending') continue;

      const formData = new FormData();
      formData.append('file', files[i].file);

      // Check if file is a video
      const isVideo = files[i].file.type.startsWith('video/');

      // Determine endpoint based on file type and mode
      let endpoint = `${API}/test-upload`;

      if (isVideo && detectionMode === 'drone') {
        endpoint = `${API}/detect-drone-video`;
        formData.append('conf_threshold', droneConfidence.toString());
      } else if (isVideo && detectionMode === 'topview') {
        // Top-view mode supports video - use standard endpoint
        endpoint = `${API}/test-upload`;
        formData.append('model_type', 'topview');
      } else if (isVideo && detectionMode === 'gun-fusion') {
        // Gun-fusion mode supports video - use standard endpoint
        endpoint = `${API}/test-upload`;
        formData.append('model_type', 'gun-fusion');
      } else if (isVideo) {
        // Only drone, topview, and gun-fusion modes support video
        toast.error(`${files[i].file.name}: Video upload only supported in Drone, Top-View, and Gun Fusion modes`);
        setFiles(prev => prev.map((f, idx) =>
          idx === i ? { ...f, status: 'error', progress: 0 } : f
        ));
        continue;
      } else {
        formData.append('model_type', detectionMode);
      }

      try {
        // Update status to uploading
        setFiles(prev => prev.map((f, idx) =>
          idx === i ? { ...f, status: 'uploading', progress: 50 } : f
        ));

        const response = await axios.post(endpoint, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setFiles(prev => prev.map((f, idx) =>
              idx === i ? { ...f, progress: percentCompleted } : f
            ));
          }
        });

        // Update status to success
        setFiles(prev => prev.map((f, idx) =>
          idx === i ? { ...f, status: 'success', progress: 100, result: response.data } : f
        ));

        toast.success(`${files[i].file.name} processed successfully!`);
      } catch (error) {
        console.error("Upload error:", error);
        setFiles(prev => prev.map((f, idx) =>
          idx === i ? { ...f, status: 'error', progress: 0 } : f
        ));
        toast.error(`Failed to process ${files[i].file.name}`);
      }
    }

    setUploading(false);
  };

  const clearAll = () => {
    files.forEach(f => URL.revokeObjectURL(f.preview));
    setFiles([]);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Upload & Analyze
        </h1>
        <p className="text-gray-600 mt-2">Upload images for comprehensive security analysis</p>
      </div>

      {/* Detection Mode Selector */}
      <Card className="bg-white border-gray-200 shadow-lg">
        <CardHeader>
          <CardTitle className="text-gray-900">Detection Mode</CardTitle>
          <CardDescription>Choose which models to run</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {detectionModes.map((mode) => {
              const Icon = mode.icon;
              return (
                <button
                  key={mode.id}
                  onClick={() => setDetectionMode(mode.id)}
                  className={`flex items-center p-4 rounded-xl border-2 transition-all ${detectionMode === mode.id
                    ? 'border-blue-500 bg-blue-50 shadow-md'
                    : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                    }`}
                >
                  <div className={`p-2 rounded-full mr-3 ${detectionMode === mode.id ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-500'
                    }`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className="text-left">
                    <h3 className={`font-semibold ${detectionMode === mode.id ? 'text-blue-700' : 'text-gray-900'
                      }`}>
                      {mode.name}
                    </h3>
                  </div>
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Drone Confidence Slider (only show in drone mode) */}
      {detectionMode === 'drone' && (
        <Card className="bg-white border-gray-200 shadow-lg">
          <CardHeader>
            <CardTitle className="text-gray-900">Drone Detection Confidence</CardTitle>
            <CardDescription>Adjust sensitivity (lower = more detections, higher = more accurate)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Confidence Threshold:</span>
                <span className="text-lg font-bold text-blue-600">{(droneConfidence * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.90"
                step="0.05"
                value={droneConfidence}
                onChange={(e) => setDroneConfidence(parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>10% (More sensitive)</span>
                <span>90% (More accurate)</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Upload Zone */}
      <Card className="bg-white border-gray-200 shadow-lg">
        <CardHeader>
          <CardTitle className="text-gray-900">Upload Images</CardTitle>
          <CardDescription>Drag and drop images or click to browse</CardDescription>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${isDragActive
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400 hover:bg-gray-50'
              }`}
          >
            <input {...getInputProps()} />
            <motion.div
              animate={{ y: isDragActive ? -10 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <UploadIcon className={`h-16 w-16 mx-auto mb-4 ${isDragActive ? 'text-blue-500' : 'text-gray-400'}`} />
              {isDragActive ? (
                <p className="text-lg font-medium text-blue-600">Drop the files here...</p>
              ) : (
                <>
                  <p className="text-lg font-medium text-gray-900 mb-2">
                    Drag & drop images or videos here, or click to select
                  </p>
                  <p className="text-sm text-gray-500">
                    Images: JPG, PNG, GIF, BMP | Videos: MP4, AVI, MOV, MKV
                  </p>
                </>
              )}
            </motion.div>
          </div>
        </CardContent>
      </Card >

      {/* File List */}
      {
        files.length > 0 && (
          <Card className="bg-white border-gray-200 shadow-lg">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-gray-900">Selected Files ({files.length})</CardTitle>
                  <CardDescription>Review and process your uploads</CardDescription>
                </div>
                <div className="flex space-x-2">
                  <Button
                    onClick={uploadFiles}
                    disabled={uploading || files.every(f => f.status !== 'pending')}
                    className="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700"
                  >
                    <UploadIcon className="h-4 w-4 mr-2" />
                    {uploading ? 'Processing...' : 'Process All'}
                  </Button>
                  <Button
                    onClick={clearAll}
                    variant="outline"
                    disabled={uploading}
                    className="border-gray-300"
                  >
                    <X className="h-4 w-4 mr-2" />
                    Clear All
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <AnimatePresence>
                  {files.map((fileItem, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -100 }}
                      className="bg-gray-50 rounded-xl border border-gray-200 overflow-hidden"
                    >
                      <div className="p-4 flex items-center space-x-4">
                        {/* Preview */}
                        <div className="relative w-20 h-20 rounded-lg overflow-hidden bg-gray-200 flex-shrink-0">
                          {fileItem.file.type.startsWith('video/') ? (
                            <video
                              src={fileItem.preview}
                              className="w-full h-full object-cover"
                              muted
                            />
                          ) : (
                            <img
                              src={fileItem.preview}
                              alt={fileItem.file.name}
                              className="w-full h-full object-cover"
                            />
                          )}
                        </div>

                        {/* File Info */}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {fileItem.file.name}
                          </p>
                          <p className="text-xs text-gray-500">
                            {(fileItem.file.size / 1024 / 1024).toFixed(2)} MB
                          </p>

                          {/* Progress Bar */}
                          {fileItem.status === 'uploading' && (
                            <div className="mt-2">
                              <Progress value={fileItem.progress} className="h-2" />
                            </div>
                          )}
                        </div>

                        {/* Status Icon */}
                        <div className="flex-shrink-0">
                          {fileItem.status === 'pending' && (
                            <FileImage className="h-6 w-6 text-gray-400" />
                          )}
                          {fileItem.status === 'uploading' && (
                            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                          )}
                          {fileItem.status === 'success' && (
                            <CheckCircle className="h-6 w-6 text-green-500" />
                          )}
                          {fileItem.status === 'error' && (
                            <AlertCircle className="h-6 w-6 text-red-500" />
                          )}
                        </div>

                        {/* Remove Button */}
                        <button
                          onClick={() => removeFile(index)}
                          disabled={uploading}
                          className="flex-shrink-0 p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
                        >
                          <X className="h-5 w-5" />
                        </button>
                      </div>

                      {/* Results Section */}
                      {fileItem.status === 'success' && fileItem.result && (
                        <div className="border-t border-gray-200 p-4 bg-white">
                          {/* Video Detection Results */}
                          {(fileItem.result.video_info || fileItem.result.video_output || fileItem.result.video_stats) ? (
                            <div className="space-y-4">

                              {/* Video Output & Stats */}
                              {(fileItem.result.video_output || fileItem.result.video_stats) && (
                                <div className="mb-4">
                                  <h4 className="font-semibold text-gray-900 mb-2">Processed Video</h4>

                                  {fileItem.result.video_output && (
                                    <div className="mb-4">
                                      <a
                                        href={`${BACKEND_URL}${fileItem.result.video_output}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                                      >
                                        <FileImage className="w-4 h-4 mr-2" />
                                        Download Annotated Video
                                      </a>
                                    </div>
                                  )}

                                  {fileItem.result.video_stats && (
                                    <div className="grid grid-cols-2 gap-2 text-sm">
                                      <div className="bg-gray-50 p-2 rounded">
                                        <span className="text-gray-600">Resolution:</span>{' '}
                                        <span className="font-medium">{fileItem.result.video_stats.resolution}</span>
                                      </div>
                                      <div className="bg-gray-50 p-2 rounded">
                                        <span className="text-gray-600">FPS:</span>{' '}
                                        <span className="font-medium">{fileItem.result.video_stats.fps}</span>
                                      </div>
                                      <div className="bg-gray-50 p-2 rounded">
                                        <span className="text-gray-600">Total Frames:</span>{' '}
                                        <span className="font-medium">{fileItem.result.video_stats.total_frames}</span>
                                      </div>
                                      <div className="bg-gray-50 p-2 rounded">
                                        <span className="text-gray-600">Detections:</span>{' '}
                                        <span className="font-medium">{fileItem.result.video_stats.total_detections}</span>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}

                              {/* Legacy Drone Video Info */}
                              {fileItem.result.video_info && (
                                <div>
                                  <h4 className="font-semibold text-gray-900 mb-2">Video Information</h4>
                                  <div className="grid grid-cols-2 gap-2 text-sm">
                                    <div className="bg-gray-50 p-2 rounded">
                                      <span className="text-gray-600">Duration:</span>{' '}
                                      <span className="font-medium">{fileItem.result.video_info.duration_seconds}s</span>
                                    </div>
                                    <div className="bg-gray-50 p-2 rounded">
                                      <span className="text-gray-600">FPS:</span>{' '}
                                      <span className="font-medium">{fileItem.result.video_info.fps}</span>
                                    </div>
                                    <div className="bg-gray-50 p-2 rounded">
                                      <span className="text-gray-600">Total Frames:</span>{' '}
                                      <span className="font-medium">{fileItem.result.video_info.total_frames}</span>
                                    </div>
                                    <div className="bg-gray-50 p-2 rounded">
                                      <span className="text-gray-600">Processed:</span>{' '}
                                      <span className="font-medium">{fileItem.result.video_info.processed_frames}</span>
                                    </div>
                                  </div>
                                </div>
                              )}

                              {/* Detection Summary (if available) */}
                              {fileItem.result.detection_summary && (
                                <div>
                                  <h4 className="font-semibold text-gray-900 mb-2">Detection Summary</h4>
                                  <div className="grid grid-cols-2 gap-2 text-sm">
                                    <div className="bg-blue-50 p-3 rounded border border-blue-200">
                                      <div className="text-blue-600 text-xs mb-1">Total Detections</div>
                                      <div className="text-2xl font-bold text-blue-700">
                                        {fileItem.result.detection_summary.total_detections}
                                      </div>
                                    </div>
                                    <div className="bg-green-50 p-3 rounded border border-green-200">
                                      <div className="text-green-600 text-xs mb-1">Max People/Frame</div>
                                      <div className="text-2xl font-bold text-green-700">
                                        {fileItem.result.detection_summary.max_people_in_single_frame}
                                      </div>
                                    </div>
                                    <div className="bg-purple-50 p-3 rounded border border-purple-200">
                                      <div className="text-purple-600 text-xs mb-1">Frames with People</div>
                                      <div className="text-2xl font-bold text-purple-700">
                                        {fileItem.result.detection_summary.frames_with_people}
                                      </div>
                                    </div>
                                    <div className="bg-orange-50 p-3 rounded border border-orange-200">
                                      <div className="text-orange-600 text-xs mb-1">Avg People/Frame</div>
                                      <div className="text-2xl font-bold text-orange-700">
                                        {fileItem.result.detection_summary.avg_people_per_frame}
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              )}

                              {/* Annotated Sample Frames */}
                              {fileItem.result.annotated_samples && fileItem.result.annotated_samples.length > 0 && (
                                <div>
                                  <h4 className="font-semibold text-gray-900 mb-2">
                                    Detection Samples (Frames with Bounding Boxes)
                                  </h4>
                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    {fileItem.result.annotated_samples.map((sample, idx) => (
                                      <div key={idx} className="border border-gray-200 rounded-lg overflow-hidden">
                                        <img
                                          src={`${BACKEND_URL}${sample.image_url}`}
                                          alt={`Frame ${sample.frame_number}`}
                                          className="w-full h-auto"
                                        />
                                        <div className="p-2 bg-gray-50 text-xs text-gray-600">
                                          <div className="flex justify-between">
                                            <span>Frame {sample.frame_number} ({sample.timestamp}s)</span>
                                            <span className="font-semibold text-blue-600">
                                              {sample.detections} {sample.detections === 1 ? 'person' : 'people'}
                                            </span>
                                          </div>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {/* Detections Timeline (first 10) */}
                              {fileItem.result.detections && fileItem.result.detections.length > 0 && (
                                <div>
                                  <h4 className="font-semibold text-gray-900 mb-2">
                                    Detections Timeline (showing first 10)
                                  </h4>
                                  <div className="space-y-1 max-h-48 overflow-y-auto">
                                    {fileItem.result.detections.slice(0, 10).map((det, i) => (
                                      <div key={i} className="flex items-center justify-between p-2 bg-gray-50 rounded text-sm">
                                        <span className="text-gray-600">
                                          Frame {det.frame_number}
                                        </span>
                                        <span className="font-medium text-blue-600">
                                          {det.class} - {(det.confidence * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                  {fileItem.result.detections.length > 10 && (
                                    <p className="text-xs text-gray-500 mt-2">
                                      + {fileItem.result.detections.length - 10} more detections
                                    </p>
                                  )}
                                </div>
                              )}
                            </div>
                          ) : (
                            /* Image Detection Results */
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              {/* Annotated Image */}
                              <div className="rounded-lg overflow-hidden border border-gray-200">
                                <img
                                  src={`${BACKEND_URL}${fileItem.result.annotated_image_url}`}
                                  alt="Analyzed Result"
                                  className="w-full h-auto"
                                />
                              </div>

                              {/* Detections List */}
                              <div>
                                <h4 className="font-semibold text-gray-900 mb-2">Detections</h4>
                                {fileItem.result.detections.length === 0 ? (
                                  <p className="text-sm text-green-600">✓ No threats detected</p>
                                ) : (
                                  <div className="space-y-2 max-h-60 overflow-y-auto">
                                    {fileItem.result.detections.map((det, i) => (
                                      <div key={i} className="flex items-center justify-between p-2 bg-red-50 rounded border border-red-100">
                                        <span className="text-sm font-medium text-red-800 capitalize">
                                          {det.class}
                                        </span>
                                        <span className="text-xs bg-red-200 text-red-800 px-2 py-1 rounded-full">
                                          {(det.confidence * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                )}
                                <div className="mt-4 text-xs text-gray-500">
                                  Processing Time: {(fileItem.result.processing_time * 1000).toFixed(0)}ms
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </CardContent>
          </Card>
        )
      }
    </div >
  );
};

export default Upload;