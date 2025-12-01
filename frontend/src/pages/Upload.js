import React, { useState, useCallback } from "react";
import axios from "axios";
import { useDropzone } from "react-dropzone";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Upload as UploadIcon, X, CheckCircle, AlertCircle, Image as ImageIcon, FileImage } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import toast from "react-hot-toast";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

const Upload = () => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState([]);
  const [detectionMode, setDetectionMode] = useState('normal'); // 'normal' or 'thermal'

  const detectionModes = [
    { id: 'normal', name: 'Normal (Both Weapon Models)', icon: Shield },
    { id: 'model1', name: 'Weapon Detection (Best.pt)', icon: Crosshair },
    { id: 'model2', name: 'Weapon Detection (Best 1.pt)', icon: Crosshair },
    { id: 'people', name: 'People Counting', icon: Users },
    { id: 'thermal', name: 'Thermal Detection', icon: Flame }
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
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp']
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
      formData.append('camera_id', 'upload');
      formData.append('camera_name', 'Manual Upload');
      formData.append('latitude', '0');
      formData.append('longitude', '0');
      formData.append('detection_mode', detectionMode); // Add detection mode

      try {
        // Update status to uploading
        setFiles(prev => prev.map((f, idx) =>
          idx === i ? { ...f, status: 'uploading', progress: 50 } : f
        ));

        const response = await axios.post(`${API}/upload`, formData, {
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
    setResults([]);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Upload & Analyze
        </h1>
        <p className="text-gray-600 mt-2">Upload images for weapon detection analysis</p>
      </div>

      {/* Detection Mode Selector */}
      <Card className="bg-white border-gray-200 shadow-lg">
        <CardHeader>
          <CardTitle className="text-gray-900">Detection Mode</CardTitle>
          <CardDescription>Choose which detection method to use</CardDescription>
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
                    Drag & drop images here, or click to select
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports: JPG, PNG, GIF, BMP (Max 10MB per file)
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
              <div className="space-y-3">
                <AnimatePresence>
                  {files.map((fileItem, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -100 }}
                      className="flex items-center space-x-4 p-4 bg-gray-50 rounded-xl border border-gray-200"
                    >
                      {/* Preview */}
                      <div className="relative w-20 h-20 rounded-lg overflow-hidden bg-gray-200 flex-shrink-0">
                        <img
                          src={fileItem.preview}
                          alt={fileItem.file.name}
                          className="w-full h-full object-cover"
                        />
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

                        {/* Results */}
                        {fileItem.status === 'success' && fileItem.result && (
                          <div className="mt-2">
                            {fileItem.result.detections && fileItem.result.detections.length > 0 ? (
                              <p className="text-xs text-red-600 font-medium">
                                ⚠️ {fileItem.result.detections.length} threat(s) detected
                              </p>
                            ) : (
                              <p className="text-xs text-green-600 font-medium">
                                ✓ No threats detected
                              </p>
                            )}
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
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </CardContent>
          </Card>
        )
      }

      {/* Info Card */}
      <Card className="bg-gradient-to-br from-blue-50 to-purple-50 border-blue-200">
        <CardContent className="pt-6">
          <div className="flex items-start space-x-4">
            <div className="bg-blue-500 p-3 rounded-xl">
              <ImageIcon className="h-6 w-6 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">How it works</h3>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>• Upload one or multiple images for analysis</li>
                <li>• Our YOLOv8 model will detect weapons (pistols, knives)</li>
                <li>• Results are stored in the blockchain for evidence</li>
                <li>• View detailed detection reports in the Detections page</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div >
  );
};

export default Upload;