import React, { useState, useEffect } from "react";
import "@/App.css";
import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import axios from "axios";
import Dashboard from "./pages/Dashboard";
import MapView from "./pages/MapView";
import Detections from "./pages/Detections";
import Upload from "./pages/Upload";

import LiveMonitoring from "./pages/LiveMonitoring";
import Cameras from "./pages/Cameras";
import Settings from "./pages/Settings";
import Analytics from "./pages/Analytics";
import AdminVerification from "./pages/AdminVerification";
import { Toaster } from "react-hot-toast";
import {
  Shield, Map, Upload as UploadIcon, Link2, AlertCircle,
  Video, Camera, BarChart3, Settings as SettingsIcon, ChevronLeft, ChevronRight,
  CheckCircle
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// Backend URL
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

function SidebarNavigation() {
  const location = useLocation();
  const [detectionCount, setDetectionCount] = useState(0);
  const [pendingVerifications, setPendingVerifications] = useState(0);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    const fetchNotifications = async () => {
      try {
        const response = await axios.get(`${BACKEND_URL}/api/detections?limit=5`);
        setDetectionCount(response.data.length);

        // Fetch pending verifications count
        const verifyResponse = await axios.get(`${BACKEND_URL}/api/verification-stats`);
        setPendingVerifications(verifyResponse.data.pending || 0);
      } catch (error) {
        console.error("Error fetching notifications:", error);
      }
    };

    fetchNotifications();
    const interval = setInterval(fetchNotifications, 30000); // Reduced frequency to 30s
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { path: "/", label: "Dashboard", icon: Shield },
    { path: "/live", label: "Live Monitor", icon: Video },
    { path: "/cameras", label: "Cameras", icon: Camera },
    { path: "/admin-verify", label: "Admin Verify", icon: CheckCircle, badge: pendingVerifications },
    { path: "/map", label: "Map View", icon: Map },
    { path: "/detections", label: "Detections", icon: AlertCircle, badge: detectionCount },
    { path: "/analytics", label: "Analytics", icon: BarChart3 },
    { path: "/upload", label: "Upload", icon: UploadIcon },

    { path: "/settings", label: "Settings", icon: SettingsIcon },
  ];

  return (
    <div className={`fixed left-0 top-0 h-screen bg-white border-r border-gray-200 shadow-lg transition-all duration-300 z-50 ${collapsed ? 'w-20' : 'w-64'}`}>
      {/* Logo */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          {!collapsed && (
            <Link to="/" className="flex items-center space-x-3">
              <div className="bg-gradient-to-br from-red-500 to-orange-600 p-2.5 rounded-xl shadow-lg">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div>
                <span className="text-lg font-bold bg-gradient-to-r from-red-600 to-orange-600 bg-clip-text text-transparent">
                  SecureView
                </span>
                <p className="text-xs text-gray-500">Alert System</p>
              </div>
            </Link>
          )}
          {collapsed && (
            <div className="bg-gradient-to-br from-red-500 to-orange-600 p-2.5 rounded-xl shadow-lg mx-auto">
              <Shield className="h-6 w-6 text-white" />
            </div>
          )}
        </div>
      </div>

      {/* Navigation Items */}
      <nav className="p-3 space-y-1 overflow-y-auto h-[calc(100vh-140px)]">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200 ${isActive
                ? "bg-gradient-to-r from-red-500 to-orange-500 text-white shadow-lg"
                : "text-gray-700 hover:bg-gray-100"
                }`}
              title={collapsed ? item.label : ""}
            >
              <Icon className="h-5 w-5 flex-shrink-0" />
              {!collapsed && (
                <>
                  <span className="text-sm font-medium flex-1">{item.label}</span>
                  {item.badge > 0 && (
                    <span className="bg-white text-red-600 text-xs px-2 py-0.5 rounded-full font-bold">
                      {item.badge}
                    </span>
                  )}
                </>
              )}
              {collapsed && item.badge > 0 && (
                <span className="absolute right-2 top-2 h-2 w-2 bg-red-500 rounded-full"></span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Collapse Button */}
      <div className="absolute bottom-4 left-0 right-0 px-3">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center px-4 py-3 bg-gray-100 hover:bg-gray-200 rounded-xl transition-colors"
        >
          {collapsed ? <ChevronRight className="h-5 w-5" /> : <ChevronLeft className="h-5 w-5" />}
        </button>
      </div>
    </div>
  );
}

function AppContent() {
  const location = useLocation();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-50">
      <SidebarNavigation />

      {/* Main Content */}
      <div className={`flex-1 transition-all duration-300 ${sidebarCollapsed ? 'ml-20' : 'ml-64'}`}>
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0.8 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0.8 }}
          transition={{ duration: 0.15 }} // Reduced from 0.3s to 0.15s for faster transitions
          className="p-8 w-full"
        >
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/map" element={<MapView />} />
            <Route path="/live" element={<LiveMonitoring />} />
            <Route path="/cameras" element={<Cameras />} />
            <Route path="/admin-verify" element={<AdminVerification />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/detections" element={<Detections />} />
            <Route path="/analytics" element={<Analytics />} />

            <Route path="/settings" element={<Settings />} />
          </Routes>
        </motion.div>
      </div>

      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#fff',
            color: '#363636',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
          },
          success: {
            iconTheme: {
              primary: '#10b981',
              secondary: '#fff',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef4444',
              secondary: '#fff',
            },
          },
        }}
      />
    </div>
  );
}

import { MonitoringProvider } from "./context/MonitoringContext";

function App() {
  return (
    <BrowserRouter>
      <MonitoringProvider>
        <AppContent />
      </MonitoringProvider>
    </BrowserRouter>
  );
}

export default App;