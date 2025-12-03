import React, { useState, useEffect } from "react";
import axios from "axios";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Shield, AlertTriangle, Link2, TrendingUp, MapPin, Camera, Video, Activity, Users } from "lucide-react";
import { motion } from "framer-motion";
import { LineChart, Line, ResponsiveContainer } from "recharts";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const [stats, setStats] = useState({
    detections: 0,
    blockchain: 0,
    cameras: 0,
    activeCameras: 0,
    pendingVerifications: 0,
    recentDetections: [],
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Use allSettled to prevent one failure from blocking all data
        const results = await Promise.allSettled([
          axios.get(`${API}/detections`),
          axios.get(`${API}/blockchain`),
          axios.get(`${API}/cameras`),
          axios.get(`${API}/verification-stats`),
        ]);

        const detections = results[0].status === 'fulfilled' ? results[0].value : { data: [] };
        const blockchain = results[1].status === 'fulfilled' ? results[1].value : { data: [] };
        const cameras = results[2].status === 'fulfilled' ? results[2].value : { data: { cameras: [], count: 0 } };
        const verifyStats = results[3].status === 'fulfilled' ? results[3].value : { data: { pending: 0 } };

        const activeCamerasCount = cameras.data.cameras?.filter(cam => cam.status === 'active').length || 0;

        setStats({
          detections: detections.data.length || 0,
          blockchain: blockchain.data.length || 0,
          cameras: cameras.data.count || 0,
          activeCameras: activeCamerasCount,
          pendingVerifications: verifyStats.data.pending || 0,
          recentDetections: detections.data.slice(0, 5) || [],
        });
        setLoading(false);
      } catch (error) {
        console.error("Error fetching dashboard data:", error);
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000); // Increased from 5s to 10s
    return () => clearInterval(interval);
  }, []);

  const statCards = [
    {
      title: "Total Detections",
      value: stats.detections,
      icon: AlertTriangle,
      color: "from-red-500 to-orange-500",
      bgColor: "bg-red-50",
      textColor: "text-red-600",
      trend: "+12.5%",
      trendUp: true,
    },
    {
      title: "Pending Verifications",
      value: stats.pendingVerifications,
      icon: Shield,
      color: "from-yellow-500 to-amber-500",
      bgColor: "bg-yellow-50",
      textColor: "text-yellow-600",
      trend: "Needs Review",
      trendUp: false,
    },
    {
      title: "Total Cameras",
      value: stats.cameras,
      icon: Camera,
      color: "from-blue-500 to-cyan-500",
      bgColor: "bg-blue-50",
      textColor: "text-blue-600",
      trend: "+2",
      trendUp: true,
    },
    {
      title: "Active Cameras",
      value: stats.activeCameras,
      icon: Video,
      color: "from-green-500 to-emerald-500",
      bgColor: "bg-green-50",
      textColor: "text-green-600",
      trend: "Live",
      trendUp: true,
    },
    {
      title: "Blockchain Records",
      value: stats.blockchain,
      icon: Link2,
      color: "from-purple-500 to-pink-500",
      bgColor: "bg-purple-50",
      textColor: "text-purple-600",
      trend: "+8.2%",
      trendUp: true,
    },
  ];

  // Generate mini chart data
  const generateMiniChartData = () => {
    return Array.from({ length: 7 }, (_, i) => ({
      value: Math.floor(Math.random() * 20) + 10
    }));
  };

  return (
    <div className="space-y-8" data-testid="dashboard">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-red-600 to-orange-600 bg-clip-text text-transparent">
            Security Dashboard
          </h1>
          <p className="text-gray-600 mt-2">
            Real-time weapon detection with blockchain evidence storage
          </p>
        </div>
        <motion.div
          animate={{ rotate: [0, 10, -10, 0] }}
          transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
          className="bg-gradient-to-br from-red-500 to-orange-600 p-4 rounded-2xl shadow-lg"
        >
          <Shield className="h-12 w-12 text-white" />
        </motion.div>
      </motion.div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        {statCards.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card
                data-testid={`stat-${stat.title.toLowerCase().replace(/\s+/g, '-')}`}
                className="bg-white border-gray-200 hover:shadow-xl transition-all duration-300 overflow-hidden group"
              >
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium text-gray-600">{stat.title}</CardTitle>
                  <div className={`bg-gradient-to-br ${stat.color} p-2.5 rounded-xl shadow-lg group-hover:scale-110 transition-transform`}>
                    <Icon className="h-5 w-5 text-white" />
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex items-baseline justify-between">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: index * 0.1 + 0.2, type: "spring" }}
                      className="text-3xl font-bold text-gray-900"
                    >
                      {loading ? "--" : stat.value}
                    </motion.div>
                    <div className="flex items-center space-x-1">
                      <TrendingUp className={`h-4 w-4 ${stat.trendUp ? 'text-green-600' : 'text-red-600'}`} />
                      <span className={`text-sm font-medium ${stat.trendUp ? 'text-green-600' : 'text-red-600'}`}>
                        {stat.trend}
                      </span>
                    </div>
                  </div>
                  <div className="mt-4">
                    <ResponsiveContainer width="100%" height={40}>
                      <LineChart data={generateMiniChartData()}>
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke={stat.textColor.replace('text-', '#')}
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Recent Detections */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <Card className="bg-white border-gray-200 shadow-lg" data-testid="recent-detections">
          <CardHeader>
            <CardTitle className="text-gray-900 flex items-center text-xl">
              <AlertTriangle className="h-6 w-6 mr-2 text-red-500" />
              Recent Detections
            </CardTitle>
            <CardDescription className="text-gray-600">Latest threat detections from all cameras</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-500"></div>
              </div>
            ) : stats.recentDetections.length === 0 ? (
              <div className="text-center py-12">
                <Shield className="h-16 w-16 mx-auto mb-4 text-gray-300" />
                <p className="text-gray-500 font-medium">No detections yet</p>
                <p className="text-gray-400 text-sm mt-1">System is actively monitoring</p>
              </div>
            ) : (
              <div className="space-y-3">
                {stats.recentDetections.map((detection, index) => (
                  <motion.div
                    key={detection.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    data-testid={`detection-item-${index}`}
                    className="flex items-center space-x-4 p-4 bg-gradient-to-r from-gray-50 to-white rounded-xl border border-gray-200 hover:border-red-300 hover:shadow-md transition-all"
                  >
                    <div className="bg-gradient-to-br from-red-500 to-orange-500 p-3 rounded-xl shadow-lg">
                      <AlertTriangle className="h-6 w-6 text-white" />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900 capitalize text-lg">{detection.detection_type}</h4>
                      <div className="flex items-center space-x-4 mt-1">
                        <p className="text-sm text-gray-600">
                          Confidence: <span className="font-semibold text-red-600">{(detection.confidence * 100).toFixed(1)}%</span>
                        </p>
                        {detection.camera_name && (
                          <p className="text-sm text-blue-600 flex items-center">
                            <Camera className="h-3 w-3 mr-1" />
                            {detection.camera_name}
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-500 flex items-center justify-end">
                        <MapPin className="h-3 w-3 mr-1" />
                        {detection.location.lat.toFixed(4)}, {detection.location.lng.toFixed(4)}
                      </p>
                      <p className="text-xs text-gray-400 mt-1">
                        {new Date(detection.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* System Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <Card className="bg-white border-gray-200 shadow-lg">
          <CardHeader>
            <CardTitle className="text-gray-900 text-xl">System Status</CardTitle>
            <CardDescription>All systems operational</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                { label: "YOLOv8 Detection Engine", status: "Online", icon: Activity },
                { label: "Blockchain Network", status: "Active", icon: Link2 },
                { label: "Database Connection", status: "Connected", icon: Shield },
              ].map((system, index) => {
                const Icon = system.icon;
                return (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.7 + index * 0.1 }}
                    className="flex items-center justify-between p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="bg-green-500 p-2 rounded-lg">
                        <Icon className="h-5 w-5 text-white" />
                      </div>
                      <span className="text-sm font-medium text-gray-900">{system.label}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
                      <span className="text-sm font-semibold text-green-600">{system.status}</span>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};

export default Dashboard;