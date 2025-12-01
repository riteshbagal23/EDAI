import React, { useState, useEffect } from "react";
import axios from "axios";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { TrendingUp, TrendingDown, Activity, Calendar, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import toast from "react-hot-toast";

    const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

const Analytics = () => {
    const [detections, setDetections] = useState([]);
    const [loading, setLoading] = useState(true);
    const [timeRange, setTimeRange] = useState("7days");

    useEffect(() => {
        fetchDetections();
    }, [timeRange]);

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

    // Process data for charts
    const getDetectionsByType = () => {
        const types = {};
        detections.forEach(d => {
            types[d.detection_type] = (types[d.detection_type] || 0) + 1;
        });
        return Object.entries(types).map(([name, value]) => ({ name, value }));
    };

    const getDetectionsByDay = () => {
        const days = {};
        detections.forEach(d => {
            const date = new Date(d.timestamp).toLocaleDateString();
            days[date] = (days[date] || 0) + 1;
        });
        return Object.entries(days).map(([date, count]) => ({ date, count })).slice(-7);
    };

    const getDetectionsByHour = () => {
        const hours = Array(24).fill(0);
        detections.forEach(d => {
            const hour = new Date(d.timestamp).getHours();
            hours[hour]++;
        });
        return hours.map((count, hour) => ({ hour: `${hour}:00`, count }));
    };

    const COLORS = ['#ef4444', '#f97316', '#f59e0b', '#84cc16', '#06b6d4', '#8b5cf6'];

    const exportData = () => {
        const csvContent = "data:text/csv;charset=utf-8,"
            + "Type,Confidence,Camera,Location,Timestamp\n"
            + detections.map(d =>
                `${d.detection_type},${d.confidence},${d.camera_name},"${d.location.lat},${d.location.lng}",${d.timestamp}`
            ).join("\n");

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `detections_${new Date().toISOString()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        toast.success("Data exported successfully!");
    };

    const stats = {
        total: detections.length,
        pistols: detections.filter(d => d.detection_type === 'pistol').length,
        knives: detections.filter(d => d.detection_type === 'knife').length,
        avgConfidence: detections.length > 0
            ? (detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length * 100).toFixed(1)
            : 0
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                        Analytics Dashboard
                    </h1>
                    <p className="text-gray-600 mt-2">Comprehensive detection insights and trends</p>
                </div>
                <Button onClick={exportData} className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600">
                    <Download className="h-4 w-4 mr-2" />
                    Export Data
                </Button>
            </div>

            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                {[
                    { label: "Total Detections", value: stats.total, icon: Activity, color: "from-blue-500 to-cyan-500", trend: "+12%" },
                    { label: "Pistol Detections", value: stats.pistols, icon: TrendingUp, color: "from-red-500 to-orange-500", trend: "+8%" },
                    { label: "Knife Detections", value: stats.knives, icon: TrendingUp, color: "from-amber-500 to-yellow-500", trend: "+5%" },
                    { label: "Avg Confidence", value: `${stats.avgConfidence}%`, icon: Activity, color: "from-green-500 to-emerald-500", trend: "+3%" },
                ].map((stat, index) => {
                    const Icon = stat.icon;
                    return (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: index * 0.1 }}
                        >
                            <Card className="bg-white border-gray-200 hover:shadow-lg transition-shadow">
                                <CardContent className="pt-6">
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <p className="text-sm text-gray-600">{stat.label}</p>
                                            <p className="text-3xl font-bold text-gray-900 mt-1">{stat.value}</p>
                                            <p className="text-sm text-green-600 mt-1 flex items-center">
                                                <TrendingUp className="h-3 w-3 mr-1" />
                                                {stat.trend}
                                            </p>
                                        </div>
                                        <div className={`bg-gradient-to-br ${stat.color} p-3 rounded-xl`}>
                                            <Icon className="h-6 w-6 text-white" />
                                        </div>
                                    </div>
                                </CardContent>
                            </Card>
                        </motion.div>
                    );
                })}
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Detection Types */}
                <Card className="bg-white border-gray-200">
                    <CardHeader>
                        <CardTitle className="text-gray-900">Detection Types</CardTitle>
                        <CardDescription>Distribution by weapon type</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie
                                    data={getDetectionsByType()}
                                    cx="50%"
                                    cy="50%"
                                    labelLine={false}
                                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                    outerRadius={100}
                                    fill="#8884d8"
                                    dataKey="value"
                                >
                                    {getDetectionsByType().map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>

                {/* Daily Trend */}
                <Card className="bg-white border-gray-200">
                    <CardHeader>
                        <CardTitle className="text-gray-900">Detection Trend</CardTitle>
                        <CardDescription>Last 7 days</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={getDetectionsByDay()}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                <XAxis dataKey="date" stroke="#6b7280" />
                                <YAxis stroke="#6b7280" />
                                <Tooltip />
                                <Line type="monotone" dataKey="count" stroke="#ef4444" strokeWidth={2} />
                            </LineChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>

                {/* Hourly Distribution */}
                <Card className="bg-white border-gray-200 lg:col-span-2">
                    <CardHeader>
                        <CardTitle className="text-gray-900">Hourly Distribution</CardTitle>
                        <CardDescription>Detection patterns throughout the day</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={getDetectionsByHour()}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                                <XAxis dataKey="hour" stroke="#6b7280" />
                                <YAxis stroke="#6b7280" />
                                <Tooltip />
                                <Bar dataKey="count" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};

export default Analytics;
