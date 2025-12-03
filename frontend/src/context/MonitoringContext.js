import React, { createContext, useState, useContext, useRef, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';

const MonitoringContext = createContext();

const BACKEND_URL = "http://localhost:8000";
const API = `${BACKEND_URL}/api`;

export const MonitoringProvider = ({ children }) => {
    const [isMonitoring, setIsMonitoring] = useState(false);
    const [detections, setDetections] = useState([]);
    const [stats, setStats] = useState({
        framesProcessed: 0,
        threatsDetected: 0,
        fps: 20,
        peopleCount: 0,
    });

    // Refs for polling to avoid dependency cycles
    const pollIntervalRef = useRef(null);
    const frameCountRef = useRef(0);
    const lastSpeechRef = useRef(0);
    const voiceEnabledRef = useRef(true); // Keep track of voice setting

    // Cleanup on unmount (app close)
    useEffect(() => {
        return () => {
            if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
            }
        };
    }, []);

    const setVoiceEnabled = (enabled) => {
        voiceEnabledRef.current = enabled;
    };

    const speakAlert = (text) => {
        if (!voiceEnabledRef.current || Date.now() - lastSpeechRef.current < 5000) return;

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.1;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        window.speechSynthesis.speak(utterance);
        lastSpeechRef.current = Date.now();
    };

    const playAlertSound = () => {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            oscillator.frequency.value = 800;
            oscillator.type = "sine";

            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.5);
        } catch (error) {
            console.error("Audio error:", error);
        }
    };

    const [emergencyContext, setEmergencyContext] = useState(null);
    const [locationStatus, setLocationStatus] = useState("Locating...");

    // Fetch location and context on mount - non-blocking
    useEffect(() => {
        // Only fetch if not already fetched
        if (emergencyContext) return;

        // Set default location immediately
        setLocationStatus("Using Default Location");

        // Try to get real location asynchronously without blocking
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                async (position) => {
                    const { latitude, longitude } = position.coords;
                    setLocationStatus("Location Found");
                    try {
                        const res = await axios.get(`${API}/emergency-context`, {
                            params: { lat: latitude, lng: longitude }
                        });
                        setEmergencyContext(res.data);
                    } catch (err) {
                        console.error("Context fetch error", err);
                        // Fallback to default context on error so UI doesn't get stuck loading
                        setEmergencyContext({
                            police_stations: [],
                            hospitals: [],
                            fire_stations: []
                        });
                    }
                },
                (err) => {
                    console.warn("Location error", err);
                    setLocationStatus("Location Access Denied");
                    // Stop loading spinner even if location fails
                    setEmergencyContext({
                        police_stations: [],
                        hospitals: [],
                        fire_stations: []
                    });
                },
                {
                    timeout: 2000, // 2 second timeout to avoid blocking
                    enableHighAccuracy: false, // Faster response
                    maximumAge: 300000 // Accept cached location up to 5 minutes old
                }
            );
        } else {
            setLocationStatus("Location Not Supported");
        }
    }, [emergencyContext]);

    const startPolling = () => {
        if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current);
        }

        pollIntervalRef.current = setInterval(async () => {
            try {
                const response = await axios.get(`${API}/status`);
                const data = response.data;

                frameCountRef.current += 1;
                setStats({
                    framesProcessed: frameCountRef.current,
                    threatsDetected: (data.guns || 0) + (data.knives || 0),
                    fps: 20,
                    peopleCount: data.people || 0,
                    latest_ipfs: data.latest_ipfs || null
                });

                const hasThreats = (data.guns > 0) || (data.knives > 0);
                if (hasThreats) {
                    const newDetections = [];
                    if (data.guns > 0) newDetections.push({ detection_type: "pistol", confidence: 0.9 });
                    if (data.knives > 0) newDetections.push({ detection_type: "knife", confidence: 0.9 });
                    setDetections(newDetections);

                    if (data.alert) {
                        playAlertSound();
                        const threat = data.guns > 0 ? "Gun" : "Knife";
                        speakAlert(`Warning. ${threat} detected.`);
                    }
                } else {
                    setDetections([]);
                }
            } catch (error) {
                console.error("Polling error:", error);
            }
        }, 500);
    };

    const startMonitoring = async () => {
        try {
            const response = await axios.post(`${API}/start`);
            if (response.data.status === 'started' || response.data.status === 'already_running') {
                setIsMonitoring(true);
                setDetections([]);
                frameCountRef.current = 0;
                toast.success("Webcam monitoring started");
                startPolling();
            }
        } catch (error) {
            console.error("Error starting monitoring:", error);
            toast.error("Failed to start monitoring");
        }
    };

    const stopMonitoring = async () => {
        try {
            await axios.post(`${API}/stop`);
            setIsMonitoring(false);
            if (pollIntervalRef.current) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
            }
            setStats({
                framesProcessed: 0,
                threatsDetected: 0,
                fps: 20,
                peopleCount: 0,
            });
            setDetections([]);
            toast.success("Monitoring stopped");
        } catch (error) {
            console.error("Error stopping monitoring:", error);
            toast.error("Failed to stop monitoring");
        }
    };

    const [cameras, setCameras] = useState([]);
    const [verifications, setVerifications] = useState([]);
    const [verificationStats, setVerificationStats] = useState({ pending: 0, confirmed: 0, rejected: 0, total: 0 });

    // Fetch cameras
    const fetchCameras = async () => {
        try {
            const response = await axios.get(`${API}/cameras`);
            setCameras(response.data.cameras || []);
        } catch (error) {
            console.error("Error fetching cameras:", error);
        }
    };

    // Fetch verifications
    const fetchVerifications = async () => {
        try {
            const [verificationsRes, statsRes] = await Promise.all([
                axios.get(`${API}/pending-verifications`, { params: { limit: 20 } }),
                axios.get(`${API}/verification-stats`)
            ]);
            setVerifications(verificationsRes.data.verifications || []);
            setVerificationStats(statsRes.data);
        } catch (error) {
            console.error("Error fetching verifications:", error);
        }
    };

    // Initial fetch and polling for background data
    useEffect(() => {
        fetchCameras();
        fetchVerifications();

        const interval = setInterval(() => {
            fetchCameras();
            fetchVerifications();
        }, 10000); // Poll every 10 seconds

        return () => clearInterval(interval);
    }, []);

    return (
        <MonitoringContext.Provider value={{
            isMonitoring,
            stats,
            detections,
            startMonitoring,
            stopMonitoring,
            setVoiceEnabled,
            emergencyContext,
            locationStatus,
            cameras,
            fetchCameras,
            verifications,
            fetchVerifications,
            verificationStats
        }}>
            {children}
        </MonitoringContext.Provider>
    );
};

export const useMonitoring = () => useContext(MonitoringContext);
