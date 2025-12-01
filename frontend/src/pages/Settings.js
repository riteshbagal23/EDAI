import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Settings as SettingsIcon, Bell, Shield, Database, Palette, Save } from "lucide-react";
import toast from "react-hot-toast";

const Settings = () => {
    const [settings, setSettings] = useState({
        notifications: {
            email: true,
            push: false,
            sms: false,
        },
        detection: {
            minConfidence: 0.3,
            enablePeopleCounting: false,
            enableThermalDetection: false,
        },
        system: {
            autoBackup: true,
            retentionDays: 30,
            maxCameras: 10,
        },
        appearance: {
            theme: "light",
            compactMode: false,
        }
    });

    const handleSave = () => {
        // Save settings logic here
        toast.success("Settings saved successfully!");
    };

    const updateSetting = (category, key, value) => {
        setSettings(prev => ({
            ...prev,
            [category]: {
                ...prev[category],
                [key]: value
            }
        }));
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                    Settings
                </h1>
                <p className="text-gray-600 mt-2">Manage your system preferences and configuration</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Settings Navigation */}
                <div className="lg:col-span-1">
                    <Card className="bg-white border-gray-200">
                        <CardContent className="pt-6">
                            <nav className="space-y-1">
                                {[
                                    { icon: Bell, label: "Notifications", id: "notifications" },
                                    { icon: Shield, label: "Detection", id: "detection" },
                                    { icon: Database, label: "System", id: "system" },
                                    { icon: Palette, label: "Appearance", id: "appearance" },
                                ].map((item) => {
                                    const Icon = item.icon;
                                    return (
                                        <button
                                            key={item.id}
                                            className="w-full flex items-center space-x-3 px-4 py-3 rounded-lg hover:bg-gray-50 transition-colors text-left"
                                        >
                                            <Icon className="h-5 w-5 text-gray-600" />
                                            <span className="text-sm font-medium text-gray-900">{item.label}</span>
                                        </button>
                                    );
                                })}
                            </nav>
                        </CardContent>
                    </Card>
                </div>

                {/* Settings Content */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Notifications */}
                    <Card className="bg-white border-gray-200">
                        <CardHeader>
                            <CardTitle className="text-gray-900 flex items-center">
                                <Bell className="h-5 w-5 mr-2" />
                                Notifications
                            </CardTitle>
                            <CardDescription>Configure how you receive alerts</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <Label htmlFor="email-notif">Email Notifications</Label>
                                    <p className="text-sm text-gray-500">Receive alerts via email</p>
                                </div>
                                <Switch
                                    id="email-notif"
                                    checked={settings.notifications.email}
                                    onCheckedChange={(checked) => updateSetting('notifications', 'email', checked)}
                                />
                            </div>
                            <Separator />
                            <div className="flex items-center justify-between">
                                <div>
                                    <Label htmlFor="push-notif">Push Notifications</Label>
                                    <p className="text-sm text-gray-500">Browser push notifications</p>
                                </div>
                                <Switch
                                    id="push-notif"
                                    checked={settings.notifications.push}
                                    onCheckedChange={(checked) => updateSetting('notifications', 'push', checked)}
                                />
                            </div>
                            <Separator />
                            <div className="flex items-center justify-between">
                                <div>
                                    <Label htmlFor="sms-notif">SMS Alerts</Label>
                                    <p className="text-sm text-gray-500">Text message alerts</p>
                                </div>
                                <Switch
                                    id="sms-notif"
                                    checked={settings.notifications.sms}
                                    onCheckedChange={(checked) => updateSetting('notifications', 'sms', checked)}
                                />
                            </div>
                        </CardContent>
                    </Card>

                    {/* Detection Settings */}
                    <Card className="bg-white border-gray-200">
                        <CardHeader>
                            <CardTitle className="text-gray-900 flex items-center">
                                <Shield className="h-5 w-5 mr-2" />
                                Detection Settings
                            </CardTitle>
                            <CardDescription>Configure detection parameters</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div>
                                <Label htmlFor="min-confidence">Minimum Confidence Threshold</Label>
                                <div className="flex items-center space-x-4 mt-2">
                                    <Input
                                        id="min-confidence"
                                        type="number"
                                        min="0"
                                        max="1"
                                        step="0.05"
                                        value={settings.detection.minConfidence}
                                        onChange={(e) => updateSetting('detection', 'minConfidence', parseFloat(e.target.value))}
                                        className="w-24"
                                    />
                                    <span className="text-sm text-gray-600">{(settings.detection.minConfidence * 100).toFixed(0)}%</span>
                                </div>
                            </div>
                            <Separator />
                            <div className="flex items-center justify-between">
                                <div>
                                    <Label htmlFor="people-counting">People Counting</Label>
                                    <p className="text-sm text-gray-500">Enable people detection</p>
                                </div>
                                <Switch
                                    id="people-counting"
                                    checked={settings.detection.enablePeopleCounting}
                                    onCheckedChange={(checked) => updateSetting('detection', 'enablePeopleCounting', checked)}
                                />
                            </div>
                            <Separator />
                            <div className="flex items-center justify-between">
                                <div>
                                    <Label htmlFor="thermal-detection">Thermal Detection</Label>
                                    <p className="text-sm text-gray-500">Enable thermal gun detection</p>
                                </div>
                                <Switch
                                    id="thermal-detection"
                                    checked={settings.detection.enableThermalDetection}
                                    onCheckedChange={(checked) => updateSetting('detection', 'enableThermalDetection', checked)}
                                />
                            </div>
                        </CardContent>
                    </Card>

                    {/* System Settings */}
                    <Card className="bg-white border-gray-200">
                        <CardHeader>
                            <CardTitle className="text-gray-900 flex items-center">
                                <Database className="h-5 w-5 mr-2" />
                                System Settings
                            </CardTitle>
                            <CardDescription>Configure system behavior</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="flex items-center justify-between">
                                <div>
                                    <Label htmlFor="auto-backup">Automatic Backup</Label>
                                    <p className="text-sm text-gray-500">Daily database backup</p>
                                </div>
                                <Switch
                                    id="auto-backup"
                                    checked={settings.system.autoBackup}
                                    onCheckedChange={(checked) => updateSetting('system', 'autoBackup', checked)}
                                />
                            </div>
                            <Separator />
                            <div>
                                <Label htmlFor="retention">Data Retention (Days)</Label>
                                <Input
                                    id="retention"
                                    type="number"
                                    min="7"
                                    max="365"
                                    value={settings.system.retentionDays}
                                    onChange={(e) => updateSetting('system', 'retentionDays', parseInt(e.target.value))}
                                    className="mt-2 w-32"
                                />
                            </div>
                            <Separator />
                            <div>
                                <Label htmlFor="max-cameras">Maximum Cameras</Label>
                                <Input
                                    id="max-cameras"
                                    type="number"
                                    min="1"
                                    max="50"
                                    value={settings.system.maxCameras}
                                    onChange={(e) => updateSetting('system', 'maxCameras', parseInt(e.target.value))}
                                    className="mt-2 w-32"
                                />
                            </div>
                        </CardContent>
                    </Card>

                    {/* Save Button */}
                    <div className="flex justify-end">
                        <Button onClick={handleSave} className="bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700">
                            <Save className="h-4 w-4 mr-2" />
                            Save Changes
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Settings;
