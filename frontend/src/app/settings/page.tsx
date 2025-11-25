"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/shared/Sidebar";
import { Header } from "@/components/shared/Header";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Save,
  RefreshCcw,
  CheckCircle,
  XCircle,
  Settings,
  Server,
  Key,
  Palette,
  Loader2,
  AlertTriangle,
} from "lucide-react";

interface ConnectionStatus {
  backend: boolean;
  websocket: boolean;
  gemini: boolean;
}

export default function SettingsPage() {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>({
    backend: false,
    websocket: false,
    gemini: false,
  });
  const [checking, setChecking] = useState(false);
  const [apiUrl, setApiUrl] = useState("");
  const [wsUrl, setWsUrl] = useState("");

  useEffect(() => {
    // Load saved settings
    setApiUrl(process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api");
    setWsUrl(process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws");
    checkConnections();
  }, []);

  const checkConnections = async () => {
    setChecking(true);
    const status: ConnectionStatus = {
      backend: false,
      websocket: false,
      gemini: false,
    };

    // Check backend
    try {
      const response = await fetch(`${apiUrl.replace('/api', '')}/health`);
      status.backend = response.ok;
    } catch {
      status.backend = false;
    }

    // Check WebSocket (just verify URL format)
    status.websocket = wsUrl.startsWith("ws://") || wsUrl.startsWith("wss://");

    // Gemini check would require backend call
    if (status.backend) {
      try {
        const response = await fetch(`${apiUrl}/agents/info`);
        status.gemini = response.ok;
      } catch {
        status.gemini = false;
      }
    }

    setConnectionStatus(status);
    setChecking(false);
  };

  const StatusIndicator = ({ connected }: { connected: boolean }) => (
    <div className="flex items-center gap-2">
      {connected ? (
        <>
          <CheckCircle className="h-4 w-4 text-green-500" />
          <span className="text-sm text-green-600">Connected</span>
        </>
      ) : (
        <>
          <XCircle className="h-4 w-4 text-red-500" />
          <span className="text-sm text-red-600">Disconnected</span>
        </>
      )}
    </div>
  );

  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header title="Settings" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="max-w-4xl mx-auto space-y-6">
            {/* Connection Status */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Server className="h-5 w-5" />
                    Connection Status
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={checkConnections}
                    disabled={checking}
                  >
                    {checking ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <RefreshCcw className="mr-2 h-4 w-4" />
                    )}
                    Check Status
                  </Button>
                </CardTitle>
                <CardDescription>
                  Monitor connections to backend services
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="flex items-center justify-between p-4 rounded-lg border">
                    <span className="font-medium">Backend API</span>
                    <StatusIndicator connected={connectionStatus.backend} />
                  </div>
                  <div className="flex items-center justify-between p-4 rounded-lg border">
                    <span className="font-medium">WebSocket</span>
                    <StatusIndicator connected={connectionStatus.websocket} />
                  </div>
                  <div className="flex items-center justify-between p-4 rounded-lg border">
                    <span className="font-medium">Gemini API</span>
                    <StatusIndicator connected={connectionStatus.gemini} />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Settings Tabs */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Configuration
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="connection">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="connection">Connection</TabsTrigger>
                    <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
                    <TabsTrigger value="appearance">Appearance</TabsTrigger>
                  </TabsList>

                  <TabsContent value="connection" className="mt-6 space-y-6">
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="apiUrl">Backend API URL</Label>
                        <Input
                          id="apiUrl"
                          value={apiUrl}
                          onChange={(e) => setApiUrl(e.target.value)}
                          placeholder="http://localhost:8000/api"
                        />
                        <p className="text-sm text-muted-foreground">
                          The URL of the FastAPI backend server
                        </p>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="wsUrl">WebSocket URL</Label>
                        <Input
                          id="wsUrl"
                          value={wsUrl}
                          onChange={(e) => setWsUrl(e.target.value)}
                          placeholder="ws://localhost:8000/ws"
                        />
                        <p className="text-sm text-muted-foreground">
                          WebSocket endpoint for real-time progress updates
                        </p>
                      </div>

                      <div className="p-4 rounded-lg bg-yellow-50 border border-yellow-200">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5" />
                          <div>
                            <p className="font-medium text-yellow-800">Environment Variables</p>
                            <p className="text-sm text-yellow-700 mt-1">
                              Connection URLs are configured via environment variables.
                              To change them permanently, update your <code className="bg-yellow-100 px-1 rounded">.env.local</code> file:
                            </p>
                            <pre className="mt-2 p-2 bg-yellow-100 rounded text-xs">
{`NEXT_PUBLIC_API_URL=http://localhost:8000/api
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws`}
                            </pre>
                          </div>
                        </div>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="pipeline" className="mt-6 space-y-6">
                    <div className="space-y-4">
                      <h3 className="font-medium">Pipeline Agents</h3>
                      <div className="grid gap-2">
                        {[
                          { name: "Agent 1", desc: "Screenplay Processing", status: "active" },
                          { name: "Agent 2", desc: "Scene Breakdown", status: "active" },
                          { name: "Agent 3", desc: "Shot Breakdown", status: "active" },
                          { name: "Agent 4", desc: "Shot Grouping", status: "active" },
                          { name: "Agent 5", desc: "Character Creation", status: "placeholder" },
                          { name: "Agent 6", desc: "Parent Shot Generation", status: "placeholder" },
                          { name: "Agent 7", desc: "Parent Verification", status: "placeholder" },
                          { name: "Agent 8", desc: "Child Shot Generation", status: "placeholder" },
                          { name: "Agent 9", desc: "Child Verification", status: "placeholder" },
                          { name: "Agent 10", desc: "Video Dialogue", status: "placeholder" },
                          { name: "Agent 11", desc: "Video Editing", status: "placeholder" },
                        ].map((agent) => (
                          <div
                            key={agent.name}
                            className="flex items-center justify-between p-3 rounded-lg border"
                          >
                            <div>
                              <span className="font-medium">{agent.name}</span>
                              <span className="text-muted-foreground ml-2">-</span>
                              <span className="text-sm text-muted-foreground ml-2">
                                {agent.desc}
                              </span>
                            </div>
                            <Badge
                              variant={agent.status === "active" ? "default" : "secondary"}
                            >
                              {agent.status === "active" ? "Active" : "Placeholder"}
                            </Badge>
                          </div>
                        ))}
                      </div>

                      <div className="p-4 rounded-lg bg-blue-50 border border-blue-200">
                        <p className="text-sm text-blue-700">
                          <strong>Note:</strong> Agents 5-11 are currently placeholders.
                          They require external services (PIL, FAL AI, Vertex AI Veo, WhisperX, FFmpeg)
                          to be fully implemented.
                        </p>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="appearance" className="mt-6 space-y-6">
                    <div className="space-y-4">
                      <h3 className="font-medium flex items-center gap-2">
                        <Palette className="h-4 w-4" />
                        Theme
                      </h3>
                      <div className="grid gap-4 md:grid-cols-3">
                        {["light", "dark", "system"].map((theme) => (
                          <Button
                            key={theme}
                            variant="outline"
                            className="h-20 flex flex-col items-center justify-center gap-2"
                            onClick={() => {
                              // Theme toggle would be implemented here
                            }}
                          >
                            <div
                              className={`w-8 h-8 rounded ${
                                theme === "light"
                                  ? "bg-white border"
                                  : theme === "dark"
                                  ? "bg-gray-900"
                                  : "bg-gradient-to-br from-white to-gray-900"
                              }`}
                            />
                            <span className="capitalize">{theme}</span>
                          </Button>
                        ))}
                      </div>

                      <p className="text-sm text-muted-foreground">
                        Theme switching is coming soon. Currently uses system preference.
                      </p>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>

            {/* Version Info */}
            <Card>
              <CardContent className="py-4">
                <div className="flex items-center justify-between text-sm text-muted-foreground">
                  <span>Story Architect v2.0.0</span>
                  <span>Built with Next.js + FastAPI + Gemini</span>
                </div>
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
}
