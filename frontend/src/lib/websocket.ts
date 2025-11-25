/**
 * WebSocket client for real-time updates
 */

import { useProgressStore } from "@/stores/progressStore";
import { useQueueStore } from "@/stores/queueStore";
import { useConnectionStore } from "@/stores/connectionStore";
import { WebSocketMessage } from "@/types/progress";

// Only access env var and WebSocket on client side
const getWsBaseUrl = () => {
  if (typeof window === "undefined") return "";
  return process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
};

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_BASE = 1000; // 1 second

class WebSocketClient {
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private isGlobal: boolean = false;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private isConnecting: boolean = false;

  connect(sessionId: string) {
    // Guard against SSR
    if (typeof window === "undefined") return;

    this.sessionId = sessionId;
    this.isGlobal = false;
    const wsUrl = `${getWsBaseUrl()}/ws/session/${sessionId}`;
    console.log("[WebSocket] Connecting to session:", wsUrl);
    this._connect(wsUrl);
  }

  connectGlobal() {
    // Guard against SSR
    if (typeof window === "undefined") return;

    this.sessionId = null;
    this.isGlobal = true;
    const wsUrl = `${getWsBaseUrl()}/ws/queue`;
    console.log("[WebSocket] Connecting to global queue:", wsUrl);
    this._connect(wsUrl);
  }

  private _connect(url: string) {
    // Guard against SSR
    if (typeof window === "undefined") return;

    // Prevent multiple simultaneous connection attempts
    if (this.isConnecting) {
      console.log("[WebSocket] Already connecting, skipping");
      return;
    }

    this.isConnecting = true;
    const connectionStore = useConnectionStore.getState();
    connectionStore.setConnecting();

    try {
      console.log("[WebSocket] Creating WebSocket connection to:", url);
      this.ws = new WebSocket(url);
      this._setupHandlers();
    } catch (error) {
      console.error("[WebSocket] Connection error:", error);
      this.isConnecting = false;
      connectionStore.setError("Failed to connect");
      this._handleReconnect();
    }
  }

  private _setupHandlers() {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log("[WebSocket] ✓ Connected successfully!");
      this.isConnecting = false;
      useConnectionStore.getState().setConnected();
      useConnectionStore.getState().resetReconnectAttempts();
      this._startPing();
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this._handleMessage(message);
      } catch (error) {
        console.error("[WebSocket] Failed to parse message:", error);
      }
    };

    this.ws.onclose = (event) => {
      console.log("[WebSocket] Connection closed:", event.code, event.reason);
      this.isConnecting = false;
      useConnectionStore.getState().setDisconnected();
      this._stopPing();
      // Only reconnect if this wasn't a clean close
      if (event.code !== 1000) {
        this._handleReconnect();
      }
    };

    this.ws.onerror = (error) => {
      console.error("[WebSocket] Connection error - is backend running on port 8000?", error);
      this.isConnecting = false;
      useConnectionStore.getState().setError("Connection error");
    };
  }

  private _handleMessage(message: WebSocketMessage) {
    const progressStore = useProgressStore.getState();
    const queueStore = useQueueStore.getState();

    switch (message.type) {
      case "progress":
        progressStore.addEvent(message.data);
        break;

      case "agent_completed":
        progressStore.setAgentCompleted(message.data.agent_name);
        break;

      case "pipeline_completed":
        progressStore.setPipelineCompleted(message.data);
        break;

      case "pipeline_failed":
        progressStore.setPipelineFailed(message.data?.error);
        break;

      case "queue_updated":
        queueStore.updateFromEvent(message.data);
        break;

      case "pong":
        // Ping response - connection is alive
        break;

      default:
        console.log("Unknown WebSocket message type:", message.type);
    }
  }

  private _handleReconnect() {
    const connectionStore = useConnectionStore.getState();

    if (connectionStore.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.log("[WebSocket] Max reconnect attempts reached, giving up");
      connectionStore.setError("Unable to reconnect - check if backend is running");
      return;
    }

    connectionStore.incrementReconnectAttempts();
    const delay = RECONNECT_DELAY_BASE * Math.pow(2, connectionStore.reconnectAttempts);

    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${connectionStore.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`);

    // Clear any existing reconnect timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    this.reconnectTimeout = setTimeout(() => {
      if (this.isGlobal) {
        this.connectGlobal();
      } else if (this.sessionId) {
        this.connect(this.sessionId);
      }
    }, delay);
  }

  private _startPing() {
    this._stopPing();
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: "ping" }));
      }
    }, 30000); // Every 30 seconds
  }

  private _stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  disconnect() {
    console.log("[WebSocket] Disconnect called");
    this._stopPing();

    // Clear reconnect timeout
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.close(1000, "Client disconnect"); // Clean close
      this.ws = null;
    }

    this.sessionId = null;
    this.isConnecting = false;
    useConnectionStore.getState().setDisconnected();
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
export const wsClient = new WebSocketClient();
