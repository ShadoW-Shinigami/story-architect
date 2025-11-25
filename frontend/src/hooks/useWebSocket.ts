/**
 * React hook for WebSocket connection management
 */

import { useEffect, useCallback } from "react";
import { wsClient } from "@/lib/websocket";
import { useProgressStore } from "@/stores/progressStore";
import { useConnectionStore } from "@/stores/connectionStore";

/**
 * Hook to manage WebSocket connection for a specific session
 */
export function useSessionWebSocket(sessionId: string | null) {
  const reset = useProgressStore((s) => s.reset);
  const status = useConnectionStore((s) => s.status);

  useEffect(() => {
    console.log("[useSessionWebSocket] Effect running, sessionId:", sessionId);

    if (!sessionId) {
      // No session - reconnect to global WebSocket
      wsClient.connectGlobal();
      return;
    }

    // Reset progress state for new session
    reset();

    // Connect to session WebSocket
    wsClient.connect(sessionId);

    // Cleanup: reconnect to global when leaving session
    return () => {
      console.log("[useSessionWebSocket] Session ended, reconnecting to global");
      wsClient.connectGlobal();
    };
  }, [sessionId, reset]);

  return {
    status,
    isConnected: status === "connected",
    reconnect: useCallback(() => {
      if (sessionId) {
        wsClient.disconnect();
        wsClient.connect(sessionId);
      }
    }, [sessionId]),
  };
}

/**
 * Hook to manage global WebSocket connection (queue updates)
 */
export function useGlobalWebSocket() {
  const status = useConnectionStore((s) => s.status);

  useEffect(() => {
    console.log("[useGlobalWebSocket] Effect running - initiating connection");
    wsClient.connectGlobal();

    return () => {
      console.log("[useGlobalWebSocket] Cleanup - disconnecting");
      wsClient.disconnect();
    };
  }, []);

  return {
    status,
    isConnected: status === "connected",
    reconnect: useCallback(() => {
      wsClient.disconnect();
      wsClient.connectGlobal();
    }, []),
  };
}

// Alias for backward compatibility
export { useSessionWebSocket as useWebSocket };
