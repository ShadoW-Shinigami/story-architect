/**
 * Zustand store for WebSocket connection status
 */

import { create } from "zustand";

export type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

interface ConnectionState {
  // Status
  status: ConnectionStatus;
  lastConnectedAt: Date | null;
  reconnectAttempts: number;
  error: string | null;

  // Actions
  setConnecting: () => void;
  setConnected: () => void;
  setDisconnected: () => void;
  setError: (error: string) => void;
  incrementReconnectAttempts: () => void;
  resetReconnectAttempts: () => void;
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  status: "disconnected",
  lastConnectedAt: null,
  reconnectAttempts: 0,
  error: null,

  setConnecting: () =>
    set({
      status: "connecting",
      error: null,
    }),

  setConnected: () =>
    set({
      status: "connected",
      lastConnectedAt: new Date(),
      reconnectAttempts: 0,
      error: null,
    }),

  setDisconnected: () =>
    set({
      status: "disconnected",
    }),

  setError: (error) =>
    set({
      status: "error",
      error,
    }),

  incrementReconnectAttempts: () =>
    set((state) => ({
      reconnectAttempts: state.reconnectAttempts + 1,
    })),

  resetReconnectAttempts: () =>
    set({
      reconnectAttempts: 0,
    }),
}));
