/**
 * Zustand store for queue status
 */

import { create } from "zustand";
import { QueueStatus } from "@/types/queue";
import { QueueUpdatedEvent } from "@/types/progress";

interface QueueState {
  // Current state
  isProcessing: boolean;
  currentSessionId: string | null;
  pendingCount: number;

  // Actions
  updateFromEvent: (event: QueueUpdatedEvent) => void;
  setStatus: (status: QueueStatus) => void;
  reset: () => void;
}

export const useQueueStore = create<QueueState>((set) => ({
  isProcessing: false,
  currentSessionId: null,
  pendingCount: 0,

  updateFromEvent: (event) =>
    set({
      isProcessing: event.is_processing,
      currentSessionId: event.current_session_id || null,
      pendingCount: event.pending_count,
    }),

  setStatus: (status) =>
    set({
      isProcessing: status.is_processing,
      currentSessionId: status.current_task?.session_id || null,
      pendingCount: status.pending_count,
    }),

  reset: () =>
    set({
      isProcessing: false,
      currentSessionId: null,
      pendingCount: 0,
    }),
}));
