/**
 * React Query hooks for queue data
 */

import { useQuery } from "@tanstack/react-query";
import { queueApi } from "@/lib/api";
import { QueueStatus, QueueTask, QueuePosition } from "@/types/queue";

/**
 * Fetch current queue status
 */
export function useQueueStatus() {
  return useQuery<QueueStatus>({
    queryKey: ["queue", "status"],
    queryFn: () => queueApi.getStatus(),
    refetchInterval: 5000, // Poll every 5 seconds
  });
}

/**
 * Fetch queue tasks
 */
export function useQueueTasks(status?: string, limit = 50) {
  return useQuery<QueueTask[]>({
    queryKey: ["queue", "tasks", { status, limit }],
    queryFn: () => queueApi.getTasks(status, limit),
  });
}

/**
 * Fetch queue position for a specific session
 */
export function useQueuePosition(sessionId: string | null) {
  return useQuery<QueuePosition>({
    queryKey: ["queue", "position", sessionId],
    queryFn: () => queueApi.getPosition(sessionId!),
    enabled: !!sessionId,
    refetchInterval: 3000, // Poll more frequently for position updates
  });
}

/**
 * Fetch queue history
 */
export function useQueueHistory(limit = 20) {
  return useQuery<QueueTask[]>({
    queryKey: ["queue", "history", { limit }],
    queryFn: () => queueApi.getHistory(limit),
  });
}
