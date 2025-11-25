/**
 * React Query hooks for session data
 */

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { sessionsApi, pipelineApi } from "@/lib/api";
import { Session, SessionListItem, CreateSessionRequest } from "@/types/session";

/**
 * Fetch list of sessions
 */
export function useSessions(limit = 20, offset = 0) {
  return useQuery<SessionListItem[]>({
    queryKey: ["sessions", { limit, offset }],
    queryFn: () => sessionsApi.list(limit, offset),
  });
}

/**
 * Fetch a single session with all outputs
 */
export function useSession(sessionId: string | null) {
  return useQuery<Session>({
    queryKey: ["session", sessionId],
    queryFn: () => sessionsApi.get(sessionId!),
    enabled: !!sessionId,
    refetchInterval: 5000, // Poll every 5 seconds for updates
  });
}

/**
 * Create a new session
 */
export function useCreateSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateSessionRequest) => sessionsApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
}

/**
 * Update a session
 */
export function useUpdateSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, name }: { id: string; name?: string }) =>
      sessionsApi.update(id, { name }),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["session", variables.id] });
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
}

/**
 * Delete a session
 */
export function useDeleteSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => sessionsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
}

/**
 * Start pipeline execution
 */
export function useStartPipeline() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      sessionId,
      startAgent,
      priority,
    }: {
      sessionId: string;
      startAgent?: string;
      priority?: number;
    }) => pipelineApi.start(sessionId, startAgent, priority),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["session", variables.sessionId] });
      queryClient.invalidateQueries({ queryKey: ["sessions"] });
      queryClient.invalidateQueries({ queryKey: ["queue"] });
    },
  });
}

/**
 * Resume pipeline from agent
 */
export function useResumePipeline() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      sessionId,
      fromAgent,
      priority,
    }: {
      sessionId: string;
      fromAgent: string;
      priority?: number;
    }) => pipelineApi.resume(sessionId, fromAgent, priority),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["session", variables.sessionId] });
      queryClient.invalidateQueries({ queryKey: ["queue"] });
    },
  });
}

/**
 * Cancel pipeline execution
 */
export function useCancelPipeline() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (sessionId: string) => pipelineApi.cancel(sessionId),
    onSuccess: (_, sessionId) => {
      queryClient.invalidateQueries({ queryKey: ["session", sessionId] });
      queryClient.invalidateQueries({ queryKey: ["queue"] });
    },
  });
}

/**
 * Dashboard stats
 */
export interface DashboardStats {
  total_projects: number;
  status_breakdown: Record<string, number>;
  recent_projects: number;
  active_count: number;
  completed_count: number;
  failed_count: number;
}

export function useDashboardStats() {
  return useQuery<DashboardStats>({
    queryKey: ["dashboard", "stats"],
    queryFn: () => sessionsApi.getStats(),
    refetchInterval: 30000, // Poll every 30 seconds
  });
}
