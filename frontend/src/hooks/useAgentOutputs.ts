import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export interface AgentOutput {
  id: string;
  session_id: string;
  agent_name: string;
  status: "pending" | "in_progress" | "completed" | "failed";
  output_data: any;
  output_summary?: string;
  error_message?: string;
  retry_count: number;
  started_at?: string;
  completed_at?: string;
}

// Backend response format for all outputs
interface AllAgentsResponse {
  session_id: string;
  agents: Record<string, {
    status: string;
    output_data: any;
    output_summary?: string;
    error_message?: string;
    retry_count: number;
    completed_at?: string;
    agent_info: { name: string; phase: number; output_type: string };
  }>;
}

export function useAgentOutputs(sessionId: string) {
  return useQuery({
    queryKey: ["agent-outputs", sessionId],
    queryFn: async () => {
      const response = await api.get<AllAgentsResponse>(
        `/agents/${sessionId}`
      );
      // Transform dict to array format expected by frontend
      return Object.entries(response.data.agents).map(([agent_name, data]) => ({
        id: `${sessionId}-${agent_name}`,
        session_id: sessionId,
        agent_name,
        status: data.status as AgentOutput["status"],
        output_data: data.output_data,
        output_summary: data.output_summary,
        error_message: data.error_message,
        retry_count: data.retry_count,
        completed_at: data.completed_at,
      }));
    },
    enabled: !!sessionId,
    refetchInterval: 5000, // Refetch every 5 seconds for updates
  });
}

export function useAgentOutput(sessionId: string, agentName: string) {
  return useQuery({
    queryKey: ["agent-output", sessionId, agentName],
    queryFn: async () => {
      const response = await api.get<AgentOutput>(
        `/agents/${sessionId}/${agentName}`
      );
      return response.data;
    },
    enabled: !!sessionId && !!agentName,
  });
}
