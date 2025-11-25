/**
 * Session-related TypeScript types
 */

export type SessionStatus =
  | "pending"
  | "queued"
  | "in_progress"
  | "completed"
  | "failed"
  | "cancelled";

export type AgentStatus =
  | "pending"
  | "in_progress"
  | "completed"
  | "failed"
  | "soft_failure";

export interface Session {
  id: string;
  name: string | null;
  input_data: string;
  start_agent: string;
  current_agent: string | null;
  status: SessionStatus;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  agent_outputs: Record<string, AgentOutput>;
}

export interface SessionListItem {
  id: string;
  name: string | null;
  status: SessionStatus;
  current_agent: string | null;
  created_at: string;
  updated_at: string;
}

export interface AgentOutput {
  id?: number;
  session_id: string;
  agent_name: string;
  status: AgentStatus;
  output_data: any;
  output_summary: string | null;
  error_message: string | null;
  retry_count: number;
  created_at?: string;
  completed_at?: string;
  agent_info?: AgentInfo;
}

export interface AgentInfo {
  name: string;
  phase: number;
  output_type: "text" | "json" | "images" | "videos";
}

export interface CreateSessionRequest {
  name?: string;
  input_data: string;
  start_agent?: "agent_1" | "agent_2";
}
