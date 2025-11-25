/**
 * Queue-related TypeScript types
 */

export type TaskStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface QueueTask {
  id: number;
  session_id: string;
  status: TaskStatus;
  priority: number;
  start_agent: string;
  resume_from: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
}

export interface QueueStatus {
  is_processing: boolean;
  current_task: {
    id: number;
    session_id: string;
    started_at: string | null;
  } | null;
  pending_count: number;
}

export interface QueuePosition {
  session_id: string;
  position: number | null;
  total_pending: number;
}
