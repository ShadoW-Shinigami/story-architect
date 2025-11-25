/**
 * Progress event types for WebSocket communication
 */

export type ProgressEventType =
  | "pipeline_started"
  | "pipeline_completed"
  | "pipeline_failed"
  | "agent_started"
  | "agent_completed"
  | "agent_failed"
  | "agent_retry"
  | "step_started"
  | "step_progress"
  | "step_completed"
  | "image_generated"
  | "video_generated"
  | "queue_position_changed";

export interface ProgressEvent {
  event_type: ProgressEventType;
  session_id: string;
  agent_name?: string;
  agent_display_name?: string;
  progress: number;
  overall_progress?: number;
  message: string;
  step_name?: string;
  step_current?: number;
  step_total?: number;
  output_path?: string;
  output_thumbnail?: string;
  output_summary?: any;
  error?: string;
  retry_count?: number;
  timestamp: string;
}

export interface AgentCompletedEvent {
  session_id: string;
  agent_name: string;
  output_summary?: string;
  duration_seconds?: number;
}

export interface PipelineCompletedEvent {
  session_id: string;
  total_duration_seconds: number;
  agents_completed: string[];
}

export interface QueueUpdatedEvent {
  is_processing: boolean;
  current_session_id?: string;
  pending_count: number;
}

export interface WebSocketMessage {
  type: "progress" | "agent_completed" | "pipeline_completed" | "pipeline_failed" | "queue_updated" | "pong";
  data?: any;
}
