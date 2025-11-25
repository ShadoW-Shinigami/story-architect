"""
Pydantic schemas for progress events (WebSocket messages)
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Any, List

from pydantic import BaseModel, Field


class ProgressEventType(str, Enum):
    """Types of progress events."""
    # Pipeline level
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"

    # Agent level
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    AGENT_RETRY = "agent_retry"

    # Step level (within agents)
    STEP_STARTED = "step_started"
    STEP_PROGRESS = "step_progress"
    STEP_COMPLETED = "step_completed"

    # Output events (for immediate display)
    IMAGE_GENERATED = "image_generated"
    VIDEO_GENERATED = "video_generated"

    # Queue events
    QUEUE_POSITION_CHANGED = "queue_position_changed"


class ProgressEvent(BaseModel):
    """
    Progress event schema for WebSocket messages.

    Sent from backend to frontend to update UI in real-time.
    """
    event_type: ProgressEventType
    session_id: str

    # Agent identification
    agent_name: Optional[str] = None
    agent_display_name: Optional[str] = None

    # Progress tracking (0.0 to 1.0)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_progress: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Message and details
    message: str = ""
    step_name: Optional[str] = None
    step_current: Optional[int] = None
    step_total: Optional[int] = None

    # Output references (for IMAGE_GENERATED, VIDEO_GENERATED)
    output_path: Optional[str] = None
    output_thumbnail: Optional[str] = None
    output_summary: Optional[Any] = None

    # Error information
    error: Optional[str] = None
    retry_count: Optional[int] = None

    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_ws_message(self) -> dict:
        """Convert to WebSocket message format."""
        return {
            "type": "progress",
            "data": self.model_dump(mode="json", exclude_none=True)
        }


class AgentCompletedEvent(BaseModel):
    """Event sent when an agent completes successfully."""
    session_id: str
    agent_name: str
    output_summary: Optional[str] = None
    duration_seconds: Optional[float] = None

    def to_ws_message(self) -> dict:
        return {
            "type": "agent_completed",
            "data": self.model_dump(mode="json", exclude_none=True)
        }


class PipelineCompletedEvent(BaseModel):
    """Event sent when the entire pipeline completes."""
    session_id: str
    total_duration_seconds: float
    agents_completed: List[str]

    def to_ws_message(self) -> dict:
        return {
            "type": "pipeline_completed",
            "data": self.model_dump(mode="json", exclude_none=True)
        }


class QueueUpdatedEvent(BaseModel):
    """Event sent when queue status changes."""
    is_processing: bool
    current_session_id: Optional[str]
    pending_count: int

    def to_ws_message(self) -> dict:
        return {
            "type": "queue_updated",
            "data": self.model_dump(mode="json", exclude_none=True)
        }
