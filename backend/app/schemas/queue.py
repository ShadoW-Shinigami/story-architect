"""
Pydantic schemas for queue-related API endpoints
"""

from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel


class QueueStatusResponse(BaseModel):
    """Response schema for queue status."""
    is_processing: bool
    current_task: Optional[Dict[str, Any]] = None
    pending_count: int


class QueueTaskResponse(BaseModel):
    """Response schema for queue task."""
    id: int
    session_id: str
    status: str
    priority: int
    start_agent: str
    resume_from: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class QueuePositionResponse(BaseModel):
    """Response schema for queue position query."""
    session_id: str
    position: Optional[int]  # None if not in queue, 0 if running
    total_pending: int
