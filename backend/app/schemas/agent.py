"""
Pydantic schemas for agent-related API endpoints
"""

from datetime import datetime
from typing import Optional, Any, Dict

from pydantic import BaseModel


class AgentInfo(BaseModel):
    """Metadata about an agent."""
    name: str
    phase: int
    output_type: str  # "text", "json", "images", "videos"


class AgentOutputResponse(BaseModel):
    """Response schema for agent output."""
    session_id: str
    agent_name: str
    status: str
    output_data: Optional[Any] = None
    output_summary: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    agent_info: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
