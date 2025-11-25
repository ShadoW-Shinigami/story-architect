"""
Pydantic schemas for session-related API endpoints
"""

from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    """Request schema for creating a new session."""
    name: Optional[str] = Field(None, max_length=255, description="Optional session name")
    input_data: str = Field(..., min_length=1, description="Input text (logline or screenplay)")
    start_agent: Optional[str] = Field(
        "agent_1",
        pattern=r"^agent_(1|2)$",
        description="Starting agent (agent_1 for logline, agent_2 for screenplay)"
    )


class SessionListResponse(BaseModel):
    """Response schema for session list items."""
    id: str
    name: Optional[str]
    status: str
    current_agent: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentOutputSummary(BaseModel):
    """Summary of an agent's output for session response."""
    status: str
    output_summary: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    completed_at: Optional[datetime] = None


class SessionResponse(BaseModel):
    """Full session response with agent outputs."""
    id: str
    name: Optional[str]
    input_data: str
    start_agent: str
    current_agent: Optional[str]
    status: str
    error_message: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    agent_outputs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        from_attributes = True
