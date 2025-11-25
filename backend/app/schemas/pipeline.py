"""
Pydantic schemas for pipeline-related API endpoints
"""

from typing import Optional, List

from pydantic import BaseModel, Field


class PipelineStartRequest(BaseModel):
    """Request schema for starting pipeline execution."""
    start_agent: Optional[str] = Field(
        None,
        pattern=r"^agent_(1[01]?|[1-9])$",
        description="Override starting agent"
    )
    priority: int = Field(
        0,
        ge=0,
        le=100,
        description="Task priority (higher = more urgent)"
    )


class PipelineResumeRequest(BaseModel):
    """Request schema for resuming pipeline from specific agent."""
    from_agent: str = Field(
        ...,
        pattern=r"^agent_(1[01]?|[1-9])$",
        description="Agent to resume from"
    )
    priority: int = Field(
        0,
        ge=0,
        le=100,
        description="Task priority"
    )


class PipelineStatusResponse(BaseModel):
    """Response schema for pipeline status."""
    session_id: str
    status: str
    current_agent: Optional[str]
    completed_agents: List[str]
    error_message: Optional[str]
    queue_position: Optional[int] = None
