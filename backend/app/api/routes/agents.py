"""
Agent output API endpoints
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.session import Session
from app.models.agent_output import AgentOutput, AgentStatus
from app.schemas.agent import AgentOutputResponse

router = APIRouter()


# Agent metadata for UI display
AGENT_INFO = {
    "agent_1": {"name": "Screenplay Generator", "phase": 1, "output_type": "text"},
    "agent_2": {"name": "Scene Breakdown", "phase": 1, "output_type": "json"},
    "agent_3": {"name": "Shot Breakdown", "phase": 1, "output_type": "json"},
    "agent_4": {"name": "Shot Grouping", "phase": 1, "output_type": "json"},
    "agent_5": {"name": "Character Creator", "phase": 2, "output_type": "images"},
    "agent_6": {"name": "Parent Image Generator", "phase": 2, "output_type": "images"},
    "agent_7": {"name": "Parent Verification", "phase": 2, "output_type": "images"},
    "agent_8": {"name": "Child Image Generator", "phase": 2, "output_type": "images"},
    "agent_9": {"name": "Child Verification", "phase": 2, "output_type": "images"},
    "agent_10": {"name": "Video Dialogue Generator", "phase": 3, "output_type": "videos"},
    "agent_11": {"name": "Intelligent Video Editor", "phase": 3, "output_type": "videos"},
}


@router.get("/info")
async def get_agent_info():
    """
    Get metadata about all agents.
    """
    return AGENT_INFO


@router.get("/{session_id}/{agent_name}", response_model=AgentOutputResponse)
async def get_agent_output(
    session_id: str,
    agent_name: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get output for a specific agent in a session.
    """
    # Validate agent name
    if agent_name not in AGENT_INFO:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent name. Must be one of: {list(AGENT_INFO.keys())}"
        )

    # Get session to verify it exists
    session_result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = session_result.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get agent output
    result = await db.execute(
        select(AgentOutput).where(
            AgentOutput.session_id == session_id,
            AgentOutput.agent_name == agent_name
        )
    )
    output = result.scalars().first()

    if not output:
        # Return empty pending state if no output exists yet
        return AgentOutputResponse(
            session_id=session_id,
            agent_name=agent_name,
            status="pending",
            output_data=None,
            output_summary=None,
            error_message=None,
            retry_count=0,
            agent_info=AGENT_INFO[agent_name]
        )

    return AgentOutputResponse(
        session_id=output.session_id,
        agent_name=output.agent_name,
        status=output.status.value,
        output_data=output.output_data,
        output_summary=output.output_summary,
        error_message=output.error_message,
        retry_count=output.retry_count,
        created_at=output.created_at,
        completed_at=output.completed_at,
        agent_info=AGENT_INFO[agent_name]
    )


@router.get("/{session_id}")
async def get_all_agent_outputs(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get all agent outputs for a session.
    """
    # Get session
    session_result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = session_result.scalars().first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build response with all agents
    outputs = {}
    for agent_name, info in AGENT_INFO.items():
        # Find output for this agent
        agent_output = next(
            (ao for ao in session.agent_outputs if ao.agent_name == agent_name),
            None
        )

        if agent_output:
            outputs[agent_name] = {
                "status": agent_output.status.value,
                "output_data": agent_output.output_data,
                "output_summary": agent_output.output_summary,
                "error_message": agent_output.error_message,
                "retry_count": agent_output.retry_count,
                "completed_at": agent_output.completed_at.isoformat() if agent_output.completed_at else None,
                "agent_info": info
            }
        else:
            outputs[agent_name] = {
                "status": "pending",
                "output_data": None,
                "output_summary": None,
                "error_message": None,
                "retry_count": 0,
                "completed_at": None,
                "agent_info": info
            }

    return {
        "session_id": session_id,
        "agents": outputs
    }
