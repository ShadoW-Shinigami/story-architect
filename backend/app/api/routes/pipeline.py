"""
Pipeline control API endpoints
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.session import Session, SessionStatus
from app.models.queue_task import QueueTask, TaskStatus
from app.schemas.pipeline import PipelineStartRequest, PipelineStatusResponse

router = APIRouter()


@router.post("/{session_id}/start")
async def start_pipeline(
    session_id: str,
    request: Request,
    data: PipelineStartRequest = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Start pipeline execution for a session.

    Adds the task to the queue - execution begins when the queue worker picks it up.
    """
    # Verify session exists
    result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if session is already in queue or running
    existing_task = await db.execute(
        select(QueueTask).where(
            QueueTask.session_id == session_id,
            QueueTask.status.in_([TaskStatus.PENDING, TaskStatus.RUNNING])
        )
    )
    if existing_task.scalars().first():
        raise HTTPException(
            status_code=400,
            detail="Session is already queued or running"
        )

    # Create queue task
    start_agent = data.start_agent if data else session.start_agent
    priority = data.priority if data else 0

    task = QueueTask(
        session_id=session_id,
        start_agent=start_agent,
        priority=priority,
        status=TaskStatus.PENDING
    )
    db.add(task)

    # Update session status
    session.status = SessionStatus.QUEUED
    session.start_agent = start_agent

    await db.commit()

    # Get queue manager and broadcast update
    queue_manager = request.app.state.queue_manager
    await queue_manager.notify_queue_updated()

    return {
        "task_id": task.id,
        "session_id": session_id,
        "status": "queued",
        "start_agent": start_agent
    }


@router.post("/{session_id}/resume")
async def resume_pipeline(
    session_id: str,
    from_agent: str,
    request: Request,
    priority: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """
    Resume pipeline execution from a specific agent.

    Useful for retrying failed agents or regenerating outputs.
    """
    # Verify session exists
    result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Validate agent name
    valid_agents = [f"agent_{i}" for i in range(1, 12)]
    if from_agent not in valid_agents:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent name. Must be one of: {valid_agents}"
        )

    # Check if session is already in queue or running
    existing_task = await db.execute(
        select(QueueTask).where(
            QueueTask.session_id == session_id,
            QueueTask.status.in_([TaskStatus.PENDING, TaskStatus.RUNNING])
        )
    )
    if existing_task.scalars().first():
        raise HTTPException(
            status_code=400,
            detail="Session is already queued or running"
        )

    # Create queue task with resume_from
    task = QueueTask(
        session_id=session_id,
        start_agent=session.start_agent,
        resume_from=from_agent,
        priority=priority,
        status=TaskStatus.PENDING
    )
    db.add(task)

    # Update session status
    session.status = SessionStatus.QUEUED

    await db.commit()

    # Notify queue updated
    queue_manager = request.app.state.queue_manager
    await queue_manager.notify_queue_updated()

    return {
        "task_id": task.id,
        "session_id": session_id,
        "status": "queued",
        "resume_from": from_agent
    }


@router.post("/{session_id}/cancel")
async def cancel_pipeline(
    session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel a queued or running pipeline.
    """
    # Find pending/running task for this session
    result = await db.execute(
        select(QueueTask).where(
            QueueTask.session_id == session_id,
            QueueTask.status.in_([TaskStatus.PENDING, TaskStatus.RUNNING])
        )
    )
    task = result.scalars().first()

    if not task:
        raise HTTPException(
            status_code=404,
            detail="No active task found for this session"
        )

    # Update task status
    task.status = TaskStatus.CANCELLED

    # Update session status
    session_result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = session_result.scalars().first()
    if session:
        session.status = SessionStatus.CANCELLED

    await db.commit()

    # If task was running, signal queue manager to stop
    queue_manager = request.app.state.queue_manager
    if queue_manager.current_task and queue_manager.current_task.id == task.id:
        await queue_manager.cancel_current_task()

    return {
        "status": "cancelled",
        "session_id": session_id,
        "task_id": task.id
    }


@router.get("/{session_id}/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get current pipeline execution status.
    """
    result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get completed agents
    completed_agents = [
        ao.agent_name for ao in session.agent_outputs
        if ao.status.value in ["completed", "soft_failure"]
    ]

    # Get queue position if queued
    queue_position = None
    if session.status == SessionStatus.QUEUED:
        queue_result = await db.execute(
            select(QueueTask).where(
                QueueTask.status == TaskStatus.PENDING
            ).order_by(QueueTask.priority.desc(), QueueTask.created_at.asc())
        )
        pending_tasks = queue_result.scalars().all()
        for i, task in enumerate(pending_tasks):
            if task.session_id == session_id:
                queue_position = i + 1
                break

    return PipelineStatusResponse(
        session_id=session_id,
        status=session.status.value,
        current_agent=session.current_agent,
        completed_agents=completed_agents,
        error_message=session.error_message,
        queue_position=queue_position
    )
