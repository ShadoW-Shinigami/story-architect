"""
Queue status API endpoints
"""

from typing import List

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.queue_task import QueueTask, TaskStatus
from app.schemas.queue import QueueStatusResponse, QueueTaskResponse

router = APIRouter()


@router.get("/status", response_model=QueueStatusResponse)
async def get_queue_status(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """
    Get current queue status including running task and pending count.
    """
    queue_manager = request.app.state.queue_manager

    # Get pending count
    pending_result = await db.execute(
        select(func.count(QueueTask.id)).where(
            QueueTask.status == TaskStatus.PENDING
        )
    )
    pending_count = pending_result.scalar() or 0

    # Get current task info
    current_task = None
    if queue_manager.current_task:
        current_task = {
            "id": queue_manager.current_task.id,
            "session_id": queue_manager.current_task.session_id,
            "started_at": queue_manager.current_task.started_at.isoformat()
            if queue_manager.current_task.started_at else None
        }

    return QueueStatusResponse(
        is_processing=queue_manager.current_task is not None,
        current_task=current_task,
        pending_count=pending_count
    )


@router.get("/tasks", response_model=List[QueueTaskResponse])
async def get_queue_tasks(
    status: TaskStatus = None,
    limit: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """
    Get list of queue tasks with optional status filter.
    """
    query = select(QueueTask).order_by(
        QueueTask.priority.desc(),
        QueueTask.created_at.asc()
    ).limit(limit)

    if status:
        query = query.where(QueueTask.status == status)

    result = await db.execute(query)
    tasks = result.scalars().all()

    return [
        QueueTaskResponse(
            id=t.id,
            session_id=t.session_id,
            status=t.status.value,
            priority=t.priority,
            start_agent=t.start_agent,
            resume_from=t.resume_from,
            created_at=t.created_at,
            started_at=t.started_at,
            completed_at=t.completed_at,
            error_message=t.error_message
        )
        for t in tasks
    ]


@router.get("/history")
async def get_queue_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """
    Get recently completed/failed tasks for history display.
    """
    result = await db.execute(
        select(QueueTask).where(
            QueueTask.status.in_([
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED
            ])
        ).order_by(QueueTask.completed_at.desc()).limit(limit)
    )
    tasks = result.scalars().all()

    return [t.to_dict() for t in tasks]


@router.get("/position/{session_id}")
async def get_queue_position(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get queue position for a specific session.
    """
    # Get all pending tasks ordered by priority and creation time
    result = await db.execute(
        select(QueueTask).where(
            QueueTask.status == TaskStatus.PENDING
        ).order_by(
            QueueTask.priority.desc(),
            QueueTask.created_at.asc()
        )
    )
    pending_tasks = result.scalars().all()

    # Find position of this session's task
    for i, task in enumerate(pending_tasks):
        if task.session_id == session_id:
            return {
                "session_id": session_id,
                "position": i + 1,
                "total_pending": len(pending_tasks)
            }

    # Check if task is running
    running_result = await db.execute(
        select(QueueTask).where(
            QueueTask.session_id == session_id,
            QueueTask.status == TaskStatus.RUNNING
        )
    )
    if running_result.scalars().first():
        return {
            "session_id": session_id,
            "position": 0,  # Currently running
            "total_pending": len(pending_tasks)
        }

    return {
        "session_id": session_id,
        "position": None,  # Not in queue
        "total_pending": len(pending_tasks)
    }
