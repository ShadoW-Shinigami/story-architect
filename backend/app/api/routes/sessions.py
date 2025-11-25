"""
Session management API endpoints
"""

import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.session import Session, SessionStatus
from app.schemas.session import SessionCreate, SessionResponse, SessionListResponse

router = APIRouter()


@router.get("/", response_model=List[SessionListResponse])
async def list_sessions(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[SessionStatus] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all sessions with pagination.

    Shared workspace - all users see all sessions.
    """
    query = select(Session).order_by(Session.updated_at.desc())

    if status:
        query = query.where(Session.status == status)

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    sessions = result.scalars().all()

    return [
        SessionListResponse(
            id=s.id,
            name=s.name,
            status=s.status.value,
            current_agent=s.current_agent,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in sessions
    ]


@router.get("/count")
async def get_session_count(
    status: Optional[SessionStatus] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get total count of sessions."""
    query = select(func.count(Session.id))
    if status:
        query = query.where(Session.status == status)
    result = await db.execute(query)
    return {"count": result.scalar()}


@router.get("/stats/overview")
async def get_session_stats(db: AsyncSession = Depends(get_db)):
    """Get overview statistics for dashboard."""
    # Total count
    total_result = await db.execute(select(func.count(Session.id)))
    total = total_result.scalar() or 0

    # Count by status
    status_counts = {}
    for status in SessionStatus:
        count_result = await db.execute(
            select(func.count(Session.id)).where(Session.status == status)
        )
        status_counts[status.value] = count_result.scalar() or 0

    # Recent activity (last 7 days)
    from datetime import datetime, timedelta
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_result = await db.execute(
        select(func.count(Session.id)).where(Session.created_at >= week_ago)
    )
    recent_count = recent_result.scalar() or 0

    return {
        "total_projects": total,
        "status_breakdown": status_counts,
        "recent_projects": recent_count,
        "active_count": status_counts.get("in_progress", 0) + status_counts.get("queued", 0),
        "completed_count": status_counts.get("completed", 0),
        "failed_count": status_counts.get("failed", 0),
    }


@router.post("/", response_model=SessionResponse)
async def create_session(
    data: SessionCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new session.

    This only creates the session record - use /api/pipeline/{id}/start to begin execution.
    """
    session = Session(
        name=data.name,
        input_data=data.input_data,
        start_agent=data.start_agent or "agent_1",
        status=SessionStatus.PENDING
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # Create session directory
    session_dir = Path(session.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    return SessionResponse.model_validate(session.to_dict())


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get session details including all agent outputs.
    """
    result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse.model_validate(session.to_dict())


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    name: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Update session metadata (currently only name).
    """
    result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if name is not None:
        session.name = name

    await db.commit()
    await db.refresh(session)

    return SessionResponse.model_validate(session.to_dict())


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a session and all associated files.
    """
    result = await db.execute(
        select(Session).where(Session.id == session_id)
    )
    session = result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete session directory and files
    session_dir = Path(session.session_dir)
    if session_dir.exists():
        shutil.rmtree(session_dir)

    # Delete from database (cascades to agent_outputs)
    await db.delete(session)
    await db.commit()

    return {"status": "deleted", "session_id": session_id}
