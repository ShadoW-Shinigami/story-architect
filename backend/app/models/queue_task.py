"""
Queue task model for managing pipeline execution queue
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, DateTime, Text, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
import enum

from app.database import Base


class TaskStatus(str, enum.Enum):
    """Task execution status in the queue."""
    PENDING = "pending"      # Waiting in queue
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"        # Execution failed
    CANCELLED = "cancelled"  # User cancelled


class QueueTask(Base):
    """
    Queue task model for managing pipeline execution order.

    Features:
    - Priority-based ordering (higher = more urgent)
    - Single pipeline execution at a time
    - Resume support (from specific agent)
    """
    __tablename__ = "queue_tasks"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Reference to session
    session_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    # Task status
    status: Mapped[TaskStatus] = mapped_column(
        SQLEnum(TaskStatus),
        default=TaskStatus.PENDING,
        index=True
    )

    # Priority (higher = more urgent, 0 = normal)
    priority: Mapped[int] = mapped_column(Integer, default=0)

    # Task configuration
    start_agent: Mapped[str] = mapped_column(String(20), default="agent_1")
    resume_from: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    def __repr__(self) -> str:
        return f"<QueueTask(id={self.id}, session_id={self.session_id}, status={self.status})>"

    def to_dict(self) -> dict:
        """Convert queue task to dictionary for API responses."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "status": self.status.value,
            "priority": self.priority,
            "start_agent": self.start_agent,
            "resume_from": self.resume_from,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }

    @property
    def wait_time(self) -> Optional[float]:
        """Calculate wait time in seconds if task is still pending."""
        if self.status == TaskStatus.PENDING and self.created_at:
            return (datetime.utcnow() - self.created_at).total_seconds()
        return None

    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time in seconds if task has completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
