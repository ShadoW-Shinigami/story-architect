"""
Session model for storing pipeline session state
"""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, Text, DateTime, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.database import Base


class SessionStatus(str, enum.Enum):
    """Session execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Session(Base):
    """
    Session model representing a pipeline execution instance.

    Each session contains:
    - Session metadata (name, timestamps)
    - Input data (logline/screenplay)
    - Execution state (status, current agent)
    - Related agent outputs
    """
    __tablename__ = "sessions"

    # Primary key - UUID string for URL-friendliness
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    # Session metadata
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    input_data: Mapped[str] = mapped_column(Text, nullable=False)

    # Execution configuration
    start_agent: Mapped[str] = mapped_column(String(20), default="agent_1")
    current_agent: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Status tracking
    status: Mapped[SessionStatus] = mapped_column(
        SQLEnum(SessionStatus),
        default=SessionStatus.PENDING
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    agent_outputs: Mapped[List["AgentOutput"]] = relationship(
        "AgentOutput",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Session(id={self.id}, name={self.name}, status={self.status})>"

    @property
    def session_dir(self) -> str:
        """Get the session output directory path."""
        return f"outputs/projects/{self.id}"

    def to_dict(self) -> dict:
        """Convert session to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "input_data": self.input_data,
            "start_agent": self.start_agent,
            "current_agent": self.current_agent,
            "status": self.status.value,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "agent_outputs": {
                ao.agent_name: ao.to_dict()
                for ao in self.agent_outputs
            } if self.agent_outputs else {}
        }
