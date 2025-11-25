"""
Agent output model for storing individual agent results
"""

from datetime import datetime
from typing import Optional, Any

from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
import enum

from app.database import Base


class AgentStatus(str, enum.Enum):
    """Agent execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SOFT_FAILURE = "soft_failure"  # Verification agents - passed with issues


class AgentOutput(Base):
    """
    Agent output model storing results from each pipeline agent.

    Each output contains:
    - Agent identification (name)
    - Output data (JSON for structured data, text for screenplay)
    - Execution metadata (status, retry count, timestamps)
    """
    __tablename__ = "agent_outputs"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign key to session
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Agent identification
    agent_name: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Status tracking
    status: Mapped[AgentStatus] = mapped_column(
        SQLEnum(AgentStatus),
        default=AgentStatus.PENDING
    )

    # Output data - stored as JSON (flexible for all agent types)
    # For agent_1 (screenplay), this contains {"text": "screenplay content"}
    # For agents 2-11, this contains the structured JSON output
    output_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Output summary for quick display
    output_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

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

    # Relationship back to session
    session: Mapped["Session"] = relationship("Session", back_populates="agent_outputs")

    def __repr__(self) -> str:
        return f"<AgentOutput(session_id={self.session_id}, agent={self.agent_name}, status={self.status})>"

    def to_dict(self) -> dict:
        """Convert agent output to dictionary for API responses."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "output_data": self.output_data,
            "output_summary": self.output_summary,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
