"""
Progress tracker for real-time pipeline updates.

Emits progress events via WebSocket to connected frontend clients.
"""

from datetime import datetime
from typing import Optional, Any, Callable, Awaitable

from loguru import logger

from app.api.websocket import get_connection_manager
from app.schemas.progress import (
    ProgressEvent,
    ProgressEventType,
    AgentCompletedEvent,
    PipelineCompletedEvent
)


# Agent weights for overall progress calculation
AGENT_WEIGHTS = {
    "agent_1": 0.05,   # Screenplay - fast
    "agent_2": 0.05,   # Scene breakdown - fast
    "agent_3": 0.05,   # Shot breakdown - fast
    "agent_4": 0.05,   # Shot grouping - fast
    "agent_5": 0.10,   # Character images - moderate
    "agent_6": 0.15,   # Parent images - slower
    "agent_7": 0.05,   # Parent verification - moderate
    "agent_8": 0.15,   # Child images - slower
    "agent_9": 0.05,   # Child verification - moderate
    "agent_10": 0.20,  # Video generation - slow
    "agent_11": 0.10,  # Video editing - moderate
}

AGENT_DISPLAY_NAMES = {
    "agent_1": "Screenplay Generator",
    "agent_2": "Scene Breakdown",
    "agent_3": "Shot Breakdown",
    "agent_4": "Shot Grouping",
    "agent_5": "Character Creator",
    "agent_6": "Parent Image Generator",
    "agent_7": "Parent Verification",
    "agent_8": "Child Image Generator",
    "agent_9": "Child Verification",
    "agent_10": "Video Dialogue Generator",
    "agent_11": "Intelligent Video Editor",
}


class ProgressTracker:
    """
    Tracks and broadcasts pipeline progress via WebSocket.

    Features:
    - Per-agent progress tracking
    - Overall pipeline progress calculation
    - Real-time WebSocket broadcasting
    - Step-level progress for detailed updates
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.completed_agents: list[str] = []
        self.current_agent: Optional[str] = None
        self.pipeline_started_at: Optional[datetime] = None

    async def emit(self, event: ProgressEvent):
        """Emit a progress event via WebSocket."""
        # Set session ID
        event.session_id = self.session_id

        # Calculate overall progress
        event.overall_progress = self._calculate_overall_progress(event)

        # Add display name if not set
        if event.agent_name and not event.agent_display_name:
            event.agent_display_name = AGENT_DISPLAY_NAMES.get(event.agent_name)

        # Broadcast to session
        manager = get_connection_manager()
        await manager.broadcast_to_session(self.session_id, event.to_ws_message())

        logger.debug(f"Progress event: {event.event_type} - {event.message}")

    def _calculate_overall_progress(self, event: ProgressEvent) -> float:
        """
        Calculate overall pipeline progress.

        Based on:
        - Completed agents (full weight)
        - Current agent progress (partial weight)
        """
        # Sum completed agent weights
        completed_progress = sum(
            AGENT_WEIGHTS.get(agent, 0)
            for agent in self.completed_agents
        )

        # Add current agent partial progress
        if event.agent_name:
            current_weight = AGENT_WEIGHTS.get(event.agent_name, 0)
            current_progress = event.progress * current_weight
            return min(completed_progress + current_progress, 1.0)

        return min(completed_progress, 1.0)

    async def pipeline_started(self):
        """Emit pipeline started event."""
        self.pipeline_started_at = datetime.utcnow()
        self.completed_agents = []

        await self.emit(ProgressEvent(
            event_type=ProgressEventType.PIPELINE_STARTED,
            session_id=self.session_id,
            message="Pipeline execution started",
            progress=0.0
        ))

    async def pipeline_completed(self):
        """Emit pipeline completed event."""
        duration = None
        if self.pipeline_started_at:
            duration = (datetime.utcnow() - self.pipeline_started_at).total_seconds()

        event = PipelineCompletedEvent(
            session_id=self.session_id,
            total_duration_seconds=duration or 0,
            agents_completed=self.completed_agents
        )

        manager = get_connection_manager()
        await manager.broadcast_to_session(self.session_id, event.to_ws_message())

        logger.info(f"Pipeline completed for session {self.session_id}")

    async def pipeline_failed(self, error: str):
        """Emit pipeline failed event."""
        await self.emit(ProgressEvent(
            event_type=ProgressEventType.PIPELINE_FAILED,
            session_id=self.session_id,
            message=f"Pipeline failed: {error}",
            error=error
        ))

        logger.error(f"Pipeline failed for session {self.session_id}: {error}")

    async def agent_started(self, agent_name: str):
        """Emit agent started event."""
        self.current_agent = agent_name

        await self.emit(ProgressEvent(
            event_type=ProgressEventType.AGENT_STARTED,
            session_id=self.session_id,
            agent_name=agent_name,
            message=f"Starting {AGENT_DISPLAY_NAMES.get(agent_name, agent_name)}",
            progress=0.0
        ))

    async def agent_completed(
        self,
        agent_name: str,
        output_summary: Optional[str] = None,
        duration: Optional[float] = None
    ):
        """Emit agent completed event."""
        self.completed_agents.append(agent_name)
        self.current_agent = None

        event = AgentCompletedEvent(
            session_id=self.session_id,
            agent_name=agent_name,
            output_summary=output_summary,
            duration_seconds=duration
        )

        manager = get_connection_manager()
        await manager.broadcast_to_session(self.session_id, event.to_ws_message())

        logger.info(f"Agent {agent_name} completed for session {self.session_id}")

    async def agent_failed(self, agent_name: str, error: str, retry_count: int = 0):
        """Emit agent failed event."""
        await self.emit(ProgressEvent(
            event_type=ProgressEventType.AGENT_FAILED,
            session_id=self.session_id,
            agent_name=agent_name,
            message=f"{AGENT_DISPLAY_NAMES.get(agent_name, agent_name)} failed: {error}",
            error=error,
            retry_count=retry_count
        ))

    async def agent_retry(self, agent_name: str, retry_count: int, error: str):
        """Emit agent retry event."""
        await self.emit(ProgressEvent(
            event_type=ProgressEventType.AGENT_RETRY,
            session_id=self.session_id,
            agent_name=agent_name,
            message=f"Retrying {AGENT_DISPLAY_NAMES.get(agent_name, agent_name)} (attempt {retry_count + 1})",
            error=error,
            retry_count=retry_count
        ))

    async def step_progress(
        self,
        agent_name: str,
        message: str,
        progress: float,
        step_name: Optional[str] = None,
        step_current: Optional[int] = None,
        step_total: Optional[int] = None
    ):
        """Emit step progress event (for detailed per-item updates)."""
        await self.emit(ProgressEvent(
            event_type=ProgressEventType.STEP_PROGRESS,
            session_id=self.session_id,
            agent_name=agent_name,
            message=message,
            progress=progress,
            step_name=step_name,
            step_current=step_current,
            step_total=step_total
        ))

    async def image_generated(
        self,
        agent_name: str,
        image_path: str,
        message: str = "Image generated",
        thumbnail_path: Optional[str] = None
    ):
        """Emit image generated event (for immediate display)."""
        await self.emit(ProgressEvent(
            event_type=ProgressEventType.IMAGE_GENERATED,
            session_id=self.session_id,
            agent_name=agent_name,
            message=message,
            output_path=image_path,
            output_thumbnail=thumbnail_path
        ))

    async def video_generated(
        self,
        agent_name: str,
        video_path: str,
        message: str = "Video generated"
    ):
        """Emit video generated event."""
        await self.emit(ProgressEvent(
            event_type=ProgressEventType.VIDEO_GENERATED,
            session_id=self.session_id,
            agent_name=agent_name,
            message=message,
            output_path=video_path
        ))

    def get_progress_callback(
        self,
        agent_name: str
    ) -> Callable[[str, float, Optional[int], Optional[int]], Awaitable[None]]:
        """
        Get a callback function for agent-internal progress updates.

        Usage in agents:
            callback = progress_tracker.get_progress_callback("agent_5")
            await callback("Generated character 3/10", 0.3, 3, 10)
        """
        async def callback(
            message: str,
            progress: float,
            current: Optional[int] = None,
            total: Optional[int] = None
        ):
            await self.step_progress(
                agent_name=agent_name,
                message=message,
                progress=progress,
                step_current=current,
                step_total=total
            )

        return callback
