"""
Task queue manager for pipeline execution.

Ensures only ONE pipeline runs at a time with multi-user queuing support.
"""

import asyncio
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy import select, func

from app.database import get_db_context
from app.models.session import Session, SessionStatus
from app.models.queue_task import QueueTask, TaskStatus
from app.api.websocket import get_connection_manager
from app.schemas.progress import QueueUpdatedEvent


class QueueManager:
    """
    Manages pipeline execution queue.

    Features:
    - Single pipeline execution at a time
    - Priority-based ordering
    - Automatic task pickup from database
    - WebSocket notifications for queue updates
    """

    def __init__(self):
        self.current_task: Optional[QueueTask] = None
        self.running = False
        self._task_loop: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._cancel_requested = False

    async def start(self):
        """Start the queue worker loop."""
        self.running = True
        self._task_loop = asyncio.create_task(self._worker_loop())
        logger.info("Queue manager started")

    async def stop(self):
        """Stop the queue worker."""
        self.running = False
        if self._task_loop:
            self._task_loop.cancel()
            try:
                await self._task_loop
            except asyncio.CancelledError:
                pass
        logger.info("Queue manager stopped")

    async def _worker_loop(self):
        """
        Main worker loop - processes queue one task at a time.

        Runs continuously, checking for pending tasks every second.
        """
        while self.running:
            try:
                async with self._lock:
                    if self.current_task is None:
                        # Get next pending task
                        task = await self._get_next_task()
                        if task:
                            await self._execute_task(task)

                # Poll interval - check for new tasks every second
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue worker loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying on error

    async def _get_next_task(self) -> Optional[QueueTask]:
        """Get next pending task by priority and creation time."""
        async with get_db_context() as db:
            result = await db.execute(
                select(QueueTask)
                .where(QueueTask.status == TaskStatus.PENDING)
                .order_by(QueueTask.priority.desc(), QueueTask.created_at.asc())
                .limit(1)
            )
            return result.scalars().first()

    async def _execute_task(self, task: QueueTask):
        """Execute a single pipeline task."""
        self.current_task = task
        self._cancel_requested = False

        logger.info(f"Starting task {task.id} for session {task.session_id}")

        async with get_db_context() as db:
            # Refresh task from database
            result = await db.execute(
                select(QueueTask).where(QueueTask.id == task.id)
            )
            task = result.scalars().first()

            # Update task status to running
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()

            # Update session status
            session_result = await db.execute(
                select(Session).where(Session.id == task.session_id)
            )
            session = session_result.scalars().first()
            if session:
                session.status = SessionStatus.IN_PROGRESS
                session.started_at = datetime.utcnow()

        # Notify queue updated
        await self.notify_queue_updated()

        try:
            # Import here to avoid circular imports
            from app.core.pipeline import AsyncPipeline
            from app.core.progress_tracker import ProgressTracker

            # Create pipeline and progress tracker
            progress_tracker = ProgressTracker(task.session_id)
            pipeline = AsyncPipeline(task.session_id, progress_tracker, queue_manager=self)

            # Run pipeline
            if task.resume_from:
                logger.info(f"Resuming from agent {task.resume_from}")
                await pipeline.resume_from_agent(task.resume_from)
            else:
                logger.info(f"Starting from agent {task.start_agent}")
                await pipeline.run(start_agent=task.start_agent)

            # Mark completed
            async with get_db_context() as db:
                result = await db.execute(
                    select(QueueTask).where(QueueTask.id == task.id)
                )
                task = result.scalars().first()
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()

                # Update session
                session_result = await db.execute(
                    select(Session).where(Session.id == task.session_id)
                )
                session = session_result.scalars().first()
                if session:
                    session.status = SessionStatus.COMPLETED
                    session.completed_at = datetime.utcnow()

            logger.info(f"Task {task.id} completed successfully")

        except asyncio.CancelledError:
            # Task was cancelled
            async with get_db_context() as db:
                result = await db.execute(
                    select(QueueTask).where(QueueTask.id == task.id)
                )
                task = result.scalars().first()
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.utcnow()

                session_result = await db.execute(
                    select(Session).where(Session.id == task.session_id)
                )
                session = session_result.scalars().first()
                if session:
                    session.status = SessionStatus.CANCELLED

            logger.info(f"Task {task.id} cancelled")

        except Exception as e:
            # Task failed
            error_msg = str(e)
            logger.error(f"Task {task.id} failed: {error_msg}")

            async with get_db_context() as db:
                result = await db.execute(
                    select(QueueTask).where(QueueTask.id == task.id)
                )
                task = result.scalars().first()
                task.status = TaskStatus.FAILED
                task.error_message = error_msg
                task.completed_at = datetime.utcnow()

                session_result = await db.execute(
                    select(Session).where(Session.id == task.session_id)
                )
                session = session_result.scalars().first()
                if session:
                    session.status = SessionStatus.FAILED
                    session.error_message = error_msg

        finally:
            self.current_task = None
            await self.notify_queue_updated()

    async def cancel_current_task(self):
        """Signal the current task to cancel."""
        self._cancel_requested = True
        logger.info("Cancel requested for current task")

    def is_cancel_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_requested

    async def add_task(
        self,
        session_id: str,
        start_agent: str = "agent_1",
        resume_from: Optional[str] = None,
        priority: int = 0
    ) -> QueueTask:
        """Add a new task to the queue."""
        async with get_db_context() as db:
            task = QueueTask(
                session_id=session_id,
                start_agent=start_agent,
                resume_from=resume_from,
                priority=priority,
                status=TaskStatus.PENDING
            )
            db.add(task)
            await db.commit()
            await db.refresh(task)

            logger.info(f"Added task {task.id} for session {session_id}")
            return task

    async def notify_queue_updated(self):
        """Broadcast queue status update to all connected clients."""
        async with get_db_context() as db:
            # Get pending count
            result = await db.execute(
                select(func.count(QueueTask.id)).where(
                    QueueTask.status == TaskStatus.PENDING
                )
            )
            pending_count = result.scalar() or 0

        event = QueueUpdatedEvent(
            is_processing=self.current_task is not None,
            current_session_id=self.current_task.session_id if self.current_task else None,
            pending_count=pending_count
        )

        manager = get_connection_manager()
        await manager.broadcast_global(event.to_ws_message())

    async def get_queue_status(self) -> dict:
        """Get current queue status."""
        async with get_db_context() as db:
            result = await db.execute(
                select(func.count(QueueTask.id)).where(
                    QueueTask.status == TaskStatus.PENDING
                )
            )
            pending_count = result.scalar() or 0

        return {
            "is_processing": self.current_task is not None,
            "current_task": {
                "id": self.current_task.id,
                "session_id": self.current_task.session_id
            } if self.current_task else None,
            "pending_count": pending_count
        }
