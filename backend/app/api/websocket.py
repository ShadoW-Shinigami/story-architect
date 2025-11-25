"""
WebSocket endpoints for real-time progress updates
"""

import json
from typing import Dict, Set
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.

    Supports:
    - Session-specific connections (for progress updates)
    - Global connections (for queue status updates)
    """

    def __init__(self):
        # Map session_id -> set of WebSocket connections
        self.session_connections: Dict[str, Set[WebSocket]] = {}
        # Global connections (queue updates, etc.)
        self.global_connections: Set[WebSocket] = set()
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect_session(self, websocket: WebSocket, session_id: str):
        """Connect a client to a specific session's updates."""
        await websocket.accept()
        async with self._lock:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = set()
            self.session_connections[session_id].add(websocket)
        logger.info(f"WebSocket connected to session {session_id}")

    async def connect_global(self, websocket: WebSocket):
        """Connect a client to global updates (queue status, etc.)."""
        await websocket.accept()
        async with self._lock:
            self.global_connections.add(websocket)
        logger.info("WebSocket connected to global updates")

    async def disconnect_session(self, websocket: WebSocket, session_id: str):
        """Disconnect a client from a session's updates."""
        async with self._lock:
            if session_id in self.session_connections:
                self.session_connections[session_id].discard(websocket)
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
        logger.info(f"WebSocket disconnected from session {session_id}")

    async def disconnect_global(self, websocket: WebSocket):
        """Disconnect a client from global updates."""
        async with self._lock:
            self.global_connections.discard(websocket)
        logger.info("WebSocket disconnected from global updates")

    async def broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast a message to all clients watching a specific session."""
        async with self._lock:
            connections = self.session_connections.get(session_id, set()).copy()

        if not connections:
            return

        disconnected = set()
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.add(websocket)

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                if session_id in self.session_connections:
                    self.session_connections[session_id] -= disconnected

    async def broadcast_global(self, message: dict):
        """Broadcast a message to all global connections."""
        async with self._lock:
            connections = self.global_connections.copy()

        if not connections:
            return

        disconnected = set()
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to global WebSocket: {e}")
                disconnected.add(websocket)

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                self.global_connections -= disconnected

    async def get_session_connection_count(self, session_id: str) -> int:
        """Get number of clients watching a session."""
        async with self._lock:
            return len(self.session_connections.get(session_id, set()))

    async def get_global_connection_count(self) -> int:
        """Get number of global connections."""
        async with self._lock:
            return len(self.global_connections)


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/session/{session_id}")
async def session_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for session-specific progress updates.

    Clients connect here to receive real-time updates for a specific session:
    - Agent progress events
    - Agent completion notifications
    - Error notifications

    Message format:
    {
        "type": "progress" | "agent_completed" | "agent_failed" | "pipeline_completed" | "pipeline_failed",
        "data": { ... event-specific data ... }
    }
    """
    await manager.connect_session(websocket, session_id)

    try:
        while True:
            # Receive client messages (for ping/pong, commands, etc.)
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                # Handle client commands
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "subscribe":
                    # Already subscribed via connection
                    await websocket.send_json({
                        "type": "subscribed",
                        "session_id": session_id
                    })

            except json.JSONDecodeError:
                # Ignore invalid JSON
                pass

    except WebSocketDisconnect:
        await manager.disconnect_session(websocket, session_id)


@router.websocket("/queue")
async def queue_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for global queue status updates.

    Clients connect here to receive updates about:
    - Queue position changes
    - New tasks added
    - Tasks completed/failed

    Message format:
    {
        "type": "queue_updated" | "task_started" | "task_completed" | "task_failed",
        "data": { ... event-specific data ... }
    }
    """
    await manager.connect_global(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        await manager.disconnect_global(websocket)


# Export manager for use in other modules
def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    return manager
