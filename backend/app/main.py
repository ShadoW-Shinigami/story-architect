"""
Story Architect v2 - FastAPI Backend
Production-grade async API with WebSocket support
"""

# Load .env file FIRST, before any other imports
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import sessions, pipeline, agents, files, queue
from app.api.websocket import router as ws_router
from app.core.queue_manager import QueueManager
from app.database import create_db_and_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    await create_db_and_tables()

    # Initialize queue manager
    app.state.queue_manager = QueueManager()
    await app.state.queue_manager.start()

    # Ensure outputs directory exists
    outputs_dir = Path("outputs/projects")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    await app.state.queue_manager.stop()


app = FastAPI(
    title="Story Architect API",
    description="Multi-agent AI pipeline for script-to-video production",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
#    allow_origins=[
#        "http://localhost:3000",  # Next.js dev server
#        "http://127.0.0.1:3000",
#    ],
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(agents.router, prefix="/api/agents", tags=["agents"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(queue.router, prefix="/api/queue", tags=["queue"])

# WebSocket routes
app.include_router(ws_router, prefix="/ws", tags=["websocket"])

# Static file serving for generated assets (outputs directory)
# This mounts after startup to ensure directory exists
@app.on_event("startup")
async def mount_static():
    outputs_path = Path("outputs/projects")
    if outputs_path.exists():
        app.mount("/outputs", StaticFiles(directory=str(outputs_path)), name="outputs")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Story Architect API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }
