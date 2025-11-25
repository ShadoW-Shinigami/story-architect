"""
File serving API endpoints for images and videos
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

router = APIRouter()

# Base outputs directory
OUTPUTS_DIR = Path("outputs/projects")


def get_content_type(file_path: Path) -> str:
    """Get content type based on file extension."""
    suffix = file_path.suffix.lower()
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".json": "application/json",
        ".txt": "text/plain",
    }
    return content_types.get(suffix, "application/octet-stream")


@router.get("/{session_id}/images/{image_path:path}")
async def get_image(session_id: str, image_path: str):
    """
    Serve an image file from a session.

    Supports paths like:
    - characters/char_john.png
    - parent_shots/SHOT_1_1_parent.png
    - child_shots/SHOT_1_2_child.png
    """
    full_path = OUTPUTS_DIR / session_id / image_path

    # Security check - prevent directory traversal
    try:
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(OUTPUTS_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    if not full_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    return FileResponse(
        full_path,
        media_type=get_content_type(full_path),
        filename=full_path.name
    )


@router.get("/{session_id}/videos/{video_path:path}")
async def get_video(
    session_id: str,
    video_path: str,
    stream: bool = Query(default=True, description="Enable streaming response")
):
    """
    Serve a video file from a session with optional streaming.

    Supports paths like:
    - assets/videos/SHOT_1_1_video.mp4
    - assets/edited/master_final.mp4
    - assets/edited/SCENE_1_edited.mp4
    """
    full_path = OUTPUTS_DIR / session_id / video_path

    # Security check
    try:
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(OUTPUTS_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    if not full_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    # Use FileResponse which supports range requests for video streaming
    return FileResponse(
        full_path,
        media_type=get_content_type(full_path),
        filename=full_path.name
    )


@router.get("/{session_id}/raw/{file_path:path}")
async def get_raw_file(session_id: str, file_path: str):
    """
    Serve any raw file from a session (JSON outputs, text files, etc.).
    """
    full_path = OUTPUTS_DIR / session_id / file_path

    # Security check
    try:
        full_path = full_path.resolve()
        if not str(full_path).startswith(str(OUTPUTS_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if not full_path.is_file():
        raise HTTPException(status_code=400, detail="Not a file")

    return FileResponse(
        full_path,
        media_type=get_content_type(full_path),
        filename=full_path.name
    )


@router.get("/{session_id}/list")
async def list_session_files(session_id: str):
    """
    List all files in a session directory.

    Returns a structured overview of available files.
    """
    session_dir = OUTPUTS_DIR / session_id

    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session directory not found")

    files = {
        "characters": [],
        "parent_shots": [],
        "child_shots": [],
        "videos": [],
        "edited": [],
        "other": []
    }

    for file_path in session_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(session_dir)
            file_info = {
                "path": str(relative_path),
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "type": get_content_type(file_path)
            }

            # Categorize files
            path_str = str(relative_path)
            if "characters" in path_str or "grids" in path_str:
                files["characters"].append(file_info)
            elif "parent_shots" in path_str:
                files["parent_shots"].append(file_info)
            elif "child_shots" in path_str:
                files["child_shots"].append(file_info)
            elif "videos" in path_str and "edited" not in path_str:
                files["videos"].append(file_info)
            elif "edited" in path_str:
                files["edited"].append(file_info)
            else:
                files["other"].append(file_info)

    return {
        "session_id": session_id,
        "files": files
    }


@router.get("/{session_id}/download")
async def download_session(session_id: str):
    """
    Download complete session as ZIP archive.
    """
    import io
    import zipfile

    session_dir = OUTPUTS_DIR / session_id

    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session directory not found")

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in session_dir.rglob("*"):
            if file_path.is_file():
                arc_name = file_path.relative_to(session_dir)
                zip_file.write(file_path, arc_name)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={session_id}.zip"
        }
    )
