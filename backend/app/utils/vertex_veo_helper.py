"""
Async Vertex AI Veo 3.1 helper for video generation using Google GenAI SDK.
"""

import asyncio
import time
import json
import base64
import os
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger

try:
    from google import genai
    from google.genai import types
    VERTEX_AVAILABLE = True
except ImportError as e:
    VERTEX_AVAILABLE = False
    logger.warning(f"Required packages not installed: {e}")
    logger.warning("Install with: pip install google-genai")


class AsyncVertexVeoClient:
    """Async client for Vertex AI Veo 3.1 Fast video generation using GenAI SDK."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_id: str = "veo-3.1-fast-generate-preview",
        credentials_file: Optional[str] = None,
    ):
        """
        Initialize Vertex AI Veo client.

        Args:
            project_id: Google Cloud project ID
            location: Location/region for Vertex AI
            model_id: Model ID (default: "veo-3.1-fast-generate-preview")
            credentials_file: Path to service account JSON (optional)
        """
        if not VERTEX_AVAILABLE:
            raise ImportError(
                "Google GenAI SDK not installed. "
                "Install with: pip install google-genai"
            )

        self.project_id = project_id
        self.location = location
        self.model_id = model_id

        # Set credentials if provided
        if credentials_file:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file

        # Temporarily remove API keys to enable Vertex AI mode
        saved_google_key = os.environ.pop("GOOGLE_API_KEY", None)
        saved_gemini_key = os.environ.pop("GEMINI_API_KEY", None)

        if saved_google_key or saved_gemini_key:
            logger.info("Temporarily removed GOOGLE_API_KEY/GEMINI_API_KEY to enable Vertex AI mode")

        # Initialize GenAI Client for Vertex AI
        logger.info(f"Initializing GenAI Client for Vertex AI (project={project_id}, location={location})")
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )

        # Restore API keys
        if saved_google_key:
            os.environ["GOOGLE_API_KEY"] = saved_google_key
        if saved_gemini_key:
            os.environ["GEMINI_API_KEY"] = saved_gemini_key

        logger.info(f"Initialized Vertex AI Veo client with model: {model_id}")

    async def generate_video(
        self,
        prompt: Dict[str, Any],
        image_base64: str,
        aspect_ratio: str = "16:9",
        duration_seconds: int = 6,
    ) -> Dict[str, Any]:
        """
        Generate video using Vertex AI Veo 3.1 Fast (async).

        Args:
            prompt: Production brief dictionary
            image_base64: Base64-encoded starting image
            aspect_ratio: Aspect ratio (default: "16:9")
            duration_seconds: Duration in seconds (default: 6)

        Returns:
            Dictionary with video_base64 and metadata
        """
        logger.info(f"Generating video with Vertex AI Veo (model={self.model_id})")

        # Convert prompt to JSON string
        if isinstance(prompt, dict):
            text_prompt = json.dumps(prompt, indent=2)
        else:
            text_prompt = str(prompt)

        logger.info(f"Using full structured prompt ({len(text_prompt)} chars)")

        # Prepare image
        try:
            raw_bytes = base64.b64decode(image_base64)
            image_input = types.Image(
                image_bytes=raw_bytes,
                mime_type="image/png"
            )
            logger.info(f"Prepared types.Image with {len(raw_bytes)} bytes")
        except Exception as e:
            raise ValueError(f"Failed to prepare image: {e}")

        # Submit video generation request
        logger.info("Submitting video generation request...")

        try:
            if duration_seconds <= 0:
                logger.warning(f"Invalid duration_seconds={duration_seconds}, defaulting to 6")
                duration_seconds = 6

            def _generate():
                return self.client.models.generate_videos(
                    model=self.model_id,
                    prompt=text_prompt,
                    image=image_input,
                    config=types.GenerateVideosConfig(
                        number_of_videos=1,
                        duration_seconds=duration_seconds,
                        aspect_ratio=aspect_ratio,
                    )
                )

            operation = await asyncio.to_thread(_generate)
            logger.info(f"Operation started: {operation.name}")

            # Poll until complete
            while not operation.done:
                logger.info("Waiting for video generation to complete...")
                await asyncio.sleep(10)

                def _get_operation():
                    return self.client.operations.get(operation)

                operation = await asyncio.to_thread(_get_operation)

            # Check for errors
            if operation.error:
                raise ValueError(f"Video generation failed: {operation.error}")

            # Get result
            if not operation.response or not operation.response.generated_videos:
                raise ValueError("No videos generated in response")

            generated_video = operation.response.generated_videos[0]
            video_bytes = generated_video.video.video_bytes

            # Re-encode to base64
            result_b64 = base64.b64encode(video_bytes).decode('utf-8')

            logger.info("✓ Video generated successfully")

            return {
                "video_base64": result_b64,
                "duration": f"{duration_seconds}s",
                "aspect_ratio": aspect_ratio,
                "prompt_text": text_prompt,
                "model": self.model_id,
                "operation_name": operation.name
            }

        except Exception as e:
            logger.error(f"Vertex AI Veo error: {str(e)}")
            raise

    async def save_video_from_base64(self, video_base64: str, output_path: Path) -> None:
        """Save base64-encoded video to file (async)."""
        def _save():
            video_bytes = base64.b64decode(video_base64)
            with open(output_path, 'wb') as f:
                f.write(video_bytes)
            return len(video_bytes)

        size = await asyncio.to_thread(_save)
        logger.info(f"Saved video to {output_path} ({size / (1024*1024):.2f} MB)")


def is_vertex_available() -> bool:
    """Check if Vertex AI SDK is available."""
    return VERTEX_AVAILABLE
