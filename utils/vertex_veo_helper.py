"""
Vertex AI Veo 3.1 helper for video generation using Google GenAI SDK.
"""

import time
import json
import base64
import io
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


class VertexVeoClient:
    """Client for Vertex AI Veo 3.1 Fast video generation using GenAI SDK."""
    
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_id: str = "veo-3.1-fast-generate-preview",
        credentials_file: Optional[str] = None,
        gcs_bucket: Optional[str] = None,  # kept for backwards-compat, not required
    ):
        """
        Initialize Vertex AI Veo client.
        
        Args:
            project_id: Google Cloud project ID
            location: Location/region for Vertex AI
            model_id: Model ID (default: "veo-3.1-fast-generate-preview")
            credentials_file: Path to service account JSON (optional)
            gcs_bucket: (Deprecated/optional) GCS bucket name. No longer required since
                images are sent inline via types.Image(image_bytes=...).
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
            import os
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
        
        # For Vertex AI mode, temporarily remove API keys that trigger Gemini Developer mode
        # The GenAI SDK prioritizes GOOGLE_API_KEY/GEMINI_API_KEY over vertexai=True
        import os
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
        
        # Restore API keys for other parts of the pipeline
        if saved_google_key:
            os.environ["GOOGLE_API_KEY"] = saved_google_key
        if saved_gemini_key:
            os.environ["GEMINI_API_KEY"] = saved_gemini_key
        
        logger.info(f"Initialized Vertex AI Veo client with model: {model_id}")
    
    def generate_video(
        self,
        prompt: Dict[str, Any],
        image_base64: str,
        aspect_ratio: str = "16:9",
        duration: str = "6s"
    ) -> Dict[str, Any]:
        """
        Generate video using Vertex AI Veo 3.1 Fast.
        """
        logger.info(f"Generating video with Vertex AI Veo (model={self.model_id})")
        
        # Use the full prompt dictionary as the prompt text (JSON string)
        # This ensures the model gets the complete context without any extraction/filtering
        if isinstance(prompt, dict):
            text_prompt = json.dumps(prompt, indent=2)
        else:
            text_prompt = str(prompt)
            
        logger.info(f"Using full structured prompt ({len(text_prompt)} chars)")
        
        # Prepare image as types.Image using raw bytes
        try:
            raw_bytes = base64.b64decode(image_base64)
            image_input = types.Image(
                image_bytes=raw_bytes,
                mime_type="image/png"
            )
            logger.info(f"Prepared types.Image with {len(raw_bytes)} bytes")
            
        except Exception as e:
            raise ValueError(f"Failed to prepare image: {e}")

        # Call generate_videos with prepared image
        logger.info("Submitting video generation request...")
        try:
            operation = self.client.models.generate_videos(
                model=self.model_id,
                prompt=text_prompt,
                image=image_input,
                config=types.GenerateVideosConfig(
                    number_of_videos=1
                )
            )
            
            logger.info(f"Operation started: {operation.name}")
            
            # Poll until complete
            while not operation.done:
                logger.info("Waiting for video generation to complete...")
                time.sleep(10)
                operation = self.client.operations.get(operation)
            
            # Check for errors
            if operation.error:
                raise ValueError(f"Video generation failed: {operation.error}")
            
            # Get result
            if not operation.response or not operation.response.generated_videos:
                raise ValueError("No videos generated in response")
                
            generated_video = operation.response.generated_videos[0]
            video_bytes = generated_video.video.video_bytes
            
            # Re-encode to base64 for consistency with interface
            result_b64 = base64.b64encode(video_bytes).decode('utf-8')
            
            logger.info("✓ Video generated successfully")
            
            return {
                "video_base64": result_b64,
                "duration": duration,
                "aspect_ratio": aspect_ratio,
                "prompt_text": text_prompt,
                "model": self.model_id,
                "operation_name": operation.name
            }

        except Exception as e:
            logger.error(f"Vertex AI Veo error: {str(e)}")
            raise

    def save_video_from_base64(self, video_base64: str, output_path: Path) -> None:
        """
        Save base64-encoded video to file.
        """
        video_bytes = base64.b64decode(video_base64)
        with open(output_path, 'wb') as f:
            f.write(video_bytes)
        logger.info(f"Saved video to {output_path} ({len(video_bytes) / (1024*1024):.2f} MB)")
