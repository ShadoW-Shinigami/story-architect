"""
Agent 10: Async Video Dialogue Generator
Generates video clips from verified shot images using FAL AI Veo 3.1 or Vertex AI Veo.
Creates production briefs for each shot with character and motion guidance.
Supports automatic aspect ratio detection and image optimization for base64 encoding.
"""

import asyncio
import json
import os
import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path
from PIL import Image
from loguru import logger
import httpx

from app.agents.base_agent import AsyncBaseAgent
from app.core.gemini_client import AsyncGeminiClient
from app.utils.image_utils import optimize_image_for_base64, get_aspect_ratio_from_image
from app.utils.vertex_veo_helper import AsyncVertexVeoClient, is_vertex_available

# Try to import fal_client
try:
    import fal_client
    FAL_AVAILABLE = True
except ImportError:
    FAL_AVAILABLE = False
    logger.warning("fal_client not installed. FAL video generation unavailable.")


class VideoDialogueGeneratorAgent(AsyncBaseAgent):
    """Async agent for generating video clips from shot images using FAL AI Veo 3.1 or Vertex AI Veo."""

    def __init__(
        self,
        session_id: str,
        config: dict,
        gemini_client: AsyncGeminiClient
    ):
        super().__init__(
            agent_name="agent_10",
            session_id=session_id,
            config=config
        )
        self.client = gemini_client
        self.videos_dir = self._ensure_directory("assets/videos")
        self.briefs_dir = self._ensure_directory("assets/video_briefs")

        # Video provider configuration (fal or vertex_ai)
        self.video_provider = config.get("video_provider", "fal").lower()
        self.aspect_ratio = config.get("aspect_ratio", "auto")
        self.max_image_size_mb = config.get("max_image_size_mb", 7.0)
        self.video_duration = config.get("video_duration", 6)
        self.max_retries = config.get("max_retries", 3)

        # FAL configuration
        self.fal_client = None
        if self.video_provider == "fal":
            self.fal_api_key = config.get("fal_api_key") or os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
            if not self.fal_api_key and FAL_AVAILABLE:
                logger.warning(f"{self.agent_name}: FAL_KEY not found, FAL video generation may fail")

            self.video_resolution = config.get("video_resolution", "1080p")
            self.generate_audio = config.get("generate_audio", True)

            if FAL_AVAILABLE:
                os.environ["FAL_KEY"] = self.fal_api_key or ""
                self.fal_client = fal_client
                logger.info(f"{self.agent_name}: FAL client initialized for video generation")
            else:
                logger.warning(f"{self.agent_name}: fal_client not available, will fall back to Vertex AI")
                self.video_provider = "vertex_ai"

        # Vertex AI configuration
        self.veo_client = None
        if self.video_provider == "vertex_ai" or not FAL_AVAILABLE:
            self.project_id = config.get("vertex_project_id") or os.getenv("GOOGLE_CLOUD_PROJECT")
            self.location = config.get("vertex_location", "us-central1")
            self.veo_model = config.get("veo_model", "veo-3.1-fast-generate-preview")

            if self.project_id and is_vertex_available():
                try:
                    self.veo_client = AsyncVertexVeoClient(
                        project_id=self.project_id,
                        location=self.location,
                        model_id=self.veo_model,
                    )
                    logger.info(f"{self.agent_name}: Vertex AI Veo client initialized")
                except Exception as e:
                    logger.warning(f"{self.agent_name}: Failed to initialize Veo client: {e}")

    async def validate_input(self, input_data: Any) -> None:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = [
            "parent_shots", "child_shots",
            "scene_breakdown", "shot_breakdown", "shot_grouping"
        ]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Input must contain '{key}'")

    async def validate_output(self, output_data: Any) -> None:
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        if "videos" not in output_data:
            raise ValueError("Output must contain 'videos'")

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Generate video clips from verified shot images."""
        logger.info(f"{self.agent_name}: Generating video clips from shot images...")

        # Extract parent_shots list from Agent 7's full output dict
        agent_7_output = input_data.get("parent_shots", {})
        parent_shots = agent_7_output.get("parent_shots", []) if isinstance(agent_7_output, dict) else agent_7_output

        # Extract child_shots list from Agent 9's full output dict
        agent_9_output = input_data.get("child_shots", {})
        child_shots = agent_9_output.get("child_shots", []) if isinstance(agent_9_output, dict) else agent_9_output

        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]

        if progress_callback:
            await progress_callback("Preparing shot data...", 0.05, None, None)

        # Combine all shots
        all_shots = parent_shots + child_shots

        # Create lookup
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        # Sort shots by scene order
        shot_order = {shot["shot_id"]: i for i, shot in enumerate(shot_breakdown.get("shots", []))}
        all_shots = sorted(all_shots, key=lambda s: shot_order.get(s["shot_id"], 999))

        total_shots = len(all_shots)
        logger.info(f"Processing {total_shots} shots for video generation")

        if progress_callback:
            await progress_callback(f"Found {total_shots} shots", 0.1, None, None)

        video_outputs = []

        for idx, shot_data in enumerate(all_shots):
            # Check for cancellation before processing each shot
            if self.queue_manager and self.queue_manager.is_cancel_requested():
                logger.info(f"{self.agent_name}: Cancellation detected, stopping after {len(video_outputs)} shots")
                break

            shot_id = shot_data["shot_id"]

            if progress_callback:
                await progress_callback(
                    f"Generating video: {shot_id}",
                    0.1 + (0.85 * (idx / max(total_shots, 1))),
                    idx + 1,
                    total_shots
                )

            try:
                logger.info(f"Generating video for shot: {shot_id}")

                shot_details = shots_by_id.get(shot_id, {})
                image_path = self.session_dir / shot_data["image_path"]

                if not image_path.exists():
                    logger.warning(f"Image not found for {shot_id}: {image_path}")
                    continue

                # Generate production brief
                production_brief = await self._generate_production_brief(
                    shot_id,
                    shot_details,
                    scene_breakdown,
                    image_path,
                    shot_number=idx + 1
                )

                # Generate video
                video_result = await self._generate_video_clip(
                    shot_id,
                    image_path,
                    production_brief,
                    shot_details
                )

                if video_result:
                    video_outputs.append({
                        "shot_id": shot_id,
                        "scene_id": shot_details.get("scene_id"),
                        "video_path": video_result["video_path"],
                        "duration_seconds": video_result.get("duration", self.video_duration),
                        "production_brief": production_brief,
                        "generation_timestamp": datetime.now().isoformat()
                    })

                    logger.info(f"✓ Generated video for: {shot_id}")
                else:
                    logger.warning(f"Video generation returned empty for {shot_id}")

            except Exception as e:
                logger.error(f"Failed to generate video for {shot_id}: {str(e)}")
                continue  # Soft failure

        # Count successful vs failed
        successful = [v for v in video_outputs if v.get("video_path") and not v.get("video_path", "").endswith(".placeholder.json")]
        failed = [v for v in video_outputs if v.get("video_path", "").endswith(".placeholder.json")]

        output = {
            "videos": video_outputs,
            "total_videos": len(video_outputs),
            "successful_videos": len(successful),
            "failed_videos": len(failed),
            "metadata": {
                "session_id": self.session_id,
                "generated_at": datetime.now().isoformat(),
                "video_provider": self.video_provider,
                "video_model": "fal-ai/veo3.1/fast/image-to-video" if self.video_provider == "fal" else (
                    getattr(self, 'veo_model', 'veo-3.1-fast') if self.veo_client else "none"
                ),
                "video_duration": self.video_duration
            }
        }

        if progress_callback:
            await progress_callback(
                f"Video generation complete: {len(video_outputs)} clips",
                1.0, None, None
            )

        logger.info(f"{self.agent_name}: Generated {len(video_outputs)} video clips")
        return output

    async def _generate_production_brief(
        self,
        shot_id: str,
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        image_path: Path,
        shot_number: int = 1
    ) -> Dict[str, Any]:
        """Generate a comprehensive production brief for video generation using Gemini."""
        scene_id = shot_details.get("scene_id", "")
        characters = shot_details.get("characters", [])
        first_frame = shot_details.get("first_frame", "")
        animation = shot_details.get("animation", "")
        camera_movement = shot_details.get("camera_movement", "static")
        location_name = shot_details.get("location", "")
        dialogue = shot_details.get("dialogue", "")
        shot_description = shot_details.get("shot_description", "")

        # Get scene context and character descriptions
        location_details = ""
        character_descriptions = []
        for scene in scene_breakdown.get("scenes", []):
            if scene.get("scene_id") == scene_id:
                location = scene.get("location", {})
                if isinstance(location, dict):
                    location_details = location.get("description", location_name)
                else:
                    location_details = location_name

                for char in scene.get("characters", []):
                    if char.get("name") in characters:
                        character_descriptions.append({
                            "name": char.get("name"),
                            "description": char.get("description", ""),
                            "position": "in frame"
                        })
                break

        # Build character details string for prompt
        char_details_str = "\n".join([
            f"- {c['name']}: {c['description']}"
            for c in character_descriptions
        ]) if character_descriptions else "No specific characters defined."

        # Prepare screenplay text (dialogue if present)
        screenplay = dialogue if dialogue else shot_description

        # Try to generate production brief using Gemini
        try:
            prompt_template = self._get_prompt_template("agent_10_prompt.txt")
            if prompt_template:
                formatted_prompt = prompt_template.replace("{{screenplay}}", screenplay or "No dialogue.")
                formatted_prompt = formatted_prompt.replace("{{shot_number}}", str(shot_number))
                formatted_prompt = formatted_prompt.replace("{{shot_description}}", shot_description)
                formatted_prompt = formatted_prompt.replace("{{character_details}}", char_details_str)
                formatted_prompt = formatted_prompt.replace("{{clip_duration}}", str(self.video_duration))
                formatted_prompt = formatted_prompt.replace("{{start_frame}}", first_frame)

                # Load image for Gemini
                pil_image = await asyncio.to_thread(Image.open, image_path)

                # Generate with Gemini (image + prompt, like Streamlit version)
                from google.genai import types
                response = await asyncio.to_thread(
                    self.client.client.models.generate_content,
                    model=self.client.model_name,
                    contents=[pil_image, formatted_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=8192,
                        response_mime_type="application/json"
                    )
                )

                response_text = response.text
                if response_text:
                    # Clean up response if needed
                    response_text = response_text.strip()
                    if response_text.startswith("```json"):
                        response_text = response_text.split('\n', 1)[1] if '\n' in response_text else response_text[7:]
                    elif response_text.startswith("```"):
                        response_text = response_text[3:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()

                    brief_data = json.loads(response_text)
                    if "video_production_brief" in brief_data:
                        production_brief = brief_data["video_production_brief"]
                        production_brief["shot_id"] = shot_id
                        production_brief["scene_id"] = scene_id
                        logger.info(f"Generated production brief via Gemini for {shot_id}")
                        return production_brief

        except Exception as e:
            logger.warning(f"Gemini production brief generation failed for {shot_id}: {e}")

        # Fallback: Build production brief programmatically
        motion_guidance = self._build_motion_guidance(
            camera_movement, first_frame, animation, shot_description
        )

        # Build temporal action plan
        temporal_action_plan = self._build_temporal_action_plan(
            dialogue, animation, shot_description, self.video_duration
        )

        production_brief = {
            "shot_id": shot_id,
            "scene_id": scene_id,
            "title": f"Shot {shot_number} - {location_name[:20]}",
            "duration_seconds": self.video_duration,
            "objective": shot_description[:100] if shot_description else "Cinematic shot transition",
            "art_style": "Cinematic, photorealistic, natural lighting",
            "temporal_action_plan": temporal_action_plan,
            "cinematography": {
                "shot_type": self._infer_shot_type(first_frame),
                "camera_movement": camera_movement
            },
            "atmosphere_and_tone": "dramatic, focused, natural",
            "constraints": [
                "No new entities or off-screen reveals",
                "Single continuous shot",
                "No text overlays",
                "No music or SFX",
                "No singing"
            ],
            "video_generation_prompt": self._build_video_prompt(
                first_frame, animation, dialogue, character_descriptions, location_details, motion_guidance
            ),
            "frame_description": {
                "start": first_frame,
                "end": first_frame  # Animation describes the end state
            },
            "characters": character_descriptions,
            "location": {
                "name": location_name,
                "details": location_details
            },
            "camera": {
                "movement": camera_movement,
                "motion_guidance": motion_guidance
            },
            "aspect_ratio": self.aspect_ratio,
            "style": "cinematic, professional film quality, smooth motion"
        }

        return production_brief

    def _build_temporal_action_plan(
        self,
        dialogue: str,
        animation: str,
        shot_description: str,
        duration: int
    ) -> List[Dict[str, Any]]:
        """Build a temporal action plan for the video."""
        segments = []

        # Calculate segment boundaries
        if duration <= 4:
            boundaries = [(0.0, 2.0), (2.0, 4.0)]
        elif duration <= 6:
            boundaries = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
        else:
            boundaries = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]

        for i, (start, end) in enumerate(boundaries):
            segment = {
                "time_segment": f"{start}-{end}s",
                "visual_event": "",
                "audio_event": "",
                "constraints_segment": ["No new entities may appear"]
            }

            if i == 0:
                # First segment - establish shot
                segment["visual_event"] = f"Establishing frame. {shot_description[:100]}" if shot_description else "Camera holds on scene"
                if dialogue:
                    segment["audio_event"] = dialogue
                else:
                    segment["audio_event"] = "Silent. No dialogue."
            elif i == len(boundaries) - 1:
                # Last segment - animation/motion
                segment["visual_event"] = animation[:150] if animation else "Subtle motion, characters remain in position"
                segment["audio_event"] = "Continuation of previous audio or silence"
            else:
                # Middle segments
                segment["visual_event"] = "Continuation of action and motion"
                segment["audio_event"] = "Natural progression of dialogue or silence"

            segments.append(segment)

        return segments

    def _infer_shot_type(self, first_frame: str) -> str:
        """Infer shot type from first frame description."""
        first_frame_lower = first_frame.lower()

        if "close-up" in first_frame_lower or "closeup" in first_frame_lower:
            return "close-up"
        elif "extreme close" in first_frame_lower:
            return "extreme close-up"
        elif "wide shot" in first_frame_lower or "establishing" in first_frame_lower:
            return "wide"
        elif "over-the-shoulder" in first_frame_lower or "over the shoulder" in first_frame_lower:
            return "over-shoulder"
        elif "medium" in first_frame_lower or "mid-shot" in first_frame_lower:
            return "medium"
        elif "two-shot" in first_frame_lower or "two shot" in first_frame_lower:
            return "medium two-shot"
        else:
            return "medium"

    def _build_video_prompt(
        self,
        first_frame: str,
        animation: str,
        dialogue: str,
        characters: List[Dict],
        location: str,
        motion_guidance: str
    ) -> str:
        """Build a single-line video generation prompt."""
        parts = []

        # Start with the frame description
        if first_frame:
            parts.append(first_frame[:200])

        # Add character context
        if characters:
            char_names = [c["name"] for c in characters[:3]]
            parts.append(f"Characters: {', '.join(char_names)}")

        # Add location
        if location:
            parts.append(f"Location: {location[:50]}")

        # Add motion
        if animation:
            parts.append(f"Motion: {animation[:100]}")
        elif motion_guidance:
            parts.append(f"Camera: {motion_guidance[:80]}")

        # Add dialogue context (not the actual words, but context)
        if dialogue:
            parts.append("Character speaks dialogue")

        # Style suffix
        parts.append("Cinematic quality, smooth natural motion, photorealistic")

        return ". ".join(parts)

    def _build_motion_guidance(
        self,
        camera_movement: str,
        first_frame: str,
        animation: str,
        action: str
    ) -> str:
        """Build motion guidance text for video generation."""
        motion_parts = []

        # Camera movement guidance
        camera_movements = {
            "static": "Camera remains still, subject movement only",
            "pan_left": "Smooth horizontal camera pan from right to left",
            "pan_right": "Smooth horizontal camera pan from left to right",
            "tilt_up": "Camera tilts upward gradually",
            "tilt_down": "Camera tilts downward gradually",
            "zoom_in": "Slow zoom toward subject, increasing intimacy",
            "zoom_out": "Slow zoom out, revealing more of the scene",
            "dolly_in": "Camera physically moves toward subject",
            "dolly_out": "Camera physically moves away from subject",
            "tracking": "Camera follows subject movement",
            "crane_up": "Camera rises smoothly upward",
            "crane_down": "Camera descends smoothly",
            "handheld": "Subtle natural camera shake for documentary feel",
            "steadicam": "Smooth flowing movement following action"
        }

        movement_lower = camera_movement.lower() if camera_movement else "static"
        for key, guidance in camera_movements.items():
            if key in movement_lower:
                motion_parts.append(guidance)
                break

        if not motion_parts:
            motion_parts.append("Subtle cinematic camera movement")

        # Animation description
        if animation:
            motion_parts.append(f"Animation: {animation[:100]}")

        # Action guidance
        if action:
            motion_parts.append(f"Action: {action[:100]}")

        return ". ".join(motion_parts)

    async def _generate_video_clip(
        self,
        shot_id: str,
        image_path: Path,
        production_brief: Dict[str, Any],
        shot_details: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate video clip using FAL Veo 3.1 or Vertex AI Veo."""
        video_filename = f"{shot_id}.mp4"
        video_path = self.videos_dir / video_filename

        # Detect aspect ratio from image if set to "auto"
        if self.aspect_ratio == "auto":
            detected_aspect_ratio = await asyncio.to_thread(
                get_aspect_ratio_from_image, image_path
            )
            logger.info(f"Detected aspect ratio: {detected_aspect_ratio}")
        else:
            detected_aspect_ratio = self.aspect_ratio

        # Normalize duration to supported values (4s, 6s, 8s)
        requested_duration = production_brief.get("duration_seconds", self.video_duration)
        try:
            requested_duration = int(round(float(requested_duration)))
        except (TypeError, ValueError):
            requested_duration = 6

        if requested_duration <= 5:
            effective_duration = 4
        elif requested_duration <= 7:
            effective_duration = 6
        else:
            effective_duration = 8

        logger.info(f"Duration: {requested_duration}s normalized to {effective_duration}s")

        video_result_metadata = {}

        # Try FAL first if configured
        if self.video_provider == "fal" and self.fal_client:
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"FAL generation attempt {attempt + 1}/{self.max_retries}")

                    # Upload image to FAL
                    image_url = await asyncio.to_thread(
                        self.fal_client.upload_file, str(image_path)
                    )
                    logger.debug(f"Image uploaded to FAL: {image_url}")

                    # Call FAL API
                    result = await self._call_fal_api(
                        prompt=production_brief,
                        image_url=image_url,
                        duration=f"{effective_duration}s",
                        aspect_ratio=detected_aspect_ratio
                    )

                    # Extract video URL
                    video_url = result.get("video", {}).get("url")
                    if not video_url:
                        raise ValueError("No video URL in FAL API response")

                    # Download video to local directory
                    await self._download_video(video_url, video_path)

                    video_result_metadata = {
                        "video_url": video_url,
                        "video_path": str(video_path.relative_to(self.session_dir)),
                        "fal_request_id": result.get("request_id", "unknown"),
                        "video_provider": "fal",
                        "duration": effective_duration,
                        "aspect_ratio": detected_aspect_ratio
                    }

                    # Save per-shot metadata
                    await self._save_video_metadata(
                        shot_id, video_path, production_brief, video_result_metadata
                    )

                    logger.info(f"✓ FAL video generated for {shot_id}")
                    return {
                        "video_path": str(video_path.relative_to(self.session_dir)),
                        "duration": effective_duration,
                        "method": "fal"
                    }

                except Exception as e:
                    logger.warning(f"FAL attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"All FAL attempts failed for {shot_id}, trying Vertex AI...")

        # Try Vertex AI Veo
        if self.veo_client:
            # Prepare image for base64 encoding
            image_base64, _ = await asyncio.to_thread(
                optimize_image_for_base64,
                image_path,
                max_size_mb=self.max_image_size_mb
            )

            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Vertex AI Veo attempt {attempt + 1}/{self.max_retries}")

                    result = await self.veo_client.generate_video(
                        prompt=production_brief,
                        image_base64=image_base64,
                        aspect_ratio=detected_aspect_ratio,
                        duration_seconds=effective_duration
                    )

                    # Save video from base64
                    await self.veo_client.save_video_from_base64(
                        result["video_base64"],
                        video_path
                    )

                    video_result_metadata = {
                        "video_url": None,
                        "video_path": str(video_path.relative_to(self.session_dir)),
                        "vertex_operation": result.get("operation_name", "N/A"),
                        "video_provider": "vertex_ai",
                        "duration": effective_duration,
                        "aspect_ratio": detected_aspect_ratio
                    }

                    # Save per-shot metadata
                    await self._save_video_metadata(
                        shot_id, video_path, production_brief, video_result_metadata
                    )

                    logger.info(f"✓ Vertex AI Veo video generated for {shot_id}")
                    return {
                        "video_path": str(video_path.relative_to(self.session_dir)),
                        "duration": effective_duration,
                        "method": "veo"
                    }

                except Exception as e:
                    logger.warning(f"Veo attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(5)
                    else:
                        logger.error(f"All Veo attempts failed for {shot_id}")

        # Fallback: Create placeholder video info
        logger.warning(f"Using placeholder for {shot_id} (no video provider available)")

        placeholder_metadata = {
            "shot_id": shot_id,
            "status": "placeholder",
            "production_brief": production_brief,
            "source_image": str(image_path.relative_to(self.session_dir)),
            "needs_generation": True,
            "created_at": datetime.now().isoformat()
        }

        metadata_path = video_path.with_suffix('.placeholder.json')

        def _save_placeholder():
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(placeholder_metadata, f, indent=2)

        await asyncio.to_thread(_save_placeholder)

        return {
            "video_path": str(metadata_path.relative_to(self.session_dir)),
            "duration": effective_duration,
            "method": "placeholder"
        }

    async def _call_fal_api(
        self,
        prompt: Dict[str, Any],
        image_url: str,
        duration: str,
        aspect_ratio: str
    ) -> Dict[str, Any]:
        """Call FAL AI API to generate video asynchronously."""
        def _run_fal():
            def on_queue_update(update):
                if isinstance(update, self.fal_client.InProgress):
                    for log in update.logs:
                        logger.info(f"FAL progress: {log.get('message', '')}")
                elif isinstance(update, self.fal_client.Queued):
                    logger.info(f"FAL queued, position: {getattr(update, 'position', 'unknown')}")

            # Convert production brief to JSON string
            prompt_text = json.dumps(prompt, indent=2)
            logger.debug(f"Sending production brief to FAL ({len(prompt_text)} chars)")

            return self.fal_client.subscribe(
                "fal-ai/veo3.1/fast/image-to-video",
                arguments={
                    "prompt": prompt_text,
                    "image_url": image_url,
                    "aspect_ratio": aspect_ratio,
                    "duration": duration,
                    "generate_audio": getattr(self, 'generate_audio', True),
                    "resolution": getattr(self, 'video_resolution', '1080p')
                },
                with_logs=True,
                on_queue_update=on_queue_update,
            )

        return await asyncio.to_thread(_run_fal)

    async def _download_video(self, video_url: str, video_path: Path) -> None:
        """Download video file from URL to local path."""
        logger.info(f"Downloading video from FAL CDN...")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(video_url)
            response.raise_for_status()

            def _write_file():
                with open(video_path, 'wb') as f:
                    f.write(response.content)

            await asyncio.to_thread(_write_file)

        logger.info(f"Video downloaded to {video_path.name}")

    async def _save_video_metadata(
        self,
        shot_id: str,
        video_path: Path,
        production_brief: Dict[str, Any],
        video_result: Dict[str, Any]
    ) -> None:
        """Save per-shot video metadata JSON file."""
        metadata_path = video_path.with_suffix('.json')

        metadata = {
            "shot_id": shot_id,
            "video_path": video_result.get("video_path"),
            "video_url": video_result.get("video_url"),
            "video_provider": video_result.get("video_provider"),
            "duration_seconds": video_result.get("duration"),
            "aspect_ratio": video_result.get("aspect_ratio"),
            "production_brief": production_brief,
            "prompt_used": production_brief,
            "generated_at": datetime.now().isoformat()
        }

        # Add provider-specific fields
        if "fal_request_id" in video_result:
            metadata["fal_request_id"] = video_result["fal_request_id"]
        if "vertex_operation" in video_result:
            metadata["vertex_operation"] = video_result["vertex_operation"]

        def _save():
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        await asyncio.to_thread(_save)
        logger.debug(f"Saved video metadata: {metadata_path.name}")


# Alias for backwards compatibility
VideoDialogueAgent = VideoDialogueGeneratorAgent
