"""
Agent 11: Async Intelligent Video Editor
Transforms individual AI-generated video clips into a cinematic final product
using audio intelligence (WhisperX), narrative awareness (Gemini), and
sophisticated editing techniques (J/L cuts, dynamic pacing).
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path
from loguru import logger

from app.agents.base_agent import AsyncBaseAgent
from app.core.gemini_client import AsyncGeminiClient
from app.utils.audio_analyzer import AsyncAudioAnalyzer, is_whisperx_available
from app.utils.ffmpeg_builder import AsyncFFmpegBuilder


class VideoEditAgent(AsyncBaseAgent):
    """
    Async agent for intelligent video editing.

    Pipeline:
    1. Analyze audio from all video clips using WhisperX
    2. Generate Edit Decision List (EDL) using Gemini 2.5 Flash
    3. Execute edits per scene using FFmpeg
    4. Assemble master timeline
    5. Export final video(s)
    """

    def __init__(
        self,
        session_id: str,
        config: dict,
        gemini_client: AsyncGeminiClient
    ):
        super().__init__(
            agent_name="agent_11",
            session_id=session_id,
            config=config
        )
        self.client = gemini_client
        self.videos_dir = self._ensure_directory("assets/videos")
        self.edit_output_dir = self._ensure_directory("assets/edited")

        # Initialize utilities
        self.audio_analyzer = AsyncAudioAnalyzer(config)
        self.ffmpeg_builder = AsyncFFmpegBuilder(config)

        # Configuration
        self.max_edl_retries = config.get("max_edl_retries", 3)
        self.use_heuristic_fallback = config.get("use_heuristic_fallback", True)
        self.scene_fade_duration = config.get("scene_fade_duration", 1.0)
        self.skip_gemini_edl = config.get("skip_gemini_edl", False)

        # Model configuration from config (not hardcoded!)
        self.edl_model = config.get("model", "gemini-2.5-flash")  # LLM for EDL generation

        logger.info(f"{self.agent_name}: Initialization complete")

    async def validate_input(self, input_data: Any) -> None:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = ["videos", "scene_breakdown", "shot_breakdown"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Missing required input key: {key}")

        # Extract videos list from Agent 10's full output dict
        agent_10_output = input_data.get("videos", {})
        videos = agent_10_output.get("videos", []) if isinstance(agent_10_output, dict) else agent_10_output

        if not videos:
            raise ValueError("No videos provided for editing")

        # Check video file existence
        missing_videos = []
        for video_info in videos:
            video_path = self.session_dir / video_info.get("video_path", "")
            if not video_path.exists():
                # Check if it's a placeholder
                if ".placeholder.json" not in str(video_path):
                    missing_videos.append(video_info.get("shot_id", "unknown"))

        if missing_videos:
            logger.warning(f"Missing video files for shots: {', '.join(missing_videos)}")

    async def validate_output(self, output_data: Any) -> None:
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        required_keys = ["master_video_path", "scene_videos", "edit_timeline"]
        for key in required_keys:
            if key not in output_data:
                raise ValueError(f"Missing required output key: {key}")

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Main processing pipeline for Agent 11."""
        logger.info(f"{self.agent_name}: Starting video editing pipeline")

        # Extract videos list from Agent 10's full output dict
        agent_10_output = input_data.get("videos", {})
        videos = agent_10_output.get("videos", []) if isinstance(agent_10_output, dict) else agent_10_output

        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]

        # Filter out placeholder videos
        real_videos = [
            v for v in videos
            if ".placeholder.json" not in v.get("video_path", "")
        ]

        if not real_videos:
            logger.warning("No real video files found - all are placeholders")
            return self._create_placeholder_output(videos)

        logger.info(f"{self.agent_name}: Processing {len(real_videos)} video clips")

        # Phase 1: Audio Intelligence
        if progress_callback:
            await progress_callback("Phase 1: Analyzing audio...", 0.1, None, None)

        logger.info(f"{self.agent_name}: Phase 1 - Audio Analysis")
        audio_metadata = await self.audio_analyzer.analyze_video_batch(
            real_videos, self.session_dir
        )

        # Save audio metadata
        audio_metadata_path = self.edit_output_dir / "audio_metadata.json"
        await self.audio_analyzer.save_metadata(audio_metadata, audio_metadata_path)

        # Phase 2: Generate Edit Decision List
        if progress_callback:
            await progress_callback("Phase 2: Generating EDL...", 0.3, None, None)

        logger.info(f"{self.agent_name}: Phase 2 - Generate Edit Decision List")
        edit_timeline = await self._generate_edit_timeline(
            real_videos,
            audio_metadata,
            scene_breakdown,
            shot_breakdown
        )

        # Save EDL
        edl_path = self.edit_output_dir / "edit_decision_list.json"

        def _save_edl():
            with open(edl_path, 'w', encoding='utf-8') as f:
                json.dump(edit_timeline, f, indent=2, ensure_ascii=False)

        await asyncio.to_thread(_save_edl)
        logger.info(f"{self.agent_name}: EDL saved to {edl_path}")

        # Phase 3: Edit Scene Videos
        if progress_callback:
            await progress_callback("Phase 3: Editing scenes...", 0.5, None, None)

        logger.info(f"{self.agent_name}: Phase 3 - Edit Scene Videos")
        scene_videos = await self._edit_all_scenes(edit_timeline, real_videos, audio_metadata)

        # Phase 4: Assemble Master Timeline
        if progress_callback:
            await progress_callback("Phase 4: Assembling master...", 0.8, None, None)

        logger.info(f"{self.agent_name}: Phase 4 - Assemble Master Timeline")
        master_video = await self._assemble_master_timeline(scene_videos)

        # Phase 5: Compile Output
        total_duration = await self.ffmpeg_builder.get_video_duration(master_video)

        output = {
            "master_video_path": str(master_video.relative_to(self.session_dir)),
            "scene_videos": scene_videos,
            "edit_timeline": edit_timeline,
            "total_duration": round(total_duration, 2),
            "edit_metadata": {
                "scenes_edited": len(scene_videos),
                "total_shots": len(real_videos),
                "audio_analysis_completed": len(audio_metadata),
                "editing_method": edit_timeline.get("editing_method", "gemini_edl")
            }
        }

        if progress_callback:
            await progress_callback(
                f"Editing complete: {output['total_duration']}s",
                1.0, None, None
            )

        logger.info(
            f"{self.agent_name}: Editing complete - "
            f"{output['total_duration']}s ({len(scene_videos)} scenes)"
        )

        return output

    def _create_placeholder_output(self, videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create placeholder output when no real videos exist."""
        return {
            "master_video_path": "assets/edited/master_final.mp4",
            "scene_videos": [],
            "edit_timeline": {
                "edit_plan": {"scenes": [], "editing_notes": "No real videos to edit"},
                "editing_method": "placeholder"
            },
            "total_duration": 0.0,
            "edit_metadata": {
                "scenes_edited": 0,
                "total_shots": len(videos),
                "audio_analysis_completed": 0,
                "editing_method": "placeholder"
            }
        }

    async def _generate_edit_timeline(
        self,
        videos: List[Dict[str, Any]],
        audio_metadata: Dict[str, Dict[str, Any]],
        scene_breakdown: Dict[str, Any],
        shot_breakdown: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Edit Decision List using Gemini 2.5 Flash."""
        # Build context
        context = self._build_editing_context(
            videos,
            audio_metadata,
            scene_breakdown,
            shot_breakdown
        )

        # Skip Gemini if configured
        if self.skip_gemini_edl:
            logger.info(f"{self.agent_name}: Skipping Gemini - using heuristic editing")
            return await self._generate_heuristic_edl(
                videos, audio_metadata, scene_breakdown, shot_breakdown
            )

        # Try Gemini EDL generation with retries and self-correction
        last_failed_response = None
        last_error = None

        for attempt in range(self.max_edl_retries):
            try:
                logger.info(
                    f"{self.agent_name}: Gemini EDL generation attempt "
                    f"{attempt + 1}/{self.max_edl_retries}"
                )

                # Build prompt
                json_str = json.dumps(context, indent=2)
                base_prompt = self._get_edl_prompt().replace('{input}', json_str)

                # Add self-correction context if we have a failed attempt
                if last_failed_response and last_error:
                    prompt = f"""{base_prompt}

IMPORTANT: Your previous response had a JSON error. Please fix it.

Previous (malformed) response:
```
{last_failed_response}
```

Error: {last_error}

Please return ONLY valid JSON this time."""
                else:
                    prompt = base_prompt

                # Call Gemini - use regular text generation (not JSON mode)
                from google.genai import types

                response = await asyncio.to_thread(
                    self.client.client.models.generate_content,
                    model=self.edl_model,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=8192
                    )
                )

                if response.text:
                    edit_timeline = self._extract_json(response.text)
                    self._validate_edl(edit_timeline)

                    edit_timeline["editing_method"] = "gemini_edl"
                    edit_timeline["gemini_attempt"] = attempt + 1

                    logger.info(f"{self.agent_name}: Gemini EDL generation successful")
                    return edit_timeline

            except Exception as e:
                logger.warning(
                    f"{self.agent_name}: Gemini EDL attempt {attempt + 1} failed: {e}"
                )
                # Store failed response for self-correction on next attempt
                last_failed_response = response.text if response and response.text else None
                last_error = str(e)

                if attempt < self.max_edl_retries - 1:
                    await asyncio.sleep(1)

        # All attempts failed
        logger.warning(
            f"{self.agent_name}: Gemini EDL generation failed after "
            f"{self.max_edl_retries} attempts"
        )

        if self.use_heuristic_fallback:
            logger.info(f"{self.agent_name}: Using heuristic fallback editing")
            return await self._generate_heuristic_edl(
                videos, audio_metadata, scene_breakdown, shot_breakdown
            )
        else:
            raise RuntimeError(
                "Gemini EDL generation failed and heuristic fallback is disabled"
            )

    def _get_edl_prompt(self) -> str:
        """Get the EDL generation prompt template."""
        try:
            return self._get_prompt_template("agent_11_prompt.txt")
        except FileNotFoundError:
            # Fallback inline prompt
            return """You are an expert video editor. Analyze the following shot and audio metadata to create an Edit Decision List (EDL) for professional video editing.

INPUT:
{input}

Generate an EDL with the following structure for each scene:
- shot_id: The shot identifier
- video_path: Path to video file
- edit_type: "hard_start", "hard_cut", "j_cut", or "l_cut"
- trim_start: Seconds to trim from beginning
- trim_end: End timestamp (where to cut)
- audio_start_offset: Audio offset (negative for J-cut, positive for L-cut)
- transition: "cut" or "dissolve"
- rationale: Brief explanation of edit choice

Return valid JSON:
{
    "edit_plan": {
        "scenes": [
            {
                "scene_id": "string",
                "shots": [
                    {
                        "shot_id": "string",
                        "video_path": "string",
                        "edit_type": "string",
                        "trim_start": number,
                        "trim_end": number,
                        "audio_start_offset": number,
                        "transition": "string",
                        "rationale": "string"
                    }
                ]
            }
        ],
        "editing_notes": "string"
    }
}
"""

    def _build_editing_context(
        self,
        videos: List[Dict[str, Any]],
        audio_metadata: Dict[str, Dict[str, Any]],
        scene_breakdown: Dict[str, Any],
        shot_breakdown: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build context dictionary for Gemini EDL generation."""
        # Group shots by scene
        shots_by_scene = {}
        for shot in shot_breakdown.get("shots", []):
            scene_id = shot.get("scene_id")
            if scene_id not in shots_by_scene:
                shots_by_scene[scene_id] = []

            shot_id = shot.get("shot_id")
            video_info = next((v for v in videos if v.get("shot_id") == shot_id), None)
            audio_info = audio_metadata.get(shot_id, {})

            merged_shot = {
                **shot,
                "video_path": video_info.get("video_path") if video_info else None,
                "duration": video_info.get("duration_seconds") if video_info else None,
                "audio_metadata": audio_info.get("dialogue") if audio_info else None
            }

            shots_by_scene[scene_id].append(merged_shot)

        # Build scene list
        scenes = []
        for scene in scene_breakdown.get("scenes", []):
            scene_id = scene.get("scene_id")
            scenes.append({
                "scene_id": scene_id,
                "location": scene.get("location", ""),
                "description": scene.get("description", ""),
                "shots": shots_by_scene.get(scene_id, [])
            })

        return {"scenes": scenes}

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from Gemini response text (matches Streamlit approach)."""
        # Try markdown code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try raw JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")
        return json.loads(json_str)

    def _validate_edl(self, edit_timeline: Dict[str, Any]):
        """Validate Edit Decision List structure."""
        if "edit_plan" not in edit_timeline:
            raise ValueError("EDL missing 'edit_plan' key")

        edit_plan = edit_timeline["edit_plan"]

        if "scenes" not in edit_plan:
            raise ValueError("Edit plan missing 'scenes' key")

        scenes = edit_plan["scenes"]
        if not isinstance(scenes, list):
            raise ValueError("Scenes must be a list")

        for scene in scenes:
            if "shots" not in scene:
                raise ValueError(f"Scene {scene.get('scene_id')} missing 'shots'")

            for shot in scene["shots"]:
                required_fields = ["shot_id", "edit_type", "trim_start", "trim_end"]
                for field in required_fields:
                    if field not in shot:
                        raise ValueError(
                            f"Shot {shot.get('shot_id', 'unknown')} missing '{field}'"
                        )

    async def _edit_all_scenes(
        self,
        edit_timeline: Dict[str, Any],
        videos: List[Dict[str, Any]],
        audio_metadata: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Edit all scenes according to EDL."""
        scene_videos = []
        edit_plan = edit_timeline.get("edit_plan", edit_timeline)
        scenes = edit_plan.get("scenes", [])

        for i, scene_edit in enumerate(scenes, 1):
            scene_id = scene_edit.get("scene_id")
            logger.info(f"{self.agent_name}: Editing scene {i}/{len(scenes)}: {scene_id}")

            try:
                scene_video = await self._edit_scene(scene_edit, videos)
                if scene_video:
                    scene_videos.append(scene_video)
            except Exception as e:
                logger.error(f"{self.agent_name}: Failed to edit {scene_id}: {e}")
                # Continue with other scenes

        return scene_videos

    async def _edit_scene(
        self,
        scene_edit: Dict[str, Any],
        videos: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Edit a single scene according to EDL."""
        scene_id = scene_edit.get("scene_id")
        shots = scene_edit.get("shots", [])

        if not shots:
            logger.warning(f"Scene {scene_id} has no shots")
            return None

        # Filter out shots without video files
        valid_shots = []
        for shot in shots:
            shot_id = shot.get("shot_id")
            video_path = shot.get("video_path")

            if video_path:
                full_path = self.session_dir / video_path
                if full_path.exists():
                    valid_shots.append(shot)
                else:
                    logger.warning(f"Video file not found for {shot_id}")
            else:
                logger.warning(f"No video path for {shot_id}")

        if not valid_shots:
            logger.warning(f"No valid videos for scene {scene_id}")
            return None

        output_path = self.edit_output_dir / f"{scene_id}.mp4"

        await self.ffmpeg_builder.edit_scene_with_edl(
            valid_shots,
            self.session_dir,
            output_path
        )

        duration = await self.ffmpeg_builder.get_video_duration(output_path)

        return {
            "scene_id": scene_id,
            "video_path": str(output_path.relative_to(self.session_dir)),
            "shot_count": len(valid_shots),
            "duration": round(duration, 2)
        }

    async def _assemble_master_timeline(
        self,
        scene_videos: List[Dict[str, Any]]
    ) -> Path:
        """Assemble all scene videos into master timeline."""
        logger.info(f"{self.agent_name}: Assembling master timeline from {len(scene_videos)} scenes")

        if not scene_videos:
            # Create empty placeholder
            master_path = self.edit_output_dir / "master_final.mp4"
            logger.warning("No scene videos to assemble")
            return master_path

        scene_paths = [
            self.session_dir / scene["video_path"]
            for scene in scene_videos
        ]

        master_path = self.edit_output_dir / "master_final.mp4"

        await self.ffmpeg_builder.concatenate_simple(
            scene_paths,
            master_path,
            add_fade_transitions=True,
            fade_duration=self.scene_fade_duration
        )

        logger.info(f"{self.agent_name}: Master timeline assembled: {master_path}")
        return master_path

    async def _generate_heuristic_edl(
        self,
        videos: List[Dict[str, Any]],
        audio_metadata: Dict[str, Dict[str, Any]],
        scene_breakdown: Dict[str, Any],
        shot_breakdown: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate intelligent heuristic-based EDL with J/L cuts when Gemini fails."""
        logger.info(f"{self.agent_name}: Generating heuristic EDL with J/L cut support")

        # Group shots by scene
        shots_by_scene = {}
        for shot in shot_breakdown.get("shots", []):
            scene_id = shot.get("scene_id")
            if scene_id not in shots_by_scene:
                shots_by_scene[scene_id] = []
            shots_by_scene[scene_id].append(shot)

        scenes = []
        for scene in scene_breakdown.get("scenes", []):
            scene_id = scene.get("scene_id")
            scene_shots = shots_by_scene.get(scene_id, [])

            shots_edl = []
            prev_had_dialogue = False
            prev_dialogue_end = 0.0

            for i, shot in enumerate(scene_shots):
                shot_id = shot.get("shot_id")
                video_info = next((v for v in videos if v.get("shot_id") == shot_id), None)
                audio_info = audio_metadata.get(shot_id, {})

                if not video_info:
                    continue

                duration = video_info.get("duration_seconds", 5.0)
                dialogue = audio_info.get("dialogue")
                current_has_dialogue = dialogue and dialogue.get("has_speech")

                # Determine trim values
                if current_has_dialogue:
                    trim_start = max(0.0, dialogue["speech_start"] - 0.3)
                    trim_end = min(duration, dialogue["speech_end"] + 0.3)

                    # CRITICAL: Protect dialogue - never cut off speech
                    if trim_end < dialogue["speech_end"]:
                        trim_end = min(duration, dialogue["speech_end"] + 0.2)
                else:
                    # Reaction shot or action shot
                    trim_start = 0.3  # Small trim at start
                    trim_end = min(duration - 0.3, 4.0)  # Cap reaction shots at 4s

                # Enforce max/min shot duration limits
                max_duration = self.config.get("max_shot_duration", 12.0)
                min_duration = self.config.get("min_shot_duration", 1.5)

                # Max duration enforcement
                if trim_end - trim_start > max_duration:
                    trim_end = trim_start + max_duration

                # Min duration enforcement
                if trim_end - trim_start < min_duration:
                    trim_end = min(duration, trim_start + min_duration)
                    if trim_end - trim_start < min_duration:
                        trim_start = 0.0
                        trim_end = min(duration, min_duration)

                # Ensure we don't exceed source duration
                if trim_end > duration:
                    trim_end = duration

                # Determine edit type and audio offset
                if i == 0:
                    # First shot of scene - always hard_start
                    edit_type = "hard_start"
                    audio_offset = 0.0
                    rationale = "First shot of scene"
                elif current_has_dialogue and prev_had_dialogue:
                    # Dialogue exchange - use J-cut for natural conversation flow
                    edit_type = "j_cut"
                    audio_offset = -0.5  # Audio starts 0.5s before video cut
                    rationale = "J-cut for dialogue exchange"
                elif current_has_dialogue and not prev_had_dialogue:
                    # Dialogue following action/reaction - hard cut
                    edit_type = "hard_cut"
                    audio_offset = 0.0
                    rationale = "Dialogue following reaction"
                elif not current_has_dialogue and prev_had_dialogue:
                    # Reaction shot after dialogue - use L-cut
                    edit_type = "l_cut"
                    audio_offset = 0.6  # Previous audio continues 0.6s into this shot
                    rationale = "L-cut reaction to dialogue"
                else:
                    # Action to action - hard cut
                    edit_type = "hard_cut"
                    audio_offset = 0.0
                    rationale = "Action continuation"

                shots_edl.append({
                    "shot_id": shot_id,
                    "video_path": video_info.get("video_path"),
                    "edit_type": edit_type,
                    "trim_start": round(trim_start, 2),
                    "trim_end": round(trim_end, 2),
                    "audio_start_offset": audio_offset,
                    "transition": "cut",
                    "rationale": rationale
                })

                # Track dialogue state for next iteration
                prev_had_dialogue = current_has_dialogue
                if current_has_dialogue:
                    prev_dialogue_end = dialogue.get("speech_end", 0.0)

            if shots_edl:
                scenes.append({
                    "scene_id": scene_id,
                    "shots": shots_edl
                })

        return {
            "edit_plan": {
                "scenes": scenes,
                "scene_count": len(scenes),
                "total_estimated_duration": None,  # Calculated during editing
                "editing_notes": "Heuristic fallback with intelligent J/L cut detection"
            },
            "editing_method": "heuristic_fallback"
        }
