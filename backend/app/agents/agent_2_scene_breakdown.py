"""
Agent 2: Async Scene Breakdown
Breaks down screenplay into scenes with verbose location/character descriptions and subscenes.
"""

import asyncio
import json
from typing import Any, Dict, Optional, Callable
from loguru import logger
from pydantic import ValidationError

from app.agents.base_agent import AsyncBaseAgent
from app.core.gemini_client import AsyncGeminiClient
from app.schemas.validators import SceneBreakdown


class SceneBreakdownAgent(AsyncBaseAgent):
    """Async agent for breaking down screenplay into detailed scenes."""

    def __init__(
        self,
        session_id: str,
        config: dict,
        gemini_client: AsyncGeminiClient
    ):
        """Initialize Scene Breakdown Agent."""
        super().__init__(
            agent_name="agent_2",
            session_id=session_id,
            config=config
        )
        self.client = gemini_client
        self.temperature = config.get("temperature", 0.7)
        self.max_output_tokens = config.get("max_output_tokens", 8192)
        self.prompt_file = config.get("prompt_file", "agent_2_prompt.txt")
        self.model_override = config.get("model")  # Optional model override

    async def validate_input(self, input_data: Any) -> None:
        """Validate input screenplay."""
        if not isinstance(input_data, str):
            raise ValueError("Input must be a string (screenplay text)")
        if len(input_data.strip()) < 100:
            raise ValueError(
                "Screenplay is too short. Please provide a complete screenplay."
            )

        screenplay = input_data.upper()
        has_scene_headings = any(
            marker in screenplay
            for marker in ["INT.", "EXT.", "INT/EXT"]
        )
        if not has_scene_headings:
            raise ValueError(
                "Input must be a properly formatted screenplay with scene headings"
            )

    async def validate_output(self, output_data: Any) -> None:
        """Validate output scene breakdown."""
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        try:
            scene_breakdown = SceneBreakdown(**output_data)

            if scene_breakdown.total_scenes == 0:
                raise ValueError("No scenes found in breakdown")

            for scene in scene_breakdown.scenes:
                if len(scene.location.description) < 50:
                    raise ValueError(
                        f"Location description for {scene.scene_id} is not verbose enough "
                        f"(minimum 50 characters, got {len(scene.location.description)})"
                    )

                for char in scene.characters:
                    if len(char.description) < 50:
                        raise ValueError(
                            f"Character description for {char.name} in {scene.scene_id} "
                            f"is not verbose enough (minimum 50 characters, "
                            f"got {len(char.description)})"
                        )

            logger.debug(f"{self.agent_name}: Output validation passed")

        except ValidationError as e:
            raise ValueError(f"Scene breakdown validation failed: {str(e)}")

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Break down screenplay into scenes."""
        logger.info(f"{self.agent_name}: Breaking down screenplay into scenes...")

        if progress_callback:
            await progress_callback("Starting scene breakdown...", 0.0, None, None)

        # Load and format prompt
        prompt_template = self._get_prompt_template(self.prompt_file)
        prompt = prompt_template.replace("{INPUT}", input_data)

        if progress_callback:
            await progress_callback("Analyzing screenplay structure...", 0.3, None, None)

        # Generate scene breakdown using Gemini
        response = await self.client.generate_json(
            prompt=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            model_name=self.model_override
        )

        if progress_callback:
            await progress_callback("Parsing scene breakdown...", 0.8, None, None)

        try:
            scene_breakdown = self._extract_json(response)
            scene_breakdown["total_scenes"] = len(scene_breakdown.get("scenes", []))

            logger.info(
                f"{self.agent_name}: Generated breakdown with "
                f"{scene_breakdown.get('total_scenes', 0)} scenes"
            )

            if progress_callback:
                await progress_callback(
                    f"Scene breakdown complete: {scene_breakdown['total_scenes']} scenes",
                    1.0, None, None
                )

            # Save scene breakdown to file
            output_dir = self._ensure_directory("outputs")
            breakdown_path = output_dir / "scene_breakdown.json"
            def _save():
                with open(breakdown_path, 'w', encoding='utf-8') as f:
                    json.dump(scene_breakdown, f, indent=2, ensure_ascii=False)
            await asyncio.to_thread(_save)
            logger.info(f"{self.agent_name}: Scene breakdown saved to {breakdown_path}")

            return scene_breakdown

        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to parse JSON response: {str(e)}")
            raise ValueError(f"Failed to parse scene breakdown JSON: {str(e)}")

    async def process_with_feedback(
        self,
        input_data: Any,
        error_feedback: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Generate scene breakdown with error feedback."""
        logger.info(f"{self.agent_name}: Regenerating with feedback...")

        if progress_callback:
            await progress_callback("Regenerating scene breakdown...", 0.0, None, None)

        prompt_template = self._get_prompt_template(self.prompt_file)
        prompt = prompt_template.replace("{INPUT}", input_data)

        response = await self.client.generate_with_feedback(
            prompt=prompt,
            error_feedback=error_feedback,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            model_name=self.model_override
        )

        try:
            scene_breakdown = self._extract_json(response)
            scene_breakdown["total_scenes"] = len(scene_breakdown.get("scenes", []))

            if progress_callback:
                await progress_callback("Scene breakdown regenerated", 1.0, None, None)

            return scene_breakdown
        except Exception as e:
            raise ValueError(f"Failed to parse scene breakdown JSON: {str(e)}")

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON object from text response."""
        # Try to parse entire response as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        start_idx = text.find('{')
        end_idx = text.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Try to find JSON array
        start_idx = text.find('[')
        end_idx = text.rfind(']')

        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            try:
                data = json.loads(json_str)
                return {"scenes": data, "total_scenes": len(data), "metadata": {}}
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract valid JSON from response")
