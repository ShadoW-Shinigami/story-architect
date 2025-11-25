"""
Agent 3: Async Shot Breakdown
Converts scenes into individual shots with strict JSON format.
Each shot has: shot_description, first_frame (with ALL elements), and animation.
"""

import asyncio
import json
from typing import Any, Dict, Optional, Callable
from loguru import logger
from pydantic import ValidationError

from app.agents.base_agent import AsyncBaseAgent
from app.core.gemini_client import AsyncGeminiClient
from app.schemas.validators import ShotBreakdown


class ShotBreakdownAgent(AsyncBaseAgent):
    """Async agent for breaking down scenes into individual shots."""

    def __init__(
        self,
        session_id: str,
        config: dict,
        gemini_client: AsyncGeminiClient
    ):
        """Initialize Shot Breakdown Agent."""
        super().__init__(
            agent_name="agent_3",
            session_id=session_id,
            config=config
        )
        self.client = gemini_client
        self.temperature = config.get("temperature", 0.7)
        self.max_output_tokens = config.get("max_output_tokens", 8192)
        self.prompt_file = config.get("prompt_file", "agent_3_prompt.txt")
        self.model_override = config.get("model")  # Optional model override

    async def validate_input(self, input_data: Any) -> None:
        """Validate input scene breakdown."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary (scene breakdown)")
        if "scenes" not in input_data:
            raise ValueError("Input must contain 'scenes' key")
        if not isinstance(input_data["scenes"], list):
            raise ValueError("'scenes' must be a list")
        if len(input_data["scenes"]) == 0:
            raise ValueError("Scene breakdown must contain at least one scene")

    async def validate_output(self, output_data: Any) -> None:
        """Validate output shot breakdown."""
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        try:
            shot_breakdown = ShotBreakdown(**output_data)

            if shot_breakdown.total_shots == 0:
                raise ValueError("No shots found in breakdown")

            for shot in shot_breakdown.shots:
                if len(shot.shot_description) < 10:
                    raise ValueError(
                        f"Shot description for {shot.shot_id} is too short "
                        f"(minimum 10 characters)"
                    )

                if len(shot.first_frame) < 10:
                    raise ValueError(
                        f"First frame description for {shot.shot_id} must be more verbose "
                        f"(minimum 10 characters)"
                    )

                if len(shot.animation) < 5:
                    raise ValueError(
                        f"Animation description for {shot.shot_id} is too short "
                        f"(minimum 5 characters)"
                    )

                if not shot.shot_id or not shot.scene_id:
                    raise ValueError(
                        f"Shot must have both shot_id and scene_id"
                    )

            logger.debug(f"{self.agent_name}: Output validation passed")

        except ValidationError as e:
            raise ValueError(f"Shot breakdown validation failed: {str(e)}")

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Break down scenes into shots."""
        logger.info(f"{self.agent_name}: Breaking down scenes into shots...")

        if progress_callback:
            await progress_callback("Starting shot breakdown...", 0.0, None, None)

        # Convert scene breakdown to JSON string for prompt
        scene_breakdown_json = json.dumps(input_data, indent=2, ensure_ascii=False)

        # Load and format prompt
        prompt_template = self._get_prompt_template(self.prompt_file)
        prompt = prompt_template.replace("{INPUT}", scene_breakdown_json)

        if progress_callback:
            await progress_callback("Analyzing scenes for shot composition...", 0.3, None, None)

        # Generate shot breakdown using Gemini
        response = await self.client.generate_json(
            prompt=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            model_name=self.model_override
        )

        if progress_callback:
            await progress_callback("Parsing shot breakdown...", 0.8, None, None)

        try:
            shot_breakdown = self._extract_json(response)
            shot_breakdown["total_shots"] = len(shot_breakdown.get("shots", []))

            logger.info(
                f"{self.agent_name}: Generated breakdown with "
                f"{shot_breakdown.get('total_shots', 0)} shots"
            )

            if progress_callback:
                await progress_callback(
                    f"Shot breakdown complete: {shot_breakdown['total_shots']} shots",
                    1.0, None, None
                )

            # Save shot breakdown to file
            output_dir = self._ensure_directory("outputs")
            breakdown_path = output_dir / "shot_breakdown.json"
            def _save():
                with open(breakdown_path, 'w', encoding='utf-8') as f:
                    json.dump(shot_breakdown, f, indent=2, ensure_ascii=False)
            await asyncio.to_thread(_save)
            logger.info(f"{self.agent_name}: Shot breakdown saved to {breakdown_path}")

            return shot_breakdown

        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to parse JSON response: {str(e)}")
            raise ValueError(f"Failed to parse shot breakdown JSON: {str(e)}")

    async def process_with_feedback(
        self,
        input_data: Any,
        error_feedback: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Generate shot breakdown with error feedback."""
        logger.info(f"{self.agent_name}: Regenerating with feedback...")

        if progress_callback:
            await progress_callback("Regenerating shot breakdown...", 0.0, None, None)

        scene_breakdown_json = json.dumps(input_data, indent=2, ensure_ascii=False)
        prompt_template = self._get_prompt_template(self.prompt_file)
        prompt = prompt_template.replace("{INPUT}", scene_breakdown_json)

        response = await self.client.generate_with_feedback(
            prompt=prompt,
            error_feedback=error_feedback,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            model_name=self.model_override
        )

        try:
            shot_breakdown = self._extract_json(response)
            shot_breakdown["total_shots"] = len(shot_breakdown.get("shots", []))

            if progress_callback:
                await progress_callback("Shot breakdown regenerated", 1.0, None, None)

            return shot_breakdown
        except Exception as e:
            raise ValueError(f"Failed to parse shot breakdown JSON: {str(e)}")

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
                return {"shots": data, "total_shots": len(data), "metadata": {}}
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract valid JSON from response")
