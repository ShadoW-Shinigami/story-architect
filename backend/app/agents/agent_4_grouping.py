"""
Agent 4: Async Shot Grouping
Groups shots into parent-child relationships for efficient image generation.
Supports cross-scene grouping and multi-level hierarchies.
"""

import asyncio
import json
from typing import Any, Dict, Optional, Callable, List
from loguru import logger
from pydantic import ValidationError

from app.agents.base_agent import AsyncBaseAgent
from app.core.gemini_client import AsyncGeminiClient
from app.schemas.validators import ShotGrouping


class ShotGroupingAgent(AsyncBaseAgent):
    """Async agent for grouping shots into parent-child hierarchies."""

    def __init__(
        self,
        session_id: str,
        config: dict,
        gemini_client: AsyncGeminiClient
    ):
        """Initialize Shot Grouping Agent."""
        super().__init__(
            agent_name="agent_4",
            session_id=session_id,
            config=config
        )
        self.client = gemini_client
        self.temperature = config.get("temperature", 0.7)
        self.max_output_tokens = config.get("max_output_tokens", 8192)
        self.prompt_file = config.get("prompt_file", "agent_4_prompt.txt")
        self.model_override = config.get("model")  # Optional model override

    async def validate_input(self, input_data: Any) -> None:
        """Validate input shot breakdown."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary (shot breakdown)")
        if "shots" not in input_data:
            raise ValueError("Input must contain 'shots' key")
        if not isinstance(input_data["shots"], list):
            raise ValueError("'shots' must be a list")
        if len(input_data["shots"]) == 0:
            raise ValueError("Shot breakdown must contain at least one shot")

    async def validate_output(self, output_data: Any) -> None:
        """Validate output shot grouping."""
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        try:
            shot_grouping = ShotGrouping(**output_data)

            if shot_grouping.total_parent_shots == 0:
                raise ValueError("No parent shots found in grouping")

            self._validate_hierarchy(shot_grouping.parent_shots)

            logger.debug(f"{self.agent_name}: Output validation passed")

        except ValidationError as e:
            raise ValueError(f"Shot grouping validation failed: {str(e)}")

    def _validate_hierarchy(self, grouped_shots, parent_id=None):
        """Recursively validate shot hierarchy."""
        for shot in grouped_shots:
            if parent_id and shot.parent_shot_id != parent_id:
                raise ValueError(
                    f"Shot {shot.shot_id} has incorrect parent_shot_id: "
                    f"expected {parent_id}, got {shot.parent_shot_id}"
                )

            if not shot.grouping_reason:
                raise ValueError(
                    f"Shot {shot.shot_id} must have a grouping_reason"
                )

            if shot.child_shots:
                self._validate_hierarchy(shot.child_shots, shot.shot_id)

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Group shots into parent-child relationships."""
        logger.info(f"{self.agent_name}: Grouping shots into parent-child relationships...")

        if progress_callback:
            await progress_callback("Starting shot grouping...", 0.0, None, None)

        # Convert shot breakdown to JSON string for prompt
        shot_breakdown_json = json.dumps(input_data, indent=2, ensure_ascii=False)

        # Load and format prompt
        prompt_template = self._get_prompt_template(self.prompt_file)
        prompt = prompt_template.replace("{INPUT}", shot_breakdown_json)

        if progress_callback:
            await progress_callback("Analyzing shot relationships...", 0.3, None, None)

        # Generate shot grouping using Gemini
        response = await self.client.generate_json(
            prompt=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            model_name=self.model_override
        )

        if progress_callback:
            await progress_callback("Parsing shot grouping...", 0.8, None, None)

        try:
            shot_grouping = self._extract_json(response)
            shot_grouping["total_parent_shots"] = len(shot_grouping.get("parent_shots", []))
            shot_grouping["total_child_shots"] = self._count_children(shot_grouping.get("parent_shots", []))

            logger.info(
                f"{self.agent_name}: Generated grouping with "
                f"{shot_grouping.get('total_parent_shots', 0)} parent shots and "
                f"{shot_grouping.get('total_child_shots', 0)} child shots"
            )

            if progress_callback:
                await progress_callback(
                    f"Shot grouping complete: {shot_grouping['total_parent_shots']} parent, "
                    f"{shot_grouping['total_child_shots']} child shots",
                    1.0, None, None
                )

            # Save shot grouping to file
            output_dir = self._ensure_directory("outputs")
            grouping_path = output_dir / "shot_grouping.json"
            def _save():
                with open(grouping_path, 'w', encoding='utf-8') as f:
                    json.dump(shot_grouping, f, indent=2, ensure_ascii=False)
            await asyncio.to_thread(_save)
            logger.info(f"{self.agent_name}: Shot grouping saved to {grouping_path}")

            return shot_grouping

        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to parse JSON response: {str(e)}")
            raise ValueError(f"Failed to parse shot grouping JSON: {str(e)}")

    async def process_with_feedback(
        self,
        input_data: Any,
        error_feedback: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Generate shot grouping with error feedback."""
        logger.info(f"{self.agent_name}: Regenerating with feedback...")

        if progress_callback:
            await progress_callback("Regenerating shot grouping...", 0.0, None, None)

        shot_breakdown_json = json.dumps(input_data, indent=2, ensure_ascii=False)
        prompt_template = self._get_prompt_template(self.prompt_file)
        prompt = prompt_template.replace("{INPUT}", shot_breakdown_json)

        response = await self.client.generate_with_feedback(
            prompt=prompt,
            error_feedback=error_feedback,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            model_name=self.model_override
        )

        try:
            shot_grouping = self._extract_json(response)
            shot_grouping["total_parent_shots"] = len(shot_grouping.get("parent_shots", []))
            shot_grouping["total_child_shots"] = self._count_children(shot_grouping.get("parent_shots", []))

            if progress_callback:
                await progress_callback("Shot grouping regenerated", 1.0, None, None)

            return shot_grouping
        except Exception as e:
            raise ValueError(f"Failed to parse shot grouping JSON: {str(e)}")

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

        # Try to find JSON array (wrap it)
        start_idx = text.find('[')
        end_idx = text.rfind(']')

        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            try:
                data = json.loads(json_str)
                return {
                    "parent_shots": data,
                    "total_parent_shots": len(data),
                    "total_child_shots": self._count_children(data),
                    "grouping_strategy": "Location and character-based grouping",
                    "metadata": {}
                }
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not extract valid JSON from response")

    def _count_children(self, parent_shots: List[Dict]) -> int:
        """Recursively count all child shots."""
        count = 0
        for shot in parent_shots:
            if "child_shots" in shot and shot["child_shots"]:
                count += len(shot["child_shots"])
                count += self._count_children(shot["child_shots"])
        return count
