"""
Agent 1: Async Screenplay Generator
Converts input (logline/story/script) into a dialogue-driven screenplay with no narration.
"""

import asyncio
from typing import Any, Optional, Callable
from loguru import logger

from app.agents.base_agent import AsyncBaseAgent
from app.core.gemini_client import AsyncGeminiClient


class ScreenplayAgent(AsyncBaseAgent):
    """Async agent for generating dialogue-driven screenplays."""

    def __init__(
        self,
        session_id: str,
        config: dict,
        gemini_client: AsyncGeminiClient
    ):
        """Initialize Screenplay Agent."""
        super().__init__(
            agent_name="agent_1",
            session_id=session_id,
            config=config
        )
        self.client = gemini_client
        self.temperature = config.get("temperature", 0.7)
        self.max_output_tokens = config.get("max_output_tokens", 8192)
        self.prompt_file = config.get("prompt_file", "agent_1_prompt.txt")
        self.model_override = config.get("model")  # Optional model override

    async def validate_input(self, input_data: Any) -> None:
        """Validate input data."""
        if not isinstance(input_data, str):
            raise ValueError("Input must be a string")
        if len(input_data.strip()) < 10:
            raise ValueError("Input is too short. Please provide at least 10 characters.")

    async def validate_output(self, output_data: Any) -> None:
        """Validate output screenplay."""
        if not isinstance(output_data, str):
            raise ValueError("Output must be a string")
        if len(output_data.strip()) < 100:
            raise ValueError("Generated screenplay is too short")

        # Check for basic screenplay formatting
        screenplay = output_data.upper()
        has_scene_headings = any(
            marker in screenplay
            for marker in ["INT.", "EXT.", "INT/EXT"]
        )
        if not has_scene_headings:
            raise ValueError(
                "Screenplay must contain proper scene headings (INT./EXT.)"
            )

        # Check for dialogue
        lines = output_data.split('\n')
        has_dialogue = False

        for i, line in enumerate(lines):
            if line.strip() and line.strip().isupper() and len(line.strip()) > 2:
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].strip() and not lines[j].strip().isupper():
                        has_dialogue = True
                        break

        if not has_dialogue:
            raise ValueError(
                "Screenplay must be dialogue-driven with character names and dialogue"
            )

        logger.debug(f"{self.agent_name}: Output validation passed")

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Generate screenplay from input."""
        logger.info(f"{self.agent_name}: Generating screenplay...")

        if progress_callback:
            await progress_callback("Starting screenplay generation...", 0.0, None, None)

        # Load and format prompt
        prompt_template = self._get_prompt_template(self.prompt_file)
        prompt = prompt_template.replace("{INPUT}", input_data)

        if progress_callback:
            await progress_callback("Calling Gemini API...", 0.3, None, None)

        # Generate screenplay using Gemini
        screenplay = await self.client.generate(
            prompt=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            model_name=self.model_override
        )

        if progress_callback:
            await progress_callback("Screenplay generated successfully", 1.0, None, None)

        logger.info(
            f"{self.agent_name}: Generated screenplay "
            f"({len(screenplay)} characters, {len(screenplay.split())} words)"
        )

        # Save screenplay to file
        output_dir = self._ensure_directory("outputs")
        screenplay_path = output_dir / "screenplay.txt"
        await asyncio.to_thread(screenplay_path.write_text, screenplay, encoding='utf-8')
        logger.info(f"{self.agent_name}: Screenplay saved to {screenplay_path}")

        return screenplay

    async def process_with_feedback(
        self,
        input_data: Any,
        error_feedback: str,
        progress_callback: Optional[Callable] = None
    ) -> str:
        """Generate screenplay with error feedback."""
        logger.info(f"{self.agent_name}: Regenerating with feedback...")

        if progress_callback:
            await progress_callback("Regenerating with feedback...", 0.0, None, None)

        prompt_template = self._get_prompt_template(self.prompt_file)
        prompt = prompt_template.replace("{INPUT}", input_data)

        screenplay = await self.client.generate_with_feedback(
            prompt=prompt,
            error_feedback=error_feedback,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            model_name=self.model_override
        )

        if progress_callback:
            await progress_callback("Screenplay regenerated", 1.0, None, None)

        return screenplay
