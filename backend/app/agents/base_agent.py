"""
Async base agent with progress tracking.

All agents inherit from AsyncBaseAgent and implement:
- process(): Main processing logic
- validate_input(): Input validation
- validate_output(): Output validation
"""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional

from loguru import logger


class AsyncBaseAgent(ABC):
    """
    Async base class for all pipeline agents.

    Features:
    - Async execution with progress callbacks
    - Input/output validation
    - Retry logic with exponential backoff
    - Session directory management
    """

    def __init__(
        self,
        agent_name: str,
        session_id: str,
        config: dict,
        session_dir: Optional[Path] = None
    ):
        self.agent_name = agent_name
        self.session_id = session_id
        self.config = config
        self.session_dir = session_dir or Path(f"outputs/projects/{session_id}")

    async def execute(
        self,
        input_data: Any,
        progress_callback: Optional[Callable[[str, float, Optional[int], Optional[int]], Awaitable[None]]] = None
    ) -> Any:
        """
        Execute the agent with progress tracking.

        Args:
            input_data: Input data for this agent
            progress_callback: Async callback for progress updates
                Signature: async callback(message, progress, current, total)

        Returns:
            Agent output data
        """
        # Validate input
        await self.validate_input(input_data)

        # Process with progress updates
        output = await self.process(input_data, progress_callback)

        # Validate output
        await self.validate_output(output)

        return output

    @abstractmethod
    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Any:
        """
        Main processing logic - must be implemented by subclasses.

        Args:
            input_data: Input data for this agent
            progress_callback: Optional callback for progress updates

        Returns:
            Processed output data
        """
        pass

    async def validate_input(self, input_data: Any) -> None:
        """
        Validate input data.

        Override in subclasses for specific validation.
        Raises ValueError if validation fails.
        """
        if input_data is None:
            raise ValueError("Input data cannot be None")

    async def validate_output(self, output_data: Any) -> None:
        """
        Validate output data.

        Override in subclasses for specific validation.
        Raises ValueError if validation fails.
        """
        if output_data is None:
            raise ValueError("Output data cannot be None")

    def _get_prompt_template(self, prompt_file: str) -> str:
        """Load prompt template from file."""
        # Handle both formats:
        # - "prompts/agent_1_prompt.txt" (from config.yaml)
        # - "agent_1_prompt.txt" (from agent defaults)
        if prompt_file.startswith("prompts/") or prompt_file.startswith("prompts\\"):
            prompt_path = Path(prompt_file)
            fallback_paths = [
                prompt_path,
                Path("..") / prompt_file,
                Path("../..") / prompt_file,
            ]
        else:
            prompt_path = Path("prompts") / prompt_file
            fallback_paths = [
                prompt_path,
                Path("../prompts") / prompt_file,
                Path("../../prompts") / prompt_file,
            ]

        for path in fallback_paths:
            if path.exists():
                return path.read_text(encoding='utf-8')

        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    def _ensure_directory(self, subdir: str) -> Path:
        """Ensure a subdirectory exists in the session directory."""
        dir_path = self.session_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    async def execute_with_retry(
        self,
        input_data: Any,
        max_retries: int = 3,
        progress_callback: Optional[Callable[[str, float, Optional[int], Optional[int]], Awaitable[None]]] = None,
        error_feedback: Optional[str] = None
    ) -> Any:
        """
        Execute agent with automatic retry on failure.

        Args:
            input_data: Input data for processing
            max_retries: Maximum number of retry attempts
            progress_callback: Async callback for progress updates
            error_feedback: Optional error feedback from previous attempt

        Returns:
            Validated output data

        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"{self.agent_name}: Attempt {attempt + 1}/{max_retries}")

                # Use error feedback if provided and not first attempt
                if error_feedback and attempt > 0:
                    logger.info(f"{self.agent_name}: Using error feedback for retry")

                output_data = await self.execute(input_data, progress_callback)
                return output_data

            except Exception as e:
                last_error = e
                logger.warning(f"{self.agent_name}: Attempt {attempt + 1} failed: {str(e)}")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"{self.agent_name}: Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    # Prepare error feedback for next attempt
                    error_feedback = str(e)

        # All attempts failed
        logger.error(f"{self.agent_name}: All {max_retries} attempts failed")
        raise Exception(
            f"{self.agent_name} failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _load_prompt_gracefully(self, prompt_path: Path) -> Optional[str]:
        """Load prompt file with fallback paths and graceful failure."""
        # Try multiple locations
        possible_paths = [
            self.session_dir / prompt_path,
            Path(__file__).parent.parent.parent / prompt_path,
            prompt_path,
            Path("prompts") / prompt_path.name,
        ]

        for path in possible_paths:
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"Failed to read {path}: {e}")

        logger.warning(f"Prompt file not found: {prompt_path}")
        return None


# Placeholder agent for testing until real agents are migrated
class PlaceholderAgent(AsyncBaseAgent):
    """
    Placeholder agent for testing the pipeline.

    Returns mock data that matches expected output formats.
    """

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Any:
        """Return placeholder output based on agent type."""
        logger.warning(f"Using placeholder agent for {self.agent_name}")

        if progress_callback:
            await progress_callback(f"Processing {self.agent_name}...", 0.5, None, None)

        # Simulate processing time
        await asyncio.sleep(0.5)

        # Return appropriate placeholder data
        if self.agent_name == "agent_1":
            return {"text": "Placeholder screenplay content..."}

        elif self.agent_name == "agent_2":
            return {
                "scenes": [{"scene_id": "SCENE_1", "location": "Test Location"}],
                "total_scenes": 1
            }

        elif self.agent_name == "agent_3":
            return {
                "shots": [{"shot_id": "SHOT_1_1", "scene_id": "SCENE_1"}],
                "total_shots": 1
            }

        elif self.agent_name == "agent_4":
            return {
                "parent_shots": [{"shot_id": "SHOT_1_1", "child_shots": []}],
                "total_parent_shots": 1
            }

        elif self.agent_name == "agent_5":
            return {
                "characters": [],
                "character_grids": []
            }

        elif self.agent_name in ["agent_6", "agent_7"]:
            return {"parent_shots": []}

        elif self.agent_name in ["agent_8", "agent_9"]:
            return {"child_shots": []}

        elif self.agent_name == "agent_10":
            return {"videos": []}

        elif self.agent_name == "agent_11":
            return {
                "master_video_path": None,
                "scene_videos": [],
                "edit_timeline": {}
            }

        return {}


async def create_placeholder_agent(
    agent_name: str,
    session_id: str,
    config: Any
) -> AsyncBaseAgent:
    """Create a placeholder agent for testing."""
    return PlaceholderAgent(
        agent_name=agent_name,
        session_id=session_id,
        config={},
        session_dir=Path(f"outputs/projects/{session_id}")
    )
