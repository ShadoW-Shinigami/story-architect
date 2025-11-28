"""
Agent Factory
Creates and configures agents based on configuration.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import yaml
from loguru import logger

from app.agents.base_agent import AsyncBaseAgent, PlaceholderAgent
from app.agents.agent_1_screenplay import ScreenplayAgent
from app.agents.agent_2_scene_breakdown import SceneBreakdownAgent
from app.agents.agent_3_shot_breakdown import ShotBreakdownAgent
from app.agents.agent_4_grouping import ShotGroupingAgent
from app.agents.agent_5_character import CharacterCreatorAgent
from app.agents.agent_6_parent_generator import ParentImageGeneratorAgent
from app.agents.agent_7_parent_verification import ParentVerificationAgent
from app.agents.agent_8_child_generator import ChildImageGeneratorAgent
from app.agents.agent_9_child_verification import ChildVerificationAgent
from app.agents.agent_10_video_dialogue import VideoDialogueAgent
from app.agents.agent_11_video_edit import VideoEditAgent
from app.core.gemini_client import AsyncGeminiClient, get_gemini_client


# Agent registry mapping agent names to their classes
AGENT_REGISTRY = {
    "agent_1": ScreenplayAgent,
    "agent_2": SceneBreakdownAgent,
    "agent_3": ShotBreakdownAgent,
    "agent_4": ShotGroupingAgent,
    "agent_5": CharacterCreatorAgent,
    "agent_6": ParentImageGeneratorAgent,
    "agent_7": ParentVerificationAgent,
    "agent_8": ChildImageGeneratorAgent,
    "agent_9": ChildVerificationAgent,
    "agent_10": VideoDialogueAgent,
    "agent_11": VideoEditAgent,
}


class AgentFactory:
    """Factory for creating and configuring agents."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        gemini_client: Optional[AsyncGeminiClient] = None
    ):
        """
        Initialize agent factory.

        Args:
            config_path: Path to config.yaml file
            gemini_client: Pre-configured Gemini client (uses singleton if not provided)
        """
        self.config = self._load_config(config_path)
        self.gemini_client = gemini_client or get_gemini_client()

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from config.yaml."""
        paths_to_try = [
            Path(config_path) if config_path else None,
            Path("config.yaml"),
            Path("../config.yaml"),
            Path("../../config.yaml"),
        ]

        for path in paths_to_try:
            if path and path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)

        logger.warning("No config.yaml found, using default configuration")
        return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "agents": {
                "agent_1": {
                    "temperature": 0.7,
                    "max_output_tokens": 8192,
                    "prompt_file": "agent_1_prompt.txt"
                },
                "agent_2": {
                    "temperature": 0.7,
                    "max_output_tokens": 8192,
                    "prompt_file": "agent_2_prompt.txt"
                },
                "agent_3": {
                    "temperature": 0.7,
                    "max_output_tokens": 8192,
                    "prompt_file": "agent_3_prompt.txt"
                },
                "agent_4": {
                    "temperature": 0.7,
                    "max_output_tokens": 8192,
                    "prompt_file": "agent_4_prompt.txt"
                },
            }
        }

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        agents_config = self.config.get("agents", {})
        return agents_config.get(agent_name, {})

    def create_agent(
        self,
        agent_name: str,
        session_id: str,
        config_override: Optional[Dict[str, Any]] = None,
        queue_manager: Optional['QueueManager'] = None
    ) -> AsyncBaseAgent:
        """
        Create an agent instance.

        Args:
            agent_name: Name of the agent (e.g., "agent_1")
            session_id: Session ID for this execution
            config_override: Optional config overrides
            queue_manager: Optional queue manager for cancellation checks

        Returns:
            Configured agent instance
        """
        # Get base config and merge with overrides
        agent_config = self.get_agent_config(agent_name)
        if config_override:
            agent_config = {**agent_config, **config_override}

        # Check if agent is implemented
        if agent_name in AGENT_REGISTRY:
            agent_class = AGENT_REGISTRY[agent_name]
            logger.info(f"Creating {agent_name} ({agent_class.__name__})")

            agent = agent_class(
                session_id=session_id,
                config=agent_config,
                gemini_client=self.gemini_client
            )
            # Store queue_manager reference if provided
            if queue_manager:
                agent.queue_manager = queue_manager
            return agent
        else:
            # Use placeholder for unimplemented agents
            logger.warning(f"Agent {agent_name} not implemented, using placeholder")
            return PlaceholderAgent(
                agent_name=agent_name,
                session_id=session_id,
                config=agent_config
            )

    def is_agent_implemented(self, agent_name: str) -> bool:
        """Check if an agent has a real implementation."""
        return agent_name in AGENT_REGISTRY


# Singleton factory instance
_factory_instance: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Get or create the global agent factory instance."""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = AgentFactory()
    return _factory_instance


async def create_agent(
    agent_name: str,
    session_id: str,
    config: Optional[Dict[str, Any]] = None,
    queue_manager: Optional['QueueManager'] = None
) -> AsyncBaseAgent:
    """
    Convenience function to create an agent.

    Args:
        agent_name: Name of the agent
        session_id: Session ID
        config: Optional config overrides
        queue_manager: Optional queue manager for cancellation checks

    Returns:
        Configured agent instance
    """
    factory = get_agent_factory()
    return factory.create_agent(agent_name, session_id, config, queue_manager=queue_manager)
