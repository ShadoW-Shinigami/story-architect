"""
Configuration management for Story Architect backend.

Loads configuration from config.yaml and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class GeminiConfig(BaseModel):
    """Gemini API configuration."""
    model: str = "gemini-3-pro-preview"
    temperature: float = 0.7
    max_output_tokens: int = 32000
    timeout: int = 300
    use_vertex_ai: bool = True

    # Vertex AI settings
    vertex_project: Optional[str] = None
    vertex_location: str = "global"
    vertex_credentials_file: Optional[str] = None


class AgentConfig(BaseModel):
    """Per-agent configuration."""
    name: str
    enabled: bool = True
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    prompt_file: Optional[str] = None
    max_retries: int = 3

    # Image generation settings (for agents 5-9)
    image_provider: Optional[str] = None
    aspect_ratio: Optional[str] = None

    # Video settings (for agents 10-11)
    video_provider: Optional[str] = None


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    # API Keys
    gemini_api_key: Optional[str] = None
    fal_key: Optional[str] = None

    # Google Cloud (Vertex AI)
    google_cloud_project: Optional[str] = None
    google_cloud_location: str = "us-central1"
    google_application_credentials: Optional[str] = None

    # Application settings
    log_level: str = "INFO"
    max_retries: int = 3

    # Paths
    config_file: str = "config.yaml"
    outputs_dir: str = "outputs/projects"
    prompts_dir: str = "prompts"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class Config:
    """
    Main configuration class.

    Combines settings from:
    1. Environment variables (via Settings)
    2. config.yaml file
    """

    _instance: Optional["Config"] = None

    def __init__(self):
        self.settings = Settings()
        self._yaml_config: dict = {}
        self._load_yaml_config()

    @classmethod
    def get_instance(cls) -> "Config":
        """Get singleton config instance."""
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance

    def _load_yaml_config(self):
        """Load configuration from YAML file."""
        config_path = Path(self.settings.config_file)

        # Try multiple locations
        possible_paths = [
            config_path,
            Path("../config.yaml"),  # Parent directory (monorepo root)
            Path("../../config.yaml"),  # Two levels up
        ]

        for path in possible_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    self._yaml_config = yaml.safe_load(f) or {}
                return

        # No config file found - use defaults
        self._yaml_config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Example:
            config.get("gemini.temperature")
            config.get("agents.agent_1.max_output_tokens")
        """
        keys = key.split(".")
        value = self._yaml_config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for a specific agent."""
        agent_data = self.get(f"agents.{agent_name}", {})

        # Merge with defaults
        config_dict = {
            "name": agent_data.get("name", agent_name),
            "enabled": agent_data.get("enabled", True),
            "temperature": agent_data.get("temperature"),
            "max_output_tokens": agent_data.get("max_output_tokens"),
            "prompt_file": agent_data.get("prompt_file"),
            "max_retries": agent_data.get("max_retries", self.settings.max_retries),
            "image_provider": agent_data.get("image_provider"),
            "aspect_ratio": agent_data.get("aspect_ratio"),
            "video_provider": agent_data.get("video_provider"),
        }

        return AgentConfig(**config_dict)

    def get_gemini_config(self) -> GeminiConfig:
        """Get Gemini API configuration."""
        gemini_data = self.get("gemini", {})

        return GeminiConfig(
            model=gemini_data.get("model", "gemini-3-pro-preview"),
            temperature=gemini_data.get("temperature", 0.7),
            max_output_tokens=gemini_data.get("max_output_tokens", 32000),
            timeout=gemini_data.get("timeout", 300),
            use_vertex_ai=gemini_data.get("use_vertex_ai", True),
            vertex_project=self.settings.google_cloud_project or gemini_data.get("vertex_ai", {}).get("project_id"),
            vertex_location=self.settings.google_cloud_location or gemini_data.get("vertex_ai", {}).get("location", "global"),
            vertex_credentials_file=self.settings.google_application_credentials or gemini_data.get("vertex_ai", {}).get("credentials_file"),
        )

    @property
    def outputs_dir(self) -> Path:
        """Get outputs directory path."""
        return Path(self.settings.outputs_dir)

    @property
    def prompts_dir(self) -> Path:
        """Get prompts directory path."""
        return Path(self.settings.prompts_dir)


def get_config() -> Config:
    """Get the global configuration instance."""
    return Config.get_instance()
