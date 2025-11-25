"""
Async Gemini API Client
Wrapper for Google Gemini API with async support, retry logic, and error handling.
"""

import os
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

from loguru import logger
from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials
import yaml


class AsyncGeminiClient:
    """Async wrapper for Google Gemini API with built-in retry logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 3,
        use_vertex_ai: bool = False,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials_file: Optional[str] = None
    ):
        """Initialize async Gemini client."""
        # Determine default model name from config if not provided
        if model_name is None:
            model_name = self._get_default_model_from_config()

        self.model_name = model_name or "gemini-3-pro-preview"
        self.max_retries = max_retries
        self.use_vertex_ai = use_vertex_ai

        # Initialize client based on authentication mode
        if use_vertex_ai:
            self._init_vertex_ai(vertex_project, vertex_location, vertex_credentials_file)
        else:
            self._init_api_key(api_key)

        logger.info(f"Initialized AsyncGeminiClient with model: {self.model_name}")

    def _get_default_model_from_config(self) -> Optional[str]:
        """Attempt to read the default model from config.yaml."""
        try:
            # Try multiple config locations
            for config_path in [Path("config.yaml"), Path("../config.yaml"), Path("../../config.yaml")]:
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        return config.get("gemini", {}).get("model")
        except Exception as e:
            logger.warning(f"Failed to read default model from config.yaml: {e}")
        return None

    def _init_api_key(self, api_key: Optional[str]) -> None:
        """Initialize client for direct API with API key."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=self.api_key)
        logger.info("Using direct Gemini API with API key")

    def _init_vertex_ai(
        self,
        project: Optional[str],
        location: Optional[str],
        credentials_file: Optional[str]
    ) -> None:
        """Initialize client for Vertex AI with service account."""
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            raise ValueError("GCP project ID required for Vertex AI.")

        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        creds_file = credentials_file or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google-credentials.json")
        creds_path = Path(creds_file)

        if not creds_path.exists():
            raise FileNotFoundError(f"Service account credentials file not found: {creds_file}")

        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        credentials = Credentials.from_service_account_file(str(creds_path), scopes=scopes)

        self.client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
            credentials=credentials
        )
        logger.info(f"Using Vertex AI (project: {self.project}, location: {self.location})")

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
        system_instruction: Optional[str] = None,
        retry_on_error: bool = True,
        model_name: Optional[str] = None,
        response_mime_type: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text using Gemini API asynchronously with retry logic.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_output_tokens: Maximum tokens in response
            system_instruction: Optional system instruction
            retry_on_error: Whether to retry on API errors
            model_name: Override model name for this call
            response_mime_type: Optional MIME type for response (e.g., "application/json")

        Returns:
            Generated text response
        """
        # Build generation config
        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type

        generation_config = types.GenerateContentConfig(**config_kwargs, **kwargs)

        last_error = None
        retries = self.max_retries if retry_on_error else 1
        model_to_use = model_name or self.model_name

        for attempt in range(retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{retries} for Gemini API call")

                # Run synchronous API call in thread pool
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_to_use,
                    contents=prompt,
                    config=generation_config
                )

                if not response.text:
                    raise ValueError("Empty response from Gemini API")

                logger.info(f"Successfully generated response (length: {len(response.text)})")
                return response.text

            except Exception as e:
                last_error = e
                error_msg = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")

                if attempt < retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {retries} attempts failed")

        raise Exception(f"Failed to generate content after {retries} attempts: {last_error}")

    async def generate_json(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
        system_instruction: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate JSON-formatted response."""
        return await self.generate(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
            model_name=model_name,
            response_mime_type="application/json",
            **kwargs
        )

    async def generate_with_feedback(
        self,
        prompt: str,
        error_feedback: str,
        **kwargs
    ) -> str:
        """Generate content with error feedback from previous attempt."""
        feedback_prompt = f"""{prompt}

IMPORTANT: The previous attempt failed with the following issue:
{error_feedback}

Please correct this issue and try again."""

        return await self.generate(feedback_prompt, **kwargs)


# Singleton instance
_client_instance: Optional[AsyncGeminiClient] = None


def get_gemini_client() -> AsyncGeminiClient:
    """Get or create the global Gemini client instance."""
    global _client_instance
    if _client_instance is None:
        # Try to load config
        use_vertex = os.getenv("USE_VERTEX_AI", "true").lower() == "true"
        _client_instance = AsyncGeminiClient(use_vertex_ai=use_vertex)
    return _client_instance
