"""
Agent 6: Async Parent Image Generator
Transforms character grids into full cinematic scenes using Gemini or FAL.
Uses character grids as PRIMARY INPUT to maintain consistency.
"""

import asyncio
import json
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path
from PIL import Image
from loguru import logger

from app.agents.base_agent import AsyncBaseAgent
from app.core.gemini_client import AsyncGeminiClient
from app.utils.image_utils import save_image_with_metadata
from app.utils.fal_helper import (
    generate_with_fal_text_to_image,
    generate_with_fal_edit,
    is_fal_available,
)


class ParentImageGeneratorAgent(AsyncBaseAgent):
    """Async agent for generating parent shot images by transforming character grids."""

    def __init__(
        self,
        session_id: str,
        config: dict,
        gemini_client: AsyncGeminiClient
    ):
        super().__init__(
            agent_name="agent_6",
            session_id=session_id,
            config=config
        )
        self.client = gemini_client
        self.parent_shots_dir = self._ensure_directory("assets/parent_shots")
        self.grids_dir = self._ensure_directory("assets/grids")

        # Image provider configuration
        self.image_provider = config.get("image_provider", "gemini").lower()
        self.use_optimizer = config.get("use_prompt_optimizer", True)
        self.fal_edit_model = config.get("fal_edit_model", "fal-ai/nano-banana-pro/edit")
        self.fal_text_model = config.get("fal_text_to_image_model", "fal-ai/nano-banana-pro")

        # Model configuration from config (not hardcoded!)
        self.optimizer_model = config.get("optimizer_model", "gemini-3-pro-preview")  # LLM for prompt optimization
        self.image_model = config.get("image_model", "gemini-2.5-flash-image")  # Gemini image generation

        # Load templates from file
        self.optimizer_template_path = Path(config.get(
            "optimizer_prompt_file",
            "prompts/agent_6_optimizer_prompt.txt"
        ))
        self.prompt_template_path = Path(config.get(
            "prompt_file",
            "prompts/agent_6_prompt.txt"
        ))
        self.optimizer_template = self._load_template(self.optimizer_template_path)
        self.prompt_template = self._load_template(self.prompt_template_path)

    def _load_template(self, template_path: Path) -> Optional[str]:
        """Load template from file with fallback paths."""
        # Try relative to session dir first
        session_path = self.session_dir / template_path
        if session_path.exists():
            return session_path.read_text(encoding="utf-8")

        # Try relative to backend root
        backend_root = Path(__file__).parent.parent.parent
        backend_path = backend_root / template_path
        if backend_path.exists():
            return backend_path.read_text(encoding="utf-8")

        # Try absolute path
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")

        logger.warning(f"Template not found: {template_path}")
        return None

    async def validate_input(self, input_data: Any) -> None:
        """Validate input data."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = ["scene_breakdown", "shot_breakdown", "shot_grouping", "character_grids"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Input must contain '{key}'")

    async def validate_output(self, output_data: Any) -> None:
        """Validate output parent shots data."""
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        if "parent_shots" not in output_data:
            raise ValueError("Output must contain 'parent_shots'")

        if output_data.get("total_parent_shots", 0) == 0:
            raise ValueError("No parent shots generated")

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Generate parent shot images."""
        logger.info(f"{self.agent_name}: Transforming character grids into parent shots...")

        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]
        shot_grouping = input_data["shot_grouping"]

        # Extract character_grids list from Agent 5's full output dict
        agent_5_output = input_data["character_grids"]
        character_grids = agent_5_output.get("character_grids", []) if isinstance(agent_5_output, dict) else agent_5_output

        if progress_callback:
            await progress_callback("Preparing shot data...", 0.05, None, None)

        # Create lookups
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        grids_by_chars = {
            tuple(sorted(grid["characters"])): grid["grid_path"]
            for grid in character_grids
        }

        # Extract parent shots
        parent_shot_ids = self._extract_parent_shots(shot_grouping)
        total_shots = len(parent_shot_ids)
        logger.info(f"Found {total_shots} parent shots to generate")

        if progress_callback:
            await progress_callback(f"Found {total_shots} parent shots", 0.1, None, None)

        # Generate parent images
        parent_shots_data = []

        for idx, shot_id in enumerate(parent_shot_ids):
            try:
                if progress_callback:
                    await progress_callback(
                        f"Generating parent shot: {shot_id}",
                        0.1 + (0.85 * (idx / max(total_shots, 1))),
                        idx + 1,
                        total_shots
                    )

                logger.info(f"Generating parent shot: {shot_id}")

                shot_details = shots_by_id.get(shot_id)
                if not shot_details:
                    logger.warning(f"Shot details not found for {shot_id}, skipping")
                    continue

                # Generate image
                image_path = await self._generate_parent_shot_image(
                    shot_id,
                    shot_details,
                    scene_breakdown,
                    grids_by_chars
                )

                parent_shots_data.append({
                    "shot_id": shot_id,
                    "scene_id": shot_details.get("scene_id"),
                    "image_path": str(image_path),
                    "generation_timestamp": datetime.now().isoformat(),
                    "verification_status": "pending",
                    "attempts": 1,
                    "final_verification": None,
                    "verification_history": []
                })

                logger.info(f"✓ Generated parent shot: {shot_id}")

            except Exception as e:
                logger.error(f"Failed to generate parent shot {shot_id}: {str(e)}")
                raise

        output = {
            "parent_shots": parent_shots_data,
            "total_parent_shots": len(parent_shots_data),
            "metadata": {
                "session_id": self.session_id,
                "generated_at": datetime.now().isoformat(),
                "image_provider": self.image_provider
            }
        }

        if progress_callback:
            await progress_callback(
                f"Parent shot generation complete: {len(parent_shots_data)} shots",
                1.0, None, None
            )

        logger.info(f"{self.agent_name}: Generated {len(parent_shots_data)} parent shots")
        return output

    def _extract_parent_shots(self, shot_grouping: Dict[str, Any]) -> List[str]:
        """Extract parent shot IDs from grouping."""
        parent_ids = []
        for parent_shot in shot_grouping.get("parent_shots", []):
            parent_ids.append(parent_shot["shot_id"])
        return parent_ids

    def _get_character_physical_descriptions(
        self,
        character_names: List[str],
        scene_breakdown: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract full physical descriptions for characters."""
        descriptions = []

        char_lookup = {}
        for scene in scene_breakdown.get("scenes", []):
            for char in scene.get("characters", []):
                char_name = char.get("name")
                char_desc = char.get("description", "")
                if char_name and char_desc:
                    char_lookup[char_name] = char_desc

        for name in character_names:
            if name in char_lookup:
                descriptions.append({
                    "name": name,
                    "physical_description": char_lookup[name]
                })
            else:
                logger.warning(f"Physical description not found for: {name}")
                descriptions.append({
                    "name": name,
                    "physical_description": f"Character named {name}"
                })

        return descriptions

    def _get_location_full_description(
        self,
        location_name: str,
        scene_id: str,
        scene_breakdown: Dict[str, Any]
    ) -> str:
        """Extract full location description."""
        for scene in scene_breakdown.get("scenes", []):
            if scene.get("scene_id") == scene_id:
                location = scene.get("location", {})
                return location.get("description", location_name)

        for scene in scene_breakdown.get("scenes", []):
            location = scene.get("location", {})
            if location.get("name") == location_name:
                return location.get("description", location_name)

        return location_name

    async def _optimize_prompt_with_pro(self, verbose_prompt: str) -> str:
        """Use Gemini Pro to optimize the verbose prompt for image generation."""
        if not self.use_optimizer:
            logger.debug("Prompt optimization disabled in config")
            return verbose_prompt

        # Use loaded template if available, otherwise fallback to inline
        if self.optimizer_template:
            try:
                optimization_prompt = self.optimizer_template.format(
                    verbose_scene_description=verbose_prompt
                )
            except KeyError as e:
                logger.warning(f"Template format error: {e}, using fallback")
                optimization_prompt = self._get_fallback_optimization_prompt(verbose_prompt)
        else:
            logger.warning("Optimizer template not loaded, using fallback")
            optimization_prompt = self._get_fallback_optimization_prompt(verbose_prompt)

        try:
            from google.genai import types

            logger.debug("Optimizing prompt with Gemini Pro...")
            response = await asyncio.to_thread(
                self.client.client.models.generate_content,
                model=self.optimizer_model,
                contents=[optimization_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=2048
                )
            )

            if response.text:
                optimized = response.text.strip()
                logger.info(f"Prompt optimized: {len(verbose_prompt)} → {len(optimized)} chars")
                return optimized

        except Exception as e:
            logger.warning(f"Prompt optimization failed: {e}")

        return verbose_prompt

    def _get_fallback_optimization_prompt(self, verbose_prompt: str) -> str:
        """Get fallback optimization prompt when template is not available."""
        return f"""You are an expert prompt optimizer for AI image generation. Your task is to take a verbose scene description and distill it into a concise, powerful prompt that will generate a stunning cinematic image.

INPUT SCENE DESCRIPTION:
{verbose_prompt}

OPTIMIZATION RULES:
1. Keep essential visual details (character appearances, location, lighting, composition)
2. Remove redundant or verbose language
3. Use specific, evocative visual language
4. Maintain the emotional tone and atmosphere
5. Keep character consistency details
6. Limit to 500 words maximum

OUTPUT: Provide only the optimized prompt, no explanations."""

    async def _generate_parent_shot_image(
        self,
        shot_id: str,
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        grids_by_chars: Dict[tuple, str]
    ) -> str:
        """Transform character grid into full parent shot image."""
        # Get shot components
        first_frame = shot_details.get("first_frame", "")
        characters = shot_details.get("characters", [])
        scene_id = shot_details.get("scene_id", "")
        location_name = shot_details.get("location", "")

        # Get verbose descriptions
        character_descriptions = self._get_character_physical_descriptions(
            characters, scene_breakdown
        )
        location_description = self._get_location_full_description(
            location_name, scene_id, scene_breakdown
        )

        # Find matching character grid
        grid_image = None
        grid_path = None

        if characters:
            char_combo = tuple(sorted(characters))
            grid_path = grids_by_chars.get(char_combo)

            if grid_path:
                full_grid_path = self.session_dir / grid_path
                if full_grid_path.exists():
                    grid_image = await asyncio.to_thread(Image.open, full_grid_path)
                    logger.info(f"Transforming character grid: {grid_path}")
                else:
                    logger.warning(f"Grid file not found: {full_grid_path}")
            else:
                logger.warning(f"No grid found for characters: {characters}")

        # Build character description text
        char_desc_text = ""
        for idx, char in enumerate(character_descriptions, 1):
            char_desc_text += f"""CHARACTER {idx}: {char['name']}
PHYSICAL APPEARANCE: {char['physical_description']}
"""

        # Build verbose prompt using template if available
        if self.prompt_template:
            verbose_prompt = self.prompt_template.format(
                shot_id=shot_id,
                first_frame=first_frame,
                location_description=location_description,
                character_descriptions=char_desc_text
            )
        else:
            # Fallback inline prompt (simplified)
            logger.warning(f"{self.agent_name}: Prompt template not loaded, using inline fallback")
            verbose_prompt = f"""
CINEMATIC SCENE GENERATION

SHOT ID: {shot_id}
FRAME DESCRIPTION: {first_frame}

LOCATION:
{location_description}

CHARACTERS:
{char_desc_text}

REQUIREMENTS:
- Generate a photorealistic cinematic frame
- 16:9 aspect ratio, film-quality composition
- Characters must match their physical descriptions exactly
- Professional lighting and color grading
- High production value, Hollywood quality

Transform the character reference images into a fully realized cinematic scene.
"""

        # Optimize prompt
        optimized_prompt = await self._optimize_prompt_with_pro(verbose_prompt)

        # Generate image
        generated_image = None
        fal_seed = None

        # Try FAL first if configured
        if self.image_provider == "fal" and is_fal_available():
            try:
                logger.info(f"Using FAL for parent shot: {shot_id}")

                if grid_image:
                    # Save grid temporarily for FAL upload
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        await asyncio.to_thread(grid_image.save, tmp.name)
                        tmp_grid_path = Path(tmp.name)

                    try:
                        generated_image, fal_seed = await generate_with_fal_edit(
                            prompt=optimized_prompt,
                            image_paths=[tmp_grid_path],
                            model=self.fal_edit_model,
                            width=1920,
                            height=1080,
                        )
                    finally:
                        tmp_grid_path.unlink(missing_ok=True)
                else:
                    generated_image, fal_seed = await generate_with_fal_text_to_image(
                        prompt=optimized_prompt,
                        model=self.fal_text_model,
                        width=1920,
                        height=1080,
                    )

                logger.info(f"Generated with FAL (seed: {fal_seed})")

            except Exception as e:
                logger.warning(f"FAL generation failed: {e}, falling back to Gemini")
                generated_image = None

        # Use Gemini if FAL not available or failed
        if generated_image is None:
            logger.info(f"Using Gemini for parent shot: {shot_id}")

            try:
                from google.genai import types

                if grid_image:
                    contents = [grid_image, optimized_prompt]
                else:
                    contents = [optimized_prompt]

                response = await asyncio.to_thread(
                    self.client.client.models.generate_content,
                    model=self.image_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio="16:9",
                        ),
                    ),
                )

                for part in response.parts:
                    if part.inline_data is not None:
                        gemini_image = part.as_image()
                        generated_image = Image.open(BytesIO(gemini_image.image_bytes))
                        break

            except Exception as e:
                logger.error(f"Gemini image generation failed: {e}")
                raise

        if not generated_image:
            raise ValueError(f"No image generated for {shot_id}")

        # Save image
        image_filename = f"{shot_id}_parent.png"
        image_path = self.parent_shots_dir / image_filename

        metadata = {
            "shot_id": shot_id,
            "characters": characters,
            "location": location_name,
            "generated_at": datetime.now().isoformat(),
            "grid_used": str(grid_path) if grid_path else "none",
            "image_provider": "fal" if fal_seed else "gemini",
            "optimized_prompt": optimized_prompt
        }

        if fal_seed:
            metadata["fal_seed"] = fal_seed

        await save_image_with_metadata(generated_image, image_path, metadata=metadata)

        return f"assets/parent_shots/{image_filename}"
