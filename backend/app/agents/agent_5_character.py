"""
Agent 5: Async Character Creator
Generates consistent character images and combination grids for shot reference.
"""

import asyncio
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple, Optional, Callable
from datetime import datetime
from pathlib import Path
from PIL import Image
from loguru import logger

from app.agents.base_agent import AsyncBaseAgent
from app.core.gemini_client import AsyncGeminiClient
from app.utils.image_utils import (
    extract_character_combinations,
    create_character_grid,
    save_image_with_metadata,
    slugify,
)
from app.utils.fal_helper import (
    generate_with_fal_text_to_image,
    is_fal_available,
)


class CharacterCreatorAgent(AsyncBaseAgent):
    """Async agent for generating character images and combination grids."""

    def __init__(
        self,
        session_id: str,
        config: dict,
        gemini_client: AsyncGeminiClient
    ):
        super().__init__(
            agent_name="agent_5",
            session_id=session_id,
            config=config
        )
        self.client = gemini_client
        self.characters_dir = self._ensure_directory("assets/characters")
        self.grids_dir = self._ensure_directory("assets/grids")

        # Image provider configuration
        self.image_provider = config.get("image_provider", "gemini").lower()
        self.fal_model = config.get("fal_text_to_image_model", "fal-ai/nano-banana-pro")
        self.resolution = config.get("resolution", "1K")
        self.image_model = config.get("image_model", "gemini-2.5-flash-image")  # Gemini image model

    async def validate_input(self, input_data: Any) -> None:
        """Validate input data (Agent 2 scene breakdown + Agent 3 shot breakdown)."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        if "scene_breakdown" not in input_data:
            raise ValueError("Input must contain 'scene_breakdown' from Agent 2")

        if "shot_breakdown" not in input_data:
            raise ValueError("Input must contain 'shot_breakdown' from Agent 3")

    async def validate_output(self, output_data: Any) -> None:
        """Validate output character creation data."""
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        if "characters" not in output_data:
            raise ValueError("Output must contain 'characters'")

        if output_data.get("total_characters", 0) == 0:
            raise ValueError("No characters generated")

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Generate character images and combination grids."""
        logger.info(f"{self.agent_name}: Generating character images and grids...")

        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]

        if progress_callback:
            await progress_callback("Extracting characters from screenplay...", 0.05, None, None)

        # Step 1: Extract unique characters
        characters = self._extract_characters(scene_breakdown, shot_breakdown)
        total_characters = len(characters)
        logger.info(f"Found {total_characters} unique characters")

        if progress_callback:
            await progress_callback(f"Found {total_characters} characters", 0.1, None, None)

        # Step 2: Generate character images
        character_data = []
        character_images = {}

        for idx, (char_name, char_desc) in enumerate(characters.items()):
            try:
                if progress_callback:
                    await progress_callback(
                        f"Generating character: {char_name}",
                        0.1 + (0.5 * (idx / max(total_characters, 1))),
                        idx + 1,
                        total_characters
                    )

                logger.info(f"Generating image for character: {char_name}")

                # Generate character image
                char_image = await self._generate_character_image(char_name, char_desc)

                # Save character image
                char_filename = f"char_{slugify(char_name)}.png"
                char_path = self.characters_dir / char_filename

                await save_image_with_metadata(
                    char_image,
                    char_path,
                    metadata={
                        "character_name": char_name,
                        "description": char_desc,
                        "generated_at": datetime.now().isoformat()
                    }
                )

                character_data.append({
                    "name": char_name,
                    "description": char_desc,
                    "image_path": f"assets/characters/{char_filename}",
                    "generation_timestamp": datetime.now().isoformat()
                })

                character_images[char_name] = char_image
                logger.info(f"✓ Generated character: {char_name}")

            except Exception as e:
                logger.error(f"Failed to generate character {char_name}: {str(e)}")
                raise

        # Step 3: Analyze shots to find needed character combinations
        if progress_callback:
            await progress_callback("Analyzing shot combinations...", 0.65, None, None)

        combinations = extract_character_combinations(shot_breakdown)
        total_grids = len(combinations)
        logger.info(f"Found {total_grids} unique character combinations")

        # Step 4: Generate character grids
        grid_data = []

        for idx, combo in enumerate(combinations):
            try:
                combo_str = "_".join(sorted(combo))

                if progress_callback:
                    await progress_callback(
                        f"Creating grid: {combo_str}",
                        0.65 + (0.3 * (idx / max(total_grids, 1))),
                        idx + 1,
                        total_grids
                    )

                logger.info(f"Creating grid for: {combo_str}")

                # Get character images for this combination
                combo_images = [character_images[name] for name in combo if name in character_images]

                if not combo_images:
                    logger.warning(f"No images found for combination: {combo}")
                    continue

                # Create grid (runs in thread pool due to PIL operations)
                grid_image = await asyncio.to_thread(
                    create_character_grid,
                    combo_images,
                    list(combo),
                    (1920, 1080)
                )

                # Save grid
                grid_filename = f"grid_{combo_str}.png"
                grid_path = self.grids_dir / grid_filename

                await save_image_with_metadata(
                    grid_image,
                    grid_path,
                    metadata={
                        "characters": list(combo),
                        "generated_at": datetime.now().isoformat()
                    }
                )

                grid_data.append({
                    "grid_id": combo_str,
                    "characters": list(combo),
                    "grid_path": f"assets/grids/{grid_filename}",
                    "generation_timestamp": datetime.now().isoformat()
                })

                logger.info(f"✓ Created grid: {combo_str}")

            except Exception as e:
                logger.error(f"Grid creation failed for {combo}: {str(e)}")
                continue

        output = {
            "characters": character_data,
            "character_grids": grid_data,
            "total_characters": len(character_data),
            "total_grids": len(grid_data),
            "metadata": {
                "session_id": self.session_id,
                "generated_at": datetime.now().isoformat(),
                "image_provider": self.image_provider
            }
        }

        if progress_callback:
            await progress_callback(
                f"Character creation complete: {len(character_data)} characters, {len(grid_data)} grids",
                1.0, None, None
            )

        logger.info(
            f"{self.agent_name}: Generated {len(character_data)} characters "
            f"and {len(grid_data)} grids"
        )

        return output

    def _extract_characters(
        self,
        scene_breakdown: Dict[str, Any],
        shot_breakdown: Dict[str, Any]
    ) -> Dict[str, str]:
        """Extract unique characters and their descriptions."""
        characters = {}

        for scene in scene_breakdown.get("scenes", []):
            # Extract from main scene characters
            for char in scene.get("characters", []):
                char_name = char.get("name")
                char_desc = char.get("description")

                if char_name and char_desc and char_name not in characters:
                    characters[char_name] = char_desc

            # Extract from subscenes (CHARACTER_ADDED events)
            for subscene in scene.get("subscenes", []):
                if subscene.get("event") == "CHARACTER_ADDED":
                    char_added = subscene.get("character_added", {})
                    char_name = char_added.get("name")
                    char_desc = char_added.get("description")

                    if char_name and char_desc and char_name not in characters:
                        characters[char_name] = char_desc
                        logger.info(f"Found character '{char_name}' in subscene")

        # Cross-reference with shot_breakdown
        shot_characters = set()
        for shot in shot_breakdown.get("shots", []):
            shot_characters.update(shot.get("characters", []))

        for char_name in shot_characters:
            if char_name not in characters:
                logger.warning(f"Character '{char_name}' found in shots but not in scene breakdown")
                characters[char_name] = (
                    f"A character named {char_name}. "
                    "(No detailed description available)"
                )

        return characters

    async def _generate_character_image(self, char_name: str, char_desc: str) -> Image.Image:
        """Generate character image using configured provider."""
        prompt = f"""
Generate a high-quality character portrait for:

Character Name: {char_name}

Physical Description: {char_desc}

Style Requirements:
- Professional character design
- Clean, neutral background (white or soft gradient)
- Studio lighting, well-lit face
- Cinematic quality
- Front-facing portrait
- Consistent character design suitable for video generation reference

Generate a clear, detailed portrait of this character.
"""

        # Determine dimensions based on resolution
        # Strictly enforce 1:1 aspect ratio for character portraits
        if self.resolution == "4K":
            width = height = 3840
        elif self.resolution == "2K":
            width = height = 2048
        else:
            width = height = 1024

        # If using Seedream model, check for legacy width/height config
        # but override to ensure 1:1 if they differ significantly
        if "seedream" in self.fal_model.lower():
            cfg_width = self.config.get("width")
            cfg_height = self.config.get("height")
            if cfg_width and cfg_height:
                # Use the smaller dimension for both to ensure square
                dim = min(int(cfg_width), int(cfg_height))
                width = dim
                height = dim
            logger.debug(f"Seedream model detected: using {width}x{height} (1:1)")

        # Try FAL first if configured
        if self.image_provider == "fal" and is_fal_available():
            try:
                logger.info(f"Using FAL for character generation: {char_name}")
                pil_image, seed = await generate_with_fal_text_to_image(
                    prompt=prompt,
                    model=self.fal_model,
                    width=width,
                    height=height,
                    num_images=1,
                )
                logger.info(f"Successfully generated with FAL (seed: {seed})")
                return pil_image
            except Exception as e:
                logger.warning(f"FAL generation failed: {e}, falling back to Gemini")

        # Use Gemini for image generation
        logger.info(f"Using Gemini for character generation: {char_name}")

        try:
            from google.genai import types

            response = await asyncio.to_thread(
                self.client.client.models.generate_content,
                model=self.image_model,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="1:1",
                    ),
                ),
            )

            for part in response.parts:
                if part.inline_data is not None:
                    gemini_image = part.as_image()
                    pil_image = Image.open(BytesIO(gemini_image.image_bytes))
                    return pil_image

        except Exception as e:
            logger.error(f"Gemini image generation failed: {e}")
            raise

        raise ValueError(f"No image generated for character: {char_name}")
