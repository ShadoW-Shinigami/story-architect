"""
Agent 8: Child Image Generator
Generates child shot images by editing parent shots using Gemini 2.5 Flash Image.
Uses physical trait descriptions and character grids for consistency.
"""

import json
from io import BytesIO
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image
from loguru import logger
from pydantic import ValidationError

from agents.base_agent import BaseAgent
from core.validators import ChildShotsOutput
from core.image_utils import save_image_with_metadata


class ChildImageGeneratorAgent(BaseAgent):
    """Agent for generating child shot images by editing parent shots."""

    def __init__(self, gemini_client, config, session_dir: Path):
        """Initialize Child Image Generator Agent."""
        super().__init__(gemini_client, config, "agent_8")
        self.session_dir = Path(session_dir)
        self.assets_dir = self.session_dir / "assets"
        self.child_shots_dir = self.assets_dir / "child_shots"
        self.parent_shots_dir = self.assets_dir / "parent_shots"
        self.grids_dir = self.assets_dir / "grids"

        # Create directory
        self.child_shots_dir.mkdir(parents=True, exist_ok=True)

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = ["scene_breakdown", "shot_breakdown", "shot_grouping", "parent_shots", "character_grids"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Input must contain '{key}'")

        return True

    def validate_output(self, output_data: Any) -> bool:
        """Validate output."""
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        try:
            child_output = ChildShotsOutput(**output_data)

            # Auto-fix count
            output_data["total_child_shots"] = len(child_output.child_shots)

            logger.debug(f"{self.agent_name}: Output validation passed")
            return True

        except ValidationError as e:
            raise ValueError(f"Child shots validation failed: {str(e)}")

    def process(self, input_data: Any) -> Dict[str, Any]:
        """Generate child shot images."""
        logger.info(f"{self.agent_name}: Editing parent shots to create child shots...")

        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]
        shot_grouping = input_data["shot_grouping"]
        parent_shots = input_data["parent_shots"]
        character_grids = input_data["character_grids"]

        # Create lookups
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        # Dynamic image lookup - starts with top-level parents, grows as children are generated
        # This allows grandchildren to use their immediate parent (which is a child shot)
        available_images_by_id = {
            parent["shot_id"]: self.session_dir / parent["image_path"]
            for parent in parent_shots
        }

        grids_by_chars = {
            tuple(sorted(grid["characters"])): self.session_dir / grid["grid_path"]
            for grid in character_grids
        }

        # Extract all child shots
        child_shot_list = self._extract_child_shots(shot_grouping)
        logger.info(f"Found {len(child_shot_list)} child shots to generate")

        # Generate child images
        child_shots_data = []

        for child_shot_info in child_shot_list:
            try:
                shot_id = child_shot_info["shot_id"]
                parent_id = child_shot_info["parent_shot_id"]

                logger.info(f"Generating child shot: {shot_id} (parent: {parent_id})")

                # Get shot details
                shot_details = shots_by_id.get(shot_id)
                if not shot_details:
                    logger.warning(f"Shot details not found for {shot_id}, skipping")
                    continue

                # Get parent image from dynamic lookup (includes previously generated children)
                parent_image_path = available_images_by_id.get(parent_id)
                if not parent_image_path or not parent_image_path.exists():
                    logger.warning(f"Parent image not found for {parent_id}, skipping")
                    continue

                # Generate child image
                child_image_path = self._generate_child_shot_image(
                    shot_id,
                    parent_id,
                    shot_details,
                    scene_breakdown,
                    parent_image_path,
                    grids_by_chars
                )

                # Add newly generated child to lookup for future grandchildren
                available_images_by_id[shot_id] = child_image_path

                # Store data
                # Try to get relative path, fallback to absolute if fails
                try:
                    rel_image_path = str(child_image_path.relative_to(self.session_dir))
                except ValueError:
                    # Fallback to absolute path (cross-drive on Windows)
                    rel_image_path = str(child_image_path)

                child_shots_data.append({
                    "shot_id": shot_id,
                    "scene_id": shot_details.get("scene_id"),
                    "image_path": rel_image_path,
                    "generation_timestamp": datetime.now().isoformat(),
                    "verification_status": "pending",
                    "attempts": 1,
                    "final_verification": None,
                    "verification_history": []
                })

                logger.info(f"✓ Generated child shot: {shot_id}")

            except Exception as e:
                logger.error(f"Failed to generate child shot {shot_id}: {str(e)}")
                # Continue with other shots (soft failure)
                continue

        # Prepare output
        output = {
            "child_shots": child_shots_data,
            "total_child_shots": len(child_shots_data),
            "metadata": {
                "session_id": self.session_dir.name,
                "generated_at": datetime.now().isoformat()
            }
        }

        logger.info(f"{self.agent_name}: Generated {len(child_shots_data)} child shots")

        return output

    def _extract_child_shots(self, shot_grouping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recursively extract all child shots from grouping."""
        child_shots = []

        def recurse(grouped_shot, parent_id):
            for child in grouped_shot.get("child_shots", []):
                child_shots.append({
                    "shot_id": child["shot_id"],
                    "parent_shot_id": parent_id
                })
                # Recursively process nested children
                recurse(child, child["shot_id"])

        # Process all parent shots
        for parent in shot_grouping.get("parent_shots", []):
            recurse(parent, parent["shot_id"])

        return child_shots

    def _get_character_physical_descriptions(
        self,
        character_names: List[str],
        scene_breakdown: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract full physical descriptions for characters."""
        descriptions = []

        # Build character lookup from all scenes
        char_lookup = {}
        for scene in scene_breakdown.get("scenes", []):
            for char in scene.get("characters", []):
                char_name = char.get("name")
                char_desc = char.get("description", "")
                if char_name and char_desc:
                    char_lookup[char_name] = char_desc

        # Get descriptions for requested characters
        for name in character_names:
            if name in char_lookup:
                descriptions.append({
                    "name": name,
                    "physical_description": char_lookup[name]
                })
            else:
                logger.warning(f"Physical description not found for character: {name}")
                descriptions.append({
                    "name": name,
                    "physical_description": f"Character named {name}"
                })

        return descriptions

    def _generate_child_shot_image(
        self,
        shot_id: str,
        parent_id: str,
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        parent_image_path: Path,
        grids_by_chars: Dict[tuple, Path]
    ) -> Path:
        """Generate child shot by editing parent shot."""
        from google.genai import types

        # Load parent image (PRIMARY INPUT #1)
        parent_image = Image.open(parent_image_path)
        logger.debug(f"Using parent image as input: {parent_image_path.name}")

        # Get shot components
        first_frame = shot_details.get("first_frame", "")
        characters = shot_details.get("characters", [])
        location_name = shot_details.get("location", "")
        scene_id = shot_details.get("scene_id", "")
        dialogue = shot_details.get("dialogue", "")

        # Get verbose character descriptions (PHYSICAL TRAITS!)
        character_descriptions = self._get_character_physical_descriptions(
            characters,
            scene_breakdown
        )

        # Find matching character grid (INPUT #2 if applicable)
        char_combo = tuple(sorted(characters))
        grid_path = grids_by_chars.get(char_combo)

        # Format character descriptions into template string
        char_desc_text = ""
        for idx, char in enumerate(character_descriptions, 1):
            char_desc_text += f"""
CHARACTER {idx}:
PHYSICAL TRAITS: {char['physical_description']}

When editing the parent shot, identify this character by these exact physical traits. If adding this character to the scene, use the character grid reference (if provided) to ensure their appearance matches precisely.

"""

        # Use template from prompt file
        if not self.prompt_template:
            raise ValueError("Prompt template not loaded")

        edit_prompt = self.prompt_template.format(
            shot_id=shot_id,
            parent_id=parent_id,
            first_frame=first_frame,
            character_descriptions=char_desc_text,
            location=location_name
        )

        # Prepare contents: [parent image, character grid (if available), edit prompt]
        contents = [parent_image]

        if grid_path and grid_path.exists():
            grid_image = Image.open(grid_path)
            contents.append(grid_image)
            logger.debug(f"Using character grid for editing: {grid_path.name}")

        contents.append(edit_prompt)

        # Generate edited image
        response = self.client.client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio="16:9",
                ),
                temperature=self.temperature,
            ),
        )

        # Extract generated image and convert to PIL Image
        generated_image = None
        for part in response.parts:
            if part.inline_data is not None:
                # Get Gemini SDK Image wrapper
                gemini_image = part.as_image()
                # Convert to PIL Image
                generated_image = Image.open(BytesIO(gemini_image.image_bytes))
                break

        if not generated_image:
            raise ValueError(f"No image generated for {shot_id}")

        # Save image
        image_filename = f"{shot_id}_child.png"
        image_path = self.child_shots_dir / image_filename

        save_image_with_metadata(
            generated_image,
            image_path,
            metadata={
                "shot_id": shot_id,
                "parent_shot_id": parent_id,
                "characters": characters,
                "location": location_name,
                "generated_at": datetime.now().isoformat()
            }
        )

        return image_path
