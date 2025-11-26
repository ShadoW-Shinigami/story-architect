"""
Agent 8: Async Child Image Generator
Generates child shot images by editing parent shots using Gemini or FAL.
Uses physical trait descriptions and character grids for consistency.
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
    generate_with_fal_edit,
    is_fal_available,
)


class ChildImageGeneratorAgent(AsyncBaseAgent):
    """Async agent for generating child shot images by editing parent shots."""

    def __init__(
        self,
        session_id: str,
        config: dict,
        gemini_client: AsyncGeminiClient
    ):
        super().__init__(
            agent_name="agent_8",
            session_id=session_id,
            config=config
        )
        self.client = gemini_client
        self.child_shots_dir = self._ensure_directory("assets/child_shots")
        self.parent_shots_dir = self._ensure_directory("assets/parent_shots")
        self.grids_dir = self._ensure_directory("assets/grids")

        self.image_provider = config.get("image_provider", "gemini").lower()
        self.use_optimizer = config.get("use_prompt_optimizer", True)
        self.fal_edit_model = config.get("fal_edit_model", "fal-ai/nano-banana-pro/edit")

        # Model configuration from config (not hardcoded!)
        self.optimizer_model = config.get("optimizer_model", "gemini-3-pro-preview")  # LLM for prompt optimization
        self.image_model = config.get("image_model", "gemini-2.5-flash-image")  # Gemini image generation

        # Load templates from file
        self.optimizer_template_path = Path(config.get(
            "optimizer_prompt_file",
            "prompts/agent_8_optimizer_prompt.txt"
        ))
        self.prompt_template_path = Path(config.get(
            "prompt_file",
            "prompts/agent_8_prompt.txt"
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
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = ["scene_breakdown", "shot_breakdown", "shot_grouping", "parent_shots", "character_grids"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Input must contain '{key}'")

    async def validate_output(self, output_data: Any) -> None:
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        if "child_shots" not in output_data:
            raise ValueError("Output must contain 'child_shots'")

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Generate child shot images by editing parent shots."""
        logger.info(f"{self.agent_name}: Editing parent shots to create child shots...")

        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]
        shot_grouping = input_data["shot_grouping"]

        # Extract parent_shots list from Agent 7's full output dict
        agent_7_output = input_data.get("parent_shots", {})
        parent_shots = agent_7_output.get("parent_shots", []) if isinstance(agent_7_output, dict) else agent_7_output

        # Extract character_grids list from Agent 5's full output dict
        agent_5_output = input_data["character_grids"]
        character_grids = agent_5_output.get("character_grids", []) if isinstance(agent_5_output, dict) else agent_5_output

        if progress_callback:
            await progress_callback("Preparing child shot data...", 0.05, None, None)

        # Create lookups
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        # Dynamic image lookup - starts with parents, grows as children generated
        available_images_by_id = {
            parent["shot_id"]: self.session_dir / parent["image_path"]
            for parent in parent_shots
        }

        grids_by_chars = {
            tuple(sorted(grid["characters"])): self.session_dir / grid["grid_path"]
            for grid in character_grids
        }

        # Extract all child shots recursively
        child_shot_list = self._extract_child_shots(shot_grouping)
        total_shots = len(child_shot_list)
        logger.info(f"Found {total_shots} child shots to generate")

        if progress_callback:
            await progress_callback(f"Found {total_shots} child shots", 0.1, None, None)

        child_shots_data = []

        for idx, child_shot_info in enumerate(child_shot_list):
            try:
                shot_id = child_shot_info["shot_id"]
                parent_id = child_shot_info["parent_shot_id"]

                if progress_callback:
                    await progress_callback(
                        f"Generating child shot: {shot_id}",
                        0.1 + (0.85 * (idx / max(total_shots, 1))),
                        idx + 1,
                        total_shots
                    )

                logger.info(f"Generating child shot: {shot_id} (parent: {parent_id})")

                shot_details = shots_by_id.get(shot_id)
                if not shot_details:
                    logger.warning(f"Shot details not found for {shot_id}, skipping")
                    continue

                parent_image_path = available_images_by_id.get(parent_id)
                if not parent_image_path or not parent_image_path.exists():
                    logger.warning(f"Parent image not found for {parent_id}, skipping")
                    continue

                # Generate child image
                child_image_path = await self._generate_child_shot_image(
                    shot_id,
                    parent_id,
                    shot_details,
                    scene_breakdown,
                    parent_image_path,
                    grids_by_chars,
                    shots_by_id
                )

                # Add to lookup for grandchildren support
                available_images_by_id[shot_id] = child_image_path

                rel_path = str(child_image_path.relative_to(self.session_dir))

                child_shots_data.append({
                    "shot_id": shot_id,
                    "scene_id": shot_details.get("scene_id"),
                    "image_path": rel_path,
                    "generation_timestamp": datetime.now().isoformat(),
                    "verification_status": "pending",
                    "attempts": 1,
                    "final_verification": None,
                    "verification_history": []
                })

                logger.info(f"✓ Generated child shot: {shot_id}")

            except Exception as e:
                logger.error(f"Failed to generate child shot {shot_id}: {str(e)}")
                continue  # Soft failure

        output = {
            "child_shots": child_shots_data,
            "total_child_shots": len(child_shots_data),
            "metadata": {
                "session_id": self.session_id,
                "generated_at": datetime.now().isoformat(),
                "image_provider": self.image_provider
            }
        }

        if progress_callback:
            await progress_callback(
                f"Child shot generation complete: {len(child_shots_data)} shots",
                1.0, None, None
            )

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
                recurse(child, child["shot_id"])

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

    def _detect_edit_type(
        self,
        shot_id: str,
        parent_id: str,
        shot_details: Dict[str, Any],
        shots_by_id: Dict[str, Any]
    ) -> str:
        """Detect the type of edit operation."""
        parent_chars = set(shots_by_id.get(parent_id, {}).get("characters", []))
        child_chars = set(shot_details.get("characters", []))

        if len(child_chars) > len(parent_chars):
            return "add_character"
        elif len(child_chars) < len(parent_chars):
            return "remove_character"
        elif parent_chars != child_chars:
            return "character_swap"
        else:
            first_frame = shot_details.get("first_frame", "").lower()
            if any(word in first_frame for word in ["close-up", "close up", "zoom", "tighter"]):
                return "camera_change"
            elif any(word in first_frame for word in ["expression", "smile", "frown", "angry", "happy"]):
                return "expression_change"
            else:
                return "camera_change"

    async def _optimize_edit_prompt(self, verbose_prompt: str, edit_type: str) -> str:
        """Use Gemini Pro to optimize the verbose edit prompt."""
        if not self.use_optimizer:
            logger.debug("Edit prompt optimization disabled in config")
            return verbose_prompt

        # Use loaded template if available, otherwise fallback to inline
        if self.optimizer_template:
            try:
                optimization_prompt = self.optimizer_template.format(
                    verbose_edit_description=verbose_prompt,
                    edit_type=edit_type
                )
            except KeyError as e:
                logger.warning(f"Template format error: {e}, using fallback")
                optimization_prompt = self._get_fallback_optimization_prompt(verbose_prompt, edit_type)
        else:
            logger.warning("Optimizer template not loaded, using fallback")
            optimization_prompt = self._get_fallback_optimization_prompt(verbose_prompt, edit_type)

        try:
            from google.genai import types

            logger.debug("Optimizing edit prompt with Gemini Pro...")
            response = await asyncio.to_thread(
                self.client.client.models.generate_content,
                model=self.optimizer_model,
                contents=[optimization_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048
                )
            )

            if response.text:
                optimized = response.text.strip()
                logger.info(f"Edit prompt optimized: {len(verbose_prompt)} → {len(optimized)} chars")
                return optimized

        except Exception as e:
            logger.warning(f"Edit prompt optimization failed: {e}")

        return verbose_prompt

    def _get_fallback_optimization_prompt(self, verbose_prompt: str, edit_type: str) -> str:
        """Get fallback optimization prompt when template is not available."""
        return f"""You are an expert prompt optimizer for AI image editing. Your task is to take a verbose edit description and distill it into precise editing instructions.

INPUT EDIT DESCRIPTION:
{verbose_prompt}

EDIT TYPE: {edit_type}

OPTIMIZATION RULES:
1. Keep essential visual details (character appearances, poses, expressions)
2. Remove redundant language
3. Use specific, actionable editing instructions
4. Maintain character consistency requirements
5. Limit to 400 words maximum

OUTPUT: Provide only the optimized edit prompt, no explanations."""

    async def _generate_child_shot_image(
        self,
        shot_id: str,
        parent_id: str,
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        parent_image_path: Path,
        grids_by_chars: Dict[tuple, Path],
        shots_by_id: Dict[str, Any]
    ) -> Path:
        """Generate child shot by editing parent shot."""
        edit_type = self._detect_edit_type(shot_id, parent_id, shot_details, shots_by_id)
        logger.debug(f"Edit type detected: {edit_type}")

        # Load parent image
        parent_image = await asyncio.to_thread(Image.open, parent_image_path)
        logger.debug(f"Using parent image: {parent_image_path.name}")

        # Get shot components
        first_frame = shot_details.get("first_frame", "")
        characters = shot_details.get("characters", [])
        location_name = shot_details.get("location", "")

        # Get character descriptions
        character_descriptions = self._get_character_physical_descriptions(
            characters, scene_breakdown
        )

        # Find matching character grid
        char_combo = tuple(sorted(characters))
        grid_path = grids_by_chars.get(char_combo)

        grid_image = None
        if grid_path and grid_path.exists():
            grid_image = await asyncio.to_thread(Image.open, grid_path)
            logger.debug(f"Using character grid: {grid_path.name}")

        # Format character descriptions
        char_desc_text = ""
        for idx, char in enumerate(character_descriptions, 1):
            char_desc_text += f"""CHARACTER {idx}:
PHYSICAL TRAITS: {char['physical_description']}

When editing the parent shot, identify this character by these exact physical traits. If adding this character to the scene, use the character grid reference (if provided) to ensure their appearance matches precisely.

"""

        # Build verbose prompt using template if available
        if self.prompt_template:
            verbose_prompt = self.prompt_template.format(
                shot_id=shot_id,
                parent_id=parent_id,
                first_frame=first_frame,
                character_descriptions=char_desc_text,
                location=location_name
            )
        else:
            # Fallback inline prompt (simplified)
            logger.warning(f"{self.agent_name}: Prompt template not loaded, using inline fallback")
            verbose_prompt = f"""
EDIT PARENT SHOT TO CREATE CHILD SHOT

SHOT ID: {shot_id}
PARENT SHOT: {parent_id}
EDIT TYPE: {edit_type}

TARGET FRAME: {first_frame}
LOCATION: {location_name}

CHARACTERS IN SCENE:
{char_desc_text}

EDITING INSTRUCTIONS:
- Edit the parent shot to match the new frame description
- Maintain character appearances exactly as described
- Keep lighting and style consistent with parent
- For {edit_type}: Apply appropriate changes smoothly
- Use character grid reference for consistency

OUTPUT: A cinematic 16:9 frame that looks like a natural continuation.
"""

        # Optimize prompt
        optimized_prompt = await self._optimize_edit_prompt(verbose_prompt, edit_type)

        generated_image = None
        fal_seed = None

        # Try FAL first if configured
        if self.image_provider == "fal" and is_fal_available():
            try:
                logger.info(f"Using FAL for child shot: {shot_id}")

                # Save parent image temporarily
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    await asyncio.to_thread(parent_image.save, tmp.name)
                    tmp_parent_path = Path(tmp.name)

                image_paths = [tmp_parent_path]
                tmp_grid_path = None

                if grid_image:
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_grid:
                        await asyncio.to_thread(grid_image.save, tmp_grid.name)
                        tmp_grid_path = Path(tmp_grid.name)
                    image_paths.append(tmp_grid_path)

                try:
                    generated_image, fal_seed = await generate_with_fal_edit(
                        prompt=optimized_prompt,
                        image_paths=image_paths,
                        model=self.fal_edit_model,
                        width=1920,
                        height=1080,
                    )
                    logger.info(f"Generated with FAL (seed: {fal_seed})")
                finally:
                    tmp_parent_path.unlink(missing_ok=True)
                    if tmp_grid_path:
                        tmp_grid_path.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(f"FAL generation failed: {e}, falling back to Gemini")
                generated_image = None

        # Use Gemini if FAL not available or failed
        if generated_image is None:
            logger.info(f"Using Gemini for child shot: {shot_id}")

            try:
                from google.genai import types

                contents = [parent_image]
                if grid_image:
                    contents.append(grid_image)
                contents.append(optimized_prompt)

                response = await asyncio.to_thread(
                    self.client.client.models.generate_content,
                    model=self.image_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(aspect_ratio="16:9"),
                        temperature=0.8,
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
        image_filename = f"{shot_id}_child.png"
        image_path = self.child_shots_dir / image_filename

        metadata = {
            "shot_id": shot_id,
            "parent_shot_id": parent_id,
            "characters": characters,
            "location": location_name,
            "generated_at": datetime.now().isoformat(),
            "edit_type": edit_type,
            "image_provider": "fal" if fal_seed else "gemini",
            "grid_used": str(grid_path) if grid_path and grid_path.exists() else None,
            "prompts": {
                "verbose_prompt": verbose_prompt,
                "optimized_prompt": optimized_prompt,
                "optimizer_used": self.use_optimizer
            }
        }

        if fal_seed is not None:
            metadata["fal_seed"] = fal_seed

        await save_image_with_metadata(generated_image, image_path, metadata=metadata)

        return image_path
