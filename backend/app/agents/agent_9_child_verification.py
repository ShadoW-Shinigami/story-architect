"""
Agent 9: Async Child Verification with Regeneration
Verifies child shot images using Gemini multimodal capabilities.
Can regenerate failed shots with feedback for up to 3 attempts.
Checks both accuracy to description and consistency with parent shot.
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
from app.schemas.validators import VerificationResult


class ChildVerificationAgent(AsyncBaseAgent):
    """Async agent for verifying and regenerating child shot images."""

    def __init__(
        self,
        session_id: str,
        config: dict,
        gemini_client: AsyncGeminiClient
    ):
        super().__init__(
            agent_name="agent_9",
            session_id=session_id,
            config=config
        )
        self.client = gemini_client
        self.child_shots_dir = self._ensure_directory("assets/child_shots")

        self.max_verification_retries = config.get("max_retries", 3)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.consistency_threshold = config.get("consistency_threshold", 0.6)
        self.soft_failure_mode = config.get("soft_failure_mode", True)
        self.image_provider = config.get("image_provider", "gemini").lower()

        # Model configuration from config (not hardcoded!)
        self.verification_model = config.get("verification_model", "gemini-3-pro-preview")  # LLM for verification
        self.image_model = config.get("image_model", "gemini-2.5-flash-image")  # Gemini image generation

        # Load template paths from config
        self.verification_template_path = Path(config.get(
            "verification_prompt_file",
            "prompts/agent_9_verification_prompt.txt"
        ))
        self.modifier_template_path = Path(config.get(
            "modifier_prompt_file",
            "prompts/agent_9_prompt_modifier.txt"
        ))
        self.agent8_template_path = Path(config.get(
            "agent_8_prompt_file",
            "prompts/agent_8_prompt.txt"
        ))

        # Load templates with graceful fallback
        self.verification_template = self._load_template(self.verification_template_path)
        self.modifier_template = self._load_template(self.modifier_template_path)

    async def validate_input(self, input_data: Any) -> None:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = ["child_shots", "parent_shots", "scene_breakdown", "shot_breakdown", "shot_grouping", "character_grids"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Input must contain '{key}'")

    async def validate_output(self, output_data: Any) -> None:
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

    def _load_template(self, template_path: Path) -> Optional[str]:
        """Load template file with fallback paths."""
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

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Verify and regenerate child shot images as needed."""
        logger.info(f"{self.agent_name}: Verifying child shot images...")

        # Extract child_shots list from Agent 8's full output dict
        agent_8_output = input_data.get("child_shots", {})
        child_shots = agent_8_output.get("child_shots", []) if isinstance(agent_8_output, dict) else agent_8_output

        # Extract parent_shots list from Agent 7's full output dict
        agent_7_output = input_data.get("parent_shots", {})
        parent_shots = agent_7_output.get("parent_shots", []) if isinstance(agent_7_output, dict) else agent_7_output

        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]
        shot_grouping = input_data["shot_grouping"]

        # Extract character_grids list from Agent 5's full output dict
        agent_5_output = input_data["character_grids"]
        character_grids = agent_5_output.get("character_grids", []) if isinstance(agent_5_output, dict) else agent_5_output

        total_shots = len(child_shots)

        if progress_callback:
            await progress_callback("Starting verification...", 0.05, None, None)

        # Create lookups
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        # Build comprehensive image lookup (parent + child)
        parent_images_by_id = {
            parent["shot_id"]: self.session_dir / parent["image_path"]
            for parent in parent_shots
        }

        # Add child shots to lookup (for grandchildren support)
        for child in child_shots:
            child_path = self.session_dir / child["image_path"]
            parent_images_by_id[child["shot_id"]] = child_path

        grids_by_chars = {
            tuple(sorted(grid["characters"])): self.session_dir / grid["grid_path"]
            for grid in character_grids
        }

        # Map child to parent
        child_to_parent = self._map_child_to_parent(shot_grouping)

        verified_shots = []

        for idx, child_shot in enumerate(child_shots):
            shot_id = child_shot["shot_id"]

            if progress_callback:
                await progress_callback(
                    f"Verifying: {shot_id}",
                    0.05 + (0.9 * (idx / max(total_shots, 1))),
                    idx + 1,
                    total_shots
                )

            try:
                logger.info(f"Verifying child shot: {shot_id}")

                shot_details = shots_by_id.get(shot_id)
                if not shot_details:
                    logger.warning(f"Shot details not found for {shot_id}")
                    continue

                parent_id = child_to_parent.get(shot_id)
                parent_image_path = parent_images_by_id.get(parent_id) if parent_id else None

                verification_result = await self._verify_and_regenerate(
                    child_shot,
                    shot_details,
                    scene_breakdown,
                    parent_image_path,
                    grids_by_chars,
                    parent_id,
                    shots_by_id
                )

                # Update child shot data
                child_shot["verification_status"] = verification_result["status"]
                child_shot["final_verification"] = verification_result["final_result"]
                child_shot["verification_history"] = verification_result.get("history", [])
                child_shot["attempts"] = verification_result["attempts"]

                if "new_image_path" in verification_result:
                    child_shot["image_path"] = verification_result["new_image_path"]
                    # Update lookup for grandchildren
                    new_full_path = self.session_dir / verification_result["new_image_path"]
                    parent_images_by_id[shot_id] = new_full_path

                verified_shots.append(child_shot)
                logger.info(f"✓ Verification for {shot_id}: {verification_result['status']}")

            except Exception as e:
                logger.error(f"Verification failed for {shot_id}: {str(e)}")

                if self.soft_failure_mode:
                    child_shot["verification_status"] = "soft_failure"
                    child_shot["final_verification"] = {
                        "approved": False,
                        "confidence": 0.0,
                        "issues": [{"category": "Execution Error", "description": str(e)}],
                        "recommendation": "manual_review"
                    }
                    verified_shots.append(child_shot)
                else:
                    raise

        output = {
            "child_shots": verified_shots,
            "total_child_shots": len(verified_shots),
            "metadata": {
                "session_id": self.session_id,
                "verified_at": datetime.now().isoformat(),
                "total_verified": sum(1 for s in verified_shots if s.get("verification_status") == "verified"),
                "total_soft_failures": sum(1 for s in verified_shots if s.get("verification_status") == "soft_failure")
            }
        }

        if progress_callback:
            await progress_callback(
                f"Verification complete: {len(verified_shots)} shots processed",
                1.0, None, None
            )

        logger.info(f"{self.agent_name}: Verified {len(verified_shots)} child shots")
        return output

    def _map_child_to_parent(self, shot_grouping: Dict[str, Any]) -> Dict[str, str]:
        """Map child shot IDs to parent shot IDs."""
        mapping = {}

        def recurse(grouped_shot):
            parent_id = grouped_shot["shot_id"]
            for child in grouped_shot.get("child_shots", []):
                mapping[child["shot_id"]] = parent_id
                recurse(child)

        for parent in shot_grouping.get("parent_shots", []):
            recurse(parent)

        return mapping

    async def _verify_and_regenerate(
        self,
        child_shot: Dict[str, Any],
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        parent_image_path: Optional[Path],
        grids_by_chars: Dict[tuple, Path],
        parent_id: Optional[str],
        shots_by_id: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify image and regenerate with feedback if verification fails."""
        shot_id = child_shot["shot_id"]
        image_path = self.session_dir / child_shot["image_path"]

        # Get parent details and detect edit type
        parent_shot_details = shots_by_id.get(parent_id) if parent_id else None
        edit_type = self._detect_edit_type(shot_details, parent_shot_details)

        verification_history = []
        best_result = None
        best_confidence = 0.0
        current_image_path = image_path

        for attempt in range(self.max_verification_retries):
            try:
                logger.debug(f"Verification attempt {attempt + 1}/{self.max_verification_retries}")

                if not current_image_path.exists():
                    raise FileNotFoundError(f"Image not found: {current_image_path}")

                child_image = await asyncio.to_thread(Image.open, current_image_path)

                parent_image = None
                if parent_image_path and parent_image_path.exists():
                    parent_image = await asyncio.to_thread(Image.open, parent_image_path)

                # Verify image
                result = await self._verify_image(
                    child_image, shot_details, parent_shot_details,
                    edit_type, parent_image is not None
                )
                verification_history.append(result)

                if result["confidence"] > best_confidence:
                    best_confidence = result["confidence"]
                    best_result = result

                # Check if approved
                if result["approved"] and result["confidence"] >= self.confidence_threshold:
                    logger.info(f"{shot_id}: Approved (confidence: {result['confidence']})")

                    return_data = {
                        "status": "verified",
                        "final_result": result,
                        "history": verification_history,
                        "attempts": attempt + 1
                    }

                    if attempt > 0:
                        return_data["new_image_path"] = str(current_image_path.relative_to(self.session_dir))

                    return return_data

                # Not approved - regenerate
                if attempt < self.max_verification_retries - 1:
                    logger.warning(f"{shot_id}: Not approved, regenerating...")

                    new_image_path = await self._regenerate_child_shot(
                        shot_id,
                        shot_details,
                        scene_breakdown,
                        parent_image_path,
                        grids_by_chars,
                        result.get("issues", []),
                        edit_type,
                        attempt + 1,
                        child_shot
                    )

                    current_image_path = new_image_path

            except Exception as e:
                logger.error(f"Verification attempt {attempt + 1} failed: {str(e)}")
                verification_history.append({
                    "approved": False,
                    "confidence": 0.0,
                    "issues": [{"category": "Error", "description": str(e)}],
                    "recommendation": "regenerate"
                })

        # All attempts failed
        logger.warning(f"{shot_id}: All verification attempts failed")

        return_data = {
            "status": "soft_failure",
            "final_result": best_result or {
                "approved": False,
                "confidence": 0.0,
                "issues": [{"category": "Verification Failure", "description": "All attempts failed"}],
                "recommendation": "manual_review"
            },
            "history": verification_history,
            "attempts": self.max_verification_retries
        }

        if current_image_path != image_path:
            return_data["new_image_path"] = str(current_image_path.relative_to(self.session_dir))

        return return_data

    def _detect_edit_type(
        self,
        shot_details: Dict[str, Any],
        parent_shot_details: Optional[Dict[str, Any]]
    ) -> str:
        """Detect the type of edit operation."""
        if not parent_shot_details:
            return "unknown"

        parent_chars = set(parent_shot_details.get("characters", []))
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
            return "expression_change"

    async def _verify_image(
        self,
        child_image: Image.Image,
        shot_details: Dict[str, Any],
        parent_shot_details: Optional[Dict[str, Any]],
        edit_type: str,
        has_parent: bool
    ) -> Dict[str, Any]:
        """Verify an image against shot requirements using Gemini."""
        characters = shot_details.get("characters", [])
        location = shot_details.get("location", "")
        first_frame = shot_details.get("first_frame", "")

        # Build parent context
        parent_context_prefix = " and compare it with the parent shot image (if provided)" if has_parent else ""
        parent_context = ""
        if has_parent and parent_shot_details:
            parent_chars = parent_shot_details.get("characters", [])
            parent_frame = parent_shot_details.get("first_frame", "")
            parent_context = f"""
PARENT SHOT CONTEXT (What we're editing from):
- Characters in parent: {', '.join(parent_chars)}
- Parent composition: {parent_frame}

NOTE: If the edit type is '{edit_type}', then differences in characters/composition may be INTENTIONAL.
"""

        # Use template file if loaded, otherwise fallback to inline
        if self.verification_template:
            verification_prompt = self.verification_template.format(
                parent_context_prefix=parent_context_prefix,
                expected_characters=', '.join(characters) if characters else 'None specified',
                location=location,
                first_frame=first_frame,
                edit_type=edit_type,
                parent_context=parent_context
            )
        else:
            # Fallback inline prompt (simplified)
            logger.warning(f"{self.agent_name}: Verification template not loaded, using inline fallback")
            verification_prompt = f"""
Analyze this child shot image for film production quality.

EXPECTED REQUIREMENTS:
- Characters: {', '.join(characters) if characters else 'None specified'}
- Location: {location}
- Frame description: {first_frame}
{parent_context}

VERIFY:
1. Are the correct characters present (considering edit type)?
2. Does the composition match the frame description?
3. Is the image cinematic and professional?
4. Is there consistency with parent shot style?

Return JSON:
{{
    "approved": true/false,
    "confidence": 0.0-1.0,
    "issues": [
        {{"category": "string", "description": "string"}}
    ],
    "recommendation": "approve" | "regenerate" | "manual_review"
}}
"""

        try:
            from google.genai import types

            response = await asyncio.to_thread(
                self.client.client.models.generate_content,
                model=self.verification_model,
                contents=[child_image, verification_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    response_json_schema=VerificationResult.model_json_schema()
                )
            )

            if response.text is None:
                raise ValueError(
                    "Gemini API returned empty response (response.text is None). "
                    "This may be due to safety filters, rate limiting, or API errors."
                )

            result = json.loads(response.text)
            return {
                "approved": result.get("approved", False),
                "confidence": result.get("confidence", 0.0),
                "issues": result.get("issues", []),
                "recommendation": result.get("recommendation", "manual_review")
            }

        except Exception as e:
            logger.error(f"Verification API call failed: {e}")

        return {
            "approved": False,
            "confidence": 0.0,
            "issues": [{"category": "API Error", "description": "Verification failed"}],
            "recommendation": "regenerate"
        }

    async def _regenerate_child_shot(
        self,
        shot_id: str,
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        parent_image_path: Optional[Path],
        grids_by_chars: Dict[tuple, Path],
        verification_issues: List[Dict[str, str]],
        edit_type: str,
        attempt_number: int,
        original_child_shot: Dict[str, Any]
    ) -> Path:
        """Regenerate child shot with intelligently rewritten prompt based on verification issues."""
        # Validate parent image exists
        if not parent_image_path:
            raise ValueError(
                f"Parent image path is None for shot {shot_id}. "
                "This shot may be a grandchild whose parent is not in the parent_images lookup. "
                "Check that the child shot's parent was successfully generated."
            )

        if not parent_image_path.exists():
            raise ValueError(
                f"Parent image required for child regeneration but file not found: {parent_image_path}. "
                f"Shot: {shot_id}"
            )

        # Load parent image
        parent_image = await asyncio.to_thread(Image.open, parent_image_path)

        # Find grid if needed
        characters = shot_details.get("characters", [])
        char_combo = tuple(sorted(characters))
        grid_path = grids_by_chars.get(char_combo)

        grid_image = None
        if grid_path and grid_path.exists():
            grid_image = await asyncio.to_thread(Image.open, grid_path)

        # Step 1: Load metadata from disk JSON file to get original optimized prompt
        image_path = self.session_dir / original_child_shot["image_path"]
        metadata_path = image_path.with_suffix('.json')

        original_optimized_prompt = None
        if metadata_path.exists():
            try:
                metadata_content = await asyncio.to_thread(metadata_path.read_text, encoding='utf-8')
                metadata = json.loads(metadata_content)
                original_optimized_prompt = metadata.get("prompts", {}).get("optimized_prompt")
                if original_optimized_prompt:
                    logger.debug(f"{shot_id}: Loaded optimized prompt from disk metadata: {metadata_path.name}")
            except Exception as e:
                logger.warning(f"{shot_id}: Failed to load metadata: {e}")

        # Fallback if no optimized prompt found
        if not original_optimized_prompt:
            logger.warning(f"{shot_id}: No optimized prompt in metadata, building fallback prompt")
            first_frame = shot_details.get("first_frame", "")
            location = shot_details.get("location", "")
            original_optimized_prompt = f"""
Generate a child shot for {shot_id}.
Edit type: {edit_type}
Frame: {first_frame}
Location: {location}
Characters: {', '.join(characters) if characters else 'None'}
"""

        # Step 2: Intelligently rewrite the OPTIMIZED prompt (not verbose!)
        rewritten_prompt = await self._rewrite_prompt_with_feedback(
            original_optimized_prompt,  # Rewrite the actual prompt that was sent to Flash
            verification_issues,
            shot_details,
            edit_type,
            shot_id
        )

        # Step 3: Use rewritten prompt for generation
        final_prompt = rewritten_prompt

        generated_image = None
        fal_seed = None

        # Try FAL first
        if self.image_provider == "fal" and is_fal_available():
            try:
                logger.info(f"Using FAL for child shot regeneration: {shot_id} (attempt {attempt_number})")

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
                    logger.debug(f"Using character grid for regeneration: {grid_path.name if grid_path else 'N/A'}")

                try:
                    generated_image, fal_seed = await generate_with_fal_edit(
                        prompt=final_prompt,
                        image_paths=image_paths,
                        model="fal-ai/nano-banana-pro/edit",
                        width=1920,
                        height=1080,
                    )
                    logger.info(f"Successfully regenerated child shot with FAL (seed: {fal_seed})")
                finally:
                    tmp_parent_path.unlink(missing_ok=True)
                    if tmp_grid_path:
                        tmp_grid_path.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(f"FAL regeneration failed: {e}, falling back to Gemini")
                generated_image = None

        # Fallback to Gemini
        if generated_image is None:
            try:
                from google.genai import types

                logger.info(f"Using Gemini for child shot regeneration: {shot_id} (attempt {attempt_number})")

                contents = [parent_image]
                if grid_image:
                    contents.append(grid_image)
                contents.append(final_prompt)

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
                logger.error(f"Gemini regeneration failed: {e}")
                raise

        if not generated_image:
            raise ValueError(f"Failed to regenerate {shot_id}")

        # Save regenerated image
        image_filename = f"{shot_id}_child_retry{attempt_number}.png"
        new_image_path = self.child_shots_dir / image_filename

        await save_image_with_metadata(
            generated_image,
            new_image_path,
            metadata={
                "shot_id": shot_id,
                "attempt": attempt_number,
                "edit_type": edit_type,
                "image_provider": self.image_provider if generated_image else "gemini",
                "grid_used": str(grid_path) if grid_path and grid_path.exists() else None,
                "prompts": {
                    "original_optimized_prompt": original_optimized_prompt,
                    "rewritten_prompt": rewritten_prompt
                },
                "regenerated_with_feedback": verification_issues,
                "generated_at": datetime.now().isoformat()
            }
        )

        return new_image_path

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
                if char.get("name"):
                    char_lookup[char["name"]] = char.get("description", "")

        for name in character_names:
            if name in char_lookup:
                descriptions.append({"name": name, "physical_description": char_lookup[name]})
            else:
                descriptions.append({"name": name, "physical_description": f"Character named {name}"})

        return descriptions

    def _analyze_failure_pattern(
        self,
        issues: List[Dict[str, str]],
        shot_details: Dict[str, Any],
        edit_type: str
    ) -> str:
        """
        Analyze issues to identify failure patterns and suggest rewrite strategy.

        Args:
            issues: List of categorized issues from verification
            shot_details: Child shot details
            edit_type: Type of edit operation

        Returns:
            Pattern analysis string for prompt modifier
        """
        categories = [issue.get("category", "") for issue in issues]

        patterns = []

        # OTS failure pattern (child-shot specific)
        first_frame_upper = shot_details.get("first_frame", "").upper()
        if "OTS Failure" in categories or (
            "Extra Character" in categories and
            any(ots in first_frame_upper for ots in ["OVER-THE-SHOULDER", "OVER THE SHOULDER", "OTS"])
        ):
            patterns.append(
                "OTS_SHOT_MISINTERPRETED: Model created new character instead of changing camera angle. "
                "Needs explicit spatial frame description with foreground/background positioning."
            )

        # Character duplication or extras
        if "Extra Character" in categories or "Duplicate Character" in categories:
            char_count = len(shot_details.get("characters", []))
            patterns.append(
                f"EXTRA_CHARACTER: More than {char_count} characters appeared. "
                f"Needs explicit 'EXACTLY {char_count} characters' constraint."
            )

        # Missing characters
        if "Missing Character" in categories:
            patterns.append(
                "MISSING_CHARACTER: Expected character not visible. "
                "Needs explicit placement instruction for each character."
            )

        # Poor integration (pasted look)
        if "Pasted Character" in categories:
            patterns.append(
                "POOR_INTEGRATION: Added character looks artificial. "
                "Needs lighting direction/intensity match, scale/depth matching, and spatial anchoring instructions."
            )

        # Grid artifacts
        if "Grid Artifact" in categories:
            patterns.append(
                "GRID_ARTIFACT: Reference grid structure leaked into output. "
                "Needs strong 'unified scene, no grid elements' emphasis at prompt start."
            )

        # Composition issues
        if "Bad Composition" in categories:
            patterns.append(
                "COMPOSITION_ISSUE: Vague framing instructions caused awkward layout. "
                "Needs objective spatial description of frame regions."
            )

        # Proportion issues
        if "Bad Proportions" in categories:
            patterns.append(
                "PROPORTION_ISSUE: Character scales inconsistent. "
                "Needs explicit scale/distance references and perspective cues."
            )

        # Lighting mismatch
        if "Lighting Mismatch" in categories:
            patterns.append(
                "LIGHTING_MISMATCH: Lighting inconsistent with parent shot. "
                "Needs explicit lighting direction and intensity matching instructions."
            )

        # Poor location match
        if "Poor Location" in categories:
            patterns.append(
                "POOR_LOCATION: Background/setting doesn't match parent. "
                "Needs explicit background continuity instructions."
            )

        return "\n".join(f"- {pattern}" for pattern in patterns) if patterns else "- GENERAL_ISSUES: Address the specific issues listed."

    async def _rewrite_prompt_with_feedback(
        self,
        original_prompt: str,
        categorized_issues: List[Dict[str, str]],
        shot_details: Dict[str, Any],
        edit_type: str,
        shot_id: str
    ) -> str:
        """
        Use Gemini Pro to intelligently rewrite prompt based on verification issues.

        Args:
            original_prompt: The optimized prompt that was used
            categorized_issues: List of issues with category and description
            shot_details: Child shot target details
            edit_type: Type of edit operation
            shot_id: Shot identifier for logging

        Returns:
            Rewritten prompt addressing root causes
        """
        from google.genai import types

        # Check if modifier template is available
        if not self.modifier_template:
            logger.warning(f"{shot_id}: Modifier template not found, using feedback append fallback")
            # Fallback: just append issues
            issues_text = "\n".join(
                f"- {issue['category']}: {issue['description']}"
                for issue in categorized_issues
            )
            return original_prompt + f"\n\nCRITICAL ISSUES TO FIX:\n{issues_text}"

        # Analyze failure patterns
        pattern_analysis = self._analyze_failure_pattern(categorized_issues, shot_details, edit_type)

        # Format issues for template
        issues_text = "\n".join(
            f"- Category: {issue['category']}\n  Description: {issue['description']}"
            for issue in categorized_issues
        )

        # Create modification prompt
        modification_prompt = self.modifier_template.format(
            original_prompt=original_prompt,
            categorized_issues=issues_text,
            pattern_analysis=pattern_analysis,
            edit_type=edit_type
        )

        try:
            logger.debug(f"{shot_id}: Rewriting prompt with Gemini Pro based on patterns...")

            response = await asyncio.to_thread(
                self.client.client.models.generate_content,
                model=self.verification_model,
                contents=[modification_prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=8192
                )
            )

            if response.text:
                rewritten_prompt = response.text.strip()
                pattern_count = len(pattern_analysis.split('\n'))
                logger.info(f"{shot_id}: Prompt intelligently rewritten (patterns: {pattern_count} detected)")
                return rewritten_prompt
            else:
                logger.warning(f"{shot_id}: Prompt rewrite returned empty, using original")
                return original_prompt

        except Exception as e:
            logger.error(f"{shot_id}: Prompt rewrite failed: {str(e)}, using original")
            return original_prompt
