"""
Agent 7: Async Parent Verification with Regeneration
Verifies parent shot images using Gemini multimodal and regenerates failures.
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
from app.schemas.validators import VerificationResult


class ParentVerificationAgent(AsyncBaseAgent):
    """Async agent for verifying and regenerating parent shot images."""

    def __init__(
        self,
        session_id: str,
        config: dict,
        gemini_client: AsyncGeminiClient
    ):
        super().__init__(
            agent_name="agent_7",
            session_id=session_id,
            config=config
        )
        self.client = gemini_client
        self.parent_shots_dir = self._ensure_directory("assets/parent_shots")

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
            "prompts/agent_7_verification_prompt.txt"
        ))
        self.modifier_template_path = Path(config.get(
            "modifier_prompt_file",
            "prompts/agent_7_prompt_modifier.txt"
        ))
        self.agent6_template_path = Path(config.get(
            "agent_6_prompt_file",
            "prompts/agent_6_prompt.txt"
        ))

        # Load templates if they exist (graceful degradation)
        self.verification_template = self._load_template(self.verification_template_path)
        self.modifier_template = self._load_template(self.modifier_template_path)

    def _load_template(self, template_path: Path) -> Optional[str]:
        """Load a template file, returning None if not found."""
        # Try multiple locations
        search_paths = [
            template_path,
            Path("prompts") / template_path.name,
            Path("../prompts") / template_path.name,
            self.session_dir.parent / "prompts" / template_path.name if hasattr(self, 'session_dir') else None
        ]

        for path in search_paths:
            if path and path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.warning(f"Failed to load template {path}: {e}")

        logger.warning(f"Template not found: {template_path}")
        return None

    async def validate_input(self, input_data: Any) -> None:
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")
        if "parent_shots" not in input_data:
            raise ValueError("Input must contain 'parent_shots'")

    async def validate_output(self, output_data: Any) -> None:
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

    async def process(
        self,
        input_data: Any,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Verify and regenerate parent shot images as needed."""
        logger.info(f"{self.agent_name}: Verifying parent shot images...")

        # Extract parent_shots list from Agent 6's full output dict
        agent_6_output = input_data.get("parent_shots", {})
        parent_shots = agent_6_output.get("parent_shots", []) if isinstance(agent_6_output, dict) else agent_6_output

        scene_breakdown = input_data.get("scene_breakdown", {})
        shot_breakdown = input_data.get("shot_breakdown", {})

        # Extract character_grids list from Agent 5's full output dict
        agent_5_output = input_data.get("character_grids", {})
        character_grids = agent_5_output.get("character_grids", []) if isinstance(agent_5_output, dict) else agent_5_output

        total_shots = len(parent_shots)

        if progress_callback:
            await progress_callback("Starting verification...", 0.05, None, None)

        # Create lookups
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        grids_by_chars = {
            tuple(sorted(grid["characters"])): grid["grid_path"]
            for grid in character_grids
        }

        verified_shots = []

        for idx, parent_shot in enumerate(parent_shots):
            shot_id = parent_shot["shot_id"]

            if progress_callback:
                await progress_callback(
                    f"Verifying: {shot_id}",
                    0.05 + (0.9 * (idx / max(total_shots, 1))),
                    idx + 1,
                    total_shots
                )

            try:
                logger.info(f"Verifying parent shot: {shot_id}")

                shot_details = shots_by_id.get(shot_id, {})

                # Verify with regeneration capability
                verification_result = await self._verify_and_regenerate(
                    parent_shot,
                    shot_details,
                    scene_breakdown,
                    grids_by_chars
                )

                # Update shot data
                parent_shot["verification_status"] = verification_result["status"]
                parent_shot["final_verification"] = verification_result["final_result"]
                parent_shot["verification_history"] = verification_result.get("history", [])
                parent_shot["attempts"] = verification_result["attempts"]

                if "new_image_path" in verification_result:
                    parent_shot["image_path"] = verification_result["new_image_path"]

                verified_shots.append(parent_shot)
                logger.info(f"✓ Verification for {shot_id}: {verification_result['status']}")

            except Exception as e:
                logger.error(f"Verification failed for {shot_id}: {str(e)}")

                if self.soft_failure_mode:
                    parent_shot["verification_status"] = "soft_failure"
                    parent_shot["final_verification"] = {
                        "approved": False,
                        "confidence": 0.0,
                        "issues": [{"category": "Execution Error", "description": str(e)}],
                        "recommendation": "manual_review"
                    }
                    verified_shots.append(parent_shot)
                else:
                    raise

        output = {
            "parent_shots": verified_shots,
            "total_parent_shots": len(verified_shots),
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

        logger.info(f"{self.agent_name}: Verified {len(verified_shots)} parent shots")
        return output

    async def _verify_and_regenerate(
        self,
        parent_shot: Dict[str, Any],
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        grids_by_chars: Dict[tuple, str]
    ) -> Dict[str, Any]:
        """Verify image and regenerate if verification fails."""
        shot_id = parent_shot["shot_id"]
        image_path = self.session_dir / parent_shot["image_path"]

        verification_history = []
        best_result = None
        best_confidence = 0.0
        current_image_path = image_path

        for attempt in range(self.max_verification_retries):
            try:
                logger.debug(f"Verification attempt {attempt + 1}/{self.max_verification_retries}")

                if not current_image_path.exists():
                    raise FileNotFoundError(f"Image not found: {current_image_path}")

                # Load current image
                image = await asyncio.to_thread(Image.open, current_image_path)

                # Verify image with Gemini
                result = await self._verify_image(image, shot_details)
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

                # Not approved - regenerate for next attempt
                if attempt < self.max_verification_retries - 1:
                    logger.warning(
                        f"{shot_id}: Not approved (confidence: {result['confidence']}), regenerating..."
                    )

                    new_image_path = await self._regenerate_parent_shot(
                        shot_id,
                        shot_details,
                        grids_by_chars,
                        result.get("issues", []),
                        attempt + 1,
                        parent_shot
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

        # All attempts failed - soft failure
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

    async def _verify_image(
        self,
        image: Image.Image,
        shot_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify an image against shot requirements using Gemini."""
        characters = shot_details.get("characters", [])
        location = shot_details.get("location", "")
        first_frame = shot_details.get("first_frame", "")

        # Use template file if loaded, otherwise fallback to inline
        if self.verification_template:
            verification_prompt = self.verification_template.format(
                expected_characters=', '.join(characters) if characters else 'None specified',
                location=location,
                first_frame=first_frame
            )
        else:
            # Fallback inline prompt (simplified)
            logger.warning(f"{self.agent_name}: Verification template not loaded, using inline fallback")
            verification_prompt = f"""
Analyze this image as a parent shot for film production.

EXPECTED REQUIREMENTS:
- Characters: {', '.join(characters) if characters else 'None specified'}
- Location: {location}
- Frame description: {first_frame}

VERIFY:
1. Are the correct characters present (no extras, none missing)?
2. Does the location match the description?
3. Is the composition cinematic and professional?
4. Are lighting and colors consistent?

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
                contents=[image, verification_prompt],
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

    async def _regenerate_parent_shot(
        self,
        shot_id: str,
        shot_details: Dict[str, Any],
        grids_by_chars: Dict[tuple, str],
        verification_issues: List[Dict[str, str]],
        attempt_number: int,
        original_parent_shot: Dict[str, Any]
    ) -> Path:
        """Regenerate parent shot with intelligent feedback from verification."""
        characters = shot_details.get("characters", [])
        first_frame = shot_details.get("first_frame", "")
        location = shot_details.get("location", "")

        # Load grid if available
        grid_image = None
        grid_path = None
        if characters:
            char_combo = tuple(sorted(characters))
            grid_path = grids_by_chars.get(char_combo)
            if grid_path:
                full_grid_path = self.session_dir / grid_path
                if full_grid_path.exists():
                    grid_image = await asyncio.to_thread(Image.open, full_grid_path)

        # Step 1: Load original prompt from metadata if available
        image_path = self.session_dir / original_parent_shot["image_path"]
        metadata_path = image_path.with_suffix('.json')

        original_optimized_prompt = None
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                original_optimized_prompt = metadata.get("prompts", {}).get("optimized_prompt")
                logger.debug(f"{shot_id}: Loaded original prompt from metadata")
            except Exception as e:
                logger.warning(f"{shot_id}: Could not load metadata: {e}")

        # Fallback base prompt if no metadata
        if not original_optimized_prompt:
            original_optimized_prompt = f"""
CINEMATIC SHOT
SHOT ID: {shot_id}
FRAME: {first_frame}
LOCATION: {location}
CHARACTERS: {', '.join(characters) if characters else 'None'}
"""

        # Step 2: Intelligently rewrite prompt using pattern analysis
        regen_prompt = await self._rewrite_prompt_with_feedback(
            original_optimized_prompt,
            verification_issues,
            shot_details,
            shot_id
        )

        generated_image = None

        # Try FAL first
        if self.image_provider == "fal" and is_fal_available():
            try:
                if grid_image:
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        await asyncio.to_thread(grid_image.save, tmp.name)
                        tmp_grid_path = Path(tmp.name)

                    try:
                        generated_image, _ = await generate_with_fal_edit(
                            prompt=regen_prompt,
                            image_paths=[tmp_grid_path],
                            model="fal-ai/nano-banana-pro/edit",
                            width=1920,
                            height=1080,
                        )
                    finally:
                        tmp_grid_path.unlink(missing_ok=True)
                else:
                    generated_image, _ = await generate_with_fal_text_to_image(
                        prompt=regen_prompt,
                        model="fal-ai/nano-banana-pro",
                        width=1920,
                        height=1080,
                    )
            except Exception as e:
                logger.warning(f"FAL regeneration failed: {e}")
                generated_image = None

        # Fallback to Gemini
        if generated_image is None:
            try:
                from google.genai import types

                contents = [grid_image, regen_prompt] if grid_image else [regen_prompt]

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
        image_filename = f"{shot_id}_parent_retry{attempt_number}.png"
        new_image_path = self.parent_shots_dir / image_filename

        regen_metadata = {
            "shot_id": shot_id,
            "attempt": attempt_number,
            "grid_used": str(grid_path) if grid_path else "none",
            "image_provider": self.image_provider,
            "issues_addressed": verification_issues,
            "prompts": {
                "original_optimized_prompt": original_optimized_prompt,
                "rewritten_prompt": regen_prompt
            },
            "generated_at": datetime.now().isoformat()
        }

        await save_image_with_metadata(
            generated_image,
            new_image_path,
            metadata=regen_metadata
        )

        return new_image_path

    def _analyze_failure_pattern(
        self,
        issues: List[Dict[str, str]],
        shot_details: Dict[str, Any]
    ) -> str:
        """
        Analyze issues to identify failure patterns for parent shot generation.

        Args:
            issues: List of categorized issues from verification
            shot_details: Parent shot target details

        Returns:
            Pattern analysis string for prompt modifier
        """
        categories = [issue.get("category", "") for issue in issues]

        patterns = []

        # Character count issues
        if "Extra Character" in categories or "Duplicate Character" in categories:
            char_count = len(shot_details.get("characters", []))
            patterns.append(f"EXTRA_CHARACTER: More than {char_count} characters appeared. Needs explicit 'EXACTLY {char_count} characters' constraint.")

        # Missing characters
        if "Missing Character" in categories:
            patterns.append("MISSING_CHARACTER: Expected character not visible. Needs explicit placement instruction for each character.")

        # Poor integration
        if "Pasted Character" in categories:
            patterns.append("POOR_INTEGRATION: Character looks artificial. Needs lighting direction/intensity match, scale/depth matching, and natural placement instructions.")

        # Grid artifacts
        if "Grid Artifact" in categories:
            patterns.append("GRID_ARTIFACT: Character grid structure leaked into output. Needs strong 'unified scene, NO grid elements, fill entire frame' emphasis at prompt start.")

        # Composition issues
        if "Bad Composition" in categories:
            patterns.append("COMPOSITION_ISSUE: Vague framing instructions caused awkward layout. Needs objective spatial description of frame regions and character positions.")

        # Proportion issues
        if "Bad Proportions" in categories:
            patterns.append("PROPORTION_ISSUE: Character scales inconsistent. Needs explicit scale/distance references and perspective cues.")

        # Lighting mismatch
        if "Lighting Mismatch" in categories:
            patterns.append("LIGHTING_MISMATCH: Characters have different lighting. Needs unified light source description affecting all characters equally.")

        # Poor location
        if "Poor Location" in categories:
            patterns.append("POOR_LOCATION: Environment lacks detail or doesn't match description. Needs more specific environmental details and architectural elements.")

        return "\n".join(f"- {pattern}" for pattern in patterns) if patterns else "- GENERAL_ISSUES: Address the specific issues listed."

    async def _rewrite_prompt_with_feedback(
        self,
        original_prompt: str,
        categorized_issues: List[Dict[str, str]],
        shot_details: Dict[str, Any],
        shot_id: str
    ) -> str:
        """
        Use Gemini Pro to intelligently rewrite prompt based on verification issues.

        Args:
            original_prompt: The prompt that was used
            categorized_issues: List of issues with category and description
            shot_details: Parent shot target details
            shot_id: Shot identifier for logging

        Returns:
            Rewritten prompt addressing root causes
        """
        # Check if modifier template is loaded
        if not self.modifier_template:
            logger.warning(f"{shot_id}: Modifier template not loaded, using fallback")
            issues_text = "\n".join(f"- {issue['category']}: {issue['description']}" for issue in categorized_issues)
            return original_prompt + f"\n\nCRITICAL ISSUES TO FIX:\n{issues_text}"

        # Analyze failure patterns
        pattern_analysis = self._analyze_failure_pattern(categorized_issues, shot_details)

        # Format issues for template
        issues_text = "\n".join(
            f"- Category: {issue['category']}\n  Description: {issue['description']}"
            for issue in categorized_issues
        )

        # Create modification prompt using template
        try:
            modification_prompt = self.modifier_template.format(
                original_prompt=original_prompt,
                categorized_issues=issues_text,
                pattern_analysis=pattern_analysis
            )
        except KeyError as e:
            logger.warning(f"{shot_id}: Template format error: {e}, using fallback")
            return original_prompt + f"\n\nCRITICAL ISSUES TO FIX:\n{issues_text}"

        try:
            from google.genai import types

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
                logger.info(f"{shot_id}: Prompt intelligently rewritten (patterns: {len(pattern_analysis.split(chr(10)))} detected)")
                return rewritten_prompt
            else:
                logger.warning(f"{shot_id}: Prompt rewrite returned empty, using original")
                return original_prompt

        except Exception as e:
            logger.error(f"{shot_id}: Prompt rewrite failed: {str(e)}, using original")
            return original_prompt
