"""
Agent 7: Parent Verification with Regeneration
Verifies parent shot images using Gemini 2.5 Pro multimodal capabilities.
Can regenerate failed shots with feedback for up to 3 attempts.
"""

import json
from io import BytesIO
from typing import Any, Dict, List
from datetime import datetime
from pathlib import Path
from PIL import Image
from loguru import logger
from pydantic import ValidationError

from agents.base_agent import BaseAgent
from core.validators import ParentShotsOutput, VerificationResult
from core.image_utils import save_image_with_metadata


class ParentVerificationAgent(BaseAgent):
    """Agent for verifying and regenerating parent shot images."""

    def __init__(self, gemini_client, config, session_dir: Path):
        """Initialize Parent Verification Agent."""
        super().__init__(gemini_client, config, "agent_7")
        self.session_dir = Path(session_dir)
        self.assets_dir = self.session_dir / "assets"
        self.parent_shots_dir = self.assets_dir / "parent_shots"
        self.max_verification_retries = config.get("max_retries", 3)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.soft_failure_mode = config.get("soft_failure_mode", True)

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = ["parent_shots", "scene_breakdown", "shot_breakdown", "shot_grouping", "character_grids"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Input must contain '{key}'")

        return True

    def validate_output(self, output_data: Any) -> bool:
        """Validate output."""
        if not isinstance(output_data, dict):
            raise ValueError("Output must be a dictionary")

        return True

    def process(self, input_data: Any) -> Dict[str, Any]:
        """Verify (and regenerate if needed) parent shot images."""
        logger.info(f"{self.agent_name}: Verifying parent shot images...")

        parent_shots = input_data["parent_shots"]
        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]
        character_grids = input_data["character_grids"]

        # Create lookups
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        grids_by_chars = {
            tuple(sorted(grid["characters"])): grid["grid_path"]
            for grid in character_grids
        }

        # Verify each parent shot
        verified_shots = []

        for parent_shot in parent_shots:
            shot_id = parent_shot["shot_id"]

            try:
                logger.info(f"Verifying parent shot: {shot_id}")

                # Get shot details
                shot_details = shots_by_id.get(shot_id)
                if not shot_details:
                    logger.warning(f"Shot details not found for {shot_id}")
                    continue

                # Verify with regeneration capability
                verification_result = self._verify_and_regenerate(
                    parent_shot,
                    shot_details,
                    scene_breakdown,
                    grids_by_chars
                )

                # Update parent shot data
                parent_shot["verification_status"] = verification_result["status"]
                parent_shot["final_verification"] = verification_result["final_result"]
                parent_shot["verification_history"] = verification_result["history"]
                parent_shot["attempts"] = verification_result["attempts"]
                # Update image path if regenerated
                if "new_image_path" in verification_result:
                    parent_shot["image_path"] = verification_result["new_image_path"]

                verified_shots.append(parent_shot)

                status = verification_result["status"]
                logger.info(f"✓ Verification for {shot_id}: {status}")

            except Exception as e:
                logger.error(f"Verification failed for {shot_id}: {str(e)}")

                if self.soft_failure_mode:
                    parent_shot["verification_status"] = "soft_failure"
                    parent_shot["final_verification"] = {
                        "approved": False,
                        "confidence": 0.0,
                        "issues": [str(e)],
                        "recommendation": "manual_review"
                    }
                    verified_shots.append(parent_shot)
                    logger.warning(f"Soft failure for {shot_id}, continuing...")
                else:
                    raise

        # Prepare output
        output = {
            "parent_shots": verified_shots,
            "total_parent_shots": len(verified_shots),
            "metadata": {
                "session_id": self.session_dir.name,
                "verified_at": datetime.now().isoformat(),
                "total_verified": sum(1 for s in verified_shots if s["verification_status"] == "verified"),
                "total_soft_failures": sum(1 for s in verified_shots if s["verification_status"] == "soft_failure")
            }
        }

        logger.info(
            f"{self.agent_name}: Verified {len(verified_shots)} parent shots "
            f"({output['metadata']['total_verified']} approved, "
            f"{output['metadata']['total_soft_failures']} soft failures)"
        )

        return output

    def _verify_and_regenerate(
        self,
        parent_shot: Dict[str, Any],
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        grids_by_chars: Dict[tuple, str]
    ) -> Dict[str, Any]:
        """Verify image and regenerate with feedback if verification fails."""
        from google.genai import types

        shot_id = parent_shot["shot_id"]
        image_path = self.session_dir / parent_shot["image_path"]

        verification_history = []
        best_result = None
        best_confidence = 0.0
        current_image_path = image_path

        for attempt in range(self.max_verification_retries):
            try:
                logger.debug(f"Verification attempt {attempt + 1}/{self.max_verification_retries}")

                # Load current image
                if not current_image_path.exists():
                    raise FileNotFoundError(f"Image not found: {current_image_path}")

                image = Image.open(current_image_path)

                # Verify image
                verification_prompt = self._format_verification_prompt(shot_details)

                response = self.client.client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[image, verification_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=2000
                    )
                )

                # Check if response.text is None (API failure, safety filter, etc.)
                if response.text is None:
                    raise ValueError(
                        "Gemini API returned empty response (response.text is None). "
                        "This may be due to safety filters, rate limiting, or API errors."
                    )

                # Parse verification result
                result = self._parse_verification_response(response.text)
                verification_history.append(result)

                # Track best result
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

                    # Return new path if we regenerated
                    if attempt > 0:
                        try:
                            return_data["new_image_path"] = str(current_image_path.relative_to(self.session_dir))
                        except ValueError:
                            return_data["new_image_path"] = str(current_image_path)

                    return return_data

                # Not approved - regenerate for next attempt
                if attempt < self.max_verification_retries - 1:
                    logger.warning(
                        f"{shot_id}: Not approved (confidence: {result['confidence']}), "
                        f"issues: {result.get('issues', [])}. Regenerating..."
                    )

                    # Regenerate with feedback
                    new_image_path = self._regenerate_parent_shot(
                        shot_id,
                        shot_details,
                        scene_breakdown,
                        grids_by_chars,
                        result.get("issues", []),
                        attempt + 1
                    )

                    current_image_path = new_image_path
                    logger.info(f"Regenerated {shot_id}, will verify new image")

            except Exception as e:
                logger.error(f"Verification attempt {attempt + 1} failed: {str(e)}")
                verification_history.append({
                    "approved": False,
                    "confidence": 0.0,
                    "issues": [str(e)],
                    "recommendation": "regenerate"
                })

        # All attempts failed - soft failure
        logger.warning(f"{shot_id}: All verification attempts failed, soft failure")

        fallback_result = {
            "approved": False,
            "confidence": 0.0,
            "issues": ["All verification attempts failed"],
            "recommendation": "manual_review"
        }

        return_data = {
            "status": "soft_failure",
            "final_result": best_result or (verification_history[-1] if verification_history else fallback_result),
            "history": verification_history,
            "attempts": self.max_verification_retries
        }

        # Return new path if we regenerated
        if current_image_path != image_path:
            try:
                return_data["new_image_path"] = str(current_image_path.relative_to(self.session_dir))
            except ValueError:
                return_data["new_image_path"] = str(current_image_path)

        return return_data

    def _regenerate_parent_shot(
        self,
        shot_id: str,
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        grids_by_chars: Dict[tuple, str],
        verification_issues: List[str],
        attempt_number: int
    ) -> Path:
        """Regenerate parent shot with feedback from verification."""
        from google.genai import types

        # Get shot components
        first_frame = shot_details.get("first_frame", "")
        characters = shot_details.get("characters", [])
        scene_id = shot_details.get("scene_id", "")
        location_name = shot_details.get("location", "")

        # Get descriptions
        character_descriptions = self._get_character_physical_descriptions(characters, scene_breakdown)
        location_description = self._get_location_full_description(location_name, scene_id, scene_breakdown)

        # Find grid if needed
        grid_image = None
        if characters:
            char_combo = tuple(sorted(characters))
            grid_path = grids_by_chars.get(char_combo)
            if grid_path:
                full_grid_path = self.session_dir / grid_path
                if full_grid_path.exists():
                    grid_image = Image.open(full_grid_path)

        # Format prompt with FEEDBACK
        prompt = self._format_regeneration_prompt(
            shot_id,
            first_frame,
            character_descriptions,
            location_description,
            verification_issues
        )

        # Generate
        if grid_image:
            contents = [grid_image, prompt]
        else:
            contents = [prompt]

        response = self.client.client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="16:9"),
                temperature=0.8,  # Slightly higher for variation
            ),
        )

        # Extract and save
        generated_image = None
        for part in response.parts:
            if part.inline_data is not None:
                gemini_image = part.as_image()
                generated_image = Image.open(BytesIO(gemini_image.image_bytes))
                break

        if not generated_image:
            raise ValueError(f"No image generated for {shot_id} retry")

        # Save with retry suffix
        image_filename = f"{shot_id}_parent_retry{attempt_number}.png"
        new_image_path = self.parent_shots_dir / image_filename

        save_image_with_metadata(generated_image, new_image_path, metadata={
            "shot_id": shot_id,
            "attempt": attempt_number,
            "regenerated_with_feedback": verification_issues
        })

        logger.info(f"Regenerated {shot_id}: {new_image_path.name}")
        return new_image_path

    def _get_character_physical_descriptions(self, character_names: List[str], scene_breakdown: Dict[str, Any]) -> List[Dict[str, str]]:
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

    def _get_location_full_description(self, location_name: str, scene_id: str, scene_breakdown: Dict[str, Any]) -> str:
        """Extract full location description."""
        for scene in scene_breakdown.get("scenes", []):
            if scene.get("scene_id") == scene_id:
                return scene.get("location", {}).get("description", location_name)
        return location_name

    def _format_regeneration_prompt(
        self,
        shot_id: str,
        first_frame: str,
        character_descriptions: List[Dict[str, str]],
        location_description: str,
        verification_issues: List[str]
    ) -> str:
        """Format prompt with verification feedback for regeneration."""

        # Load Agent 6's prompt template (for generation)
        agent6_prompt_path = Path("prompts/agent_6_prompt.txt")
        if not agent6_prompt_path.exists():
            raise ValueError("Agent 6 prompt template not found")

        with open(agent6_prompt_path, 'r', encoding='utf-8') as f:
            generation_template = f.read()

        # Format character descriptions
        char_desc_text = ""
        for idx, char in enumerate(character_descriptions, 1):
            char_desc_text += f"""
CHARACTER {idx} (from grid position {idx}):
PHYSICAL APPEARANCE: {char['physical_description']}

This character must be placed in the scene exactly as they appear in the grid, maintaining their physical appearance, clothing, hair, facial features, and all other visual characteristics precisely.

"""

        # Format base prompt
        base_prompt = generation_template.format(
            shot_id=shot_id,
            first_frame=first_frame,
            location_description=location_description,
            character_descriptions=char_desc_text
        )

        # Add verification feedback
        feedback_section = f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: PREVIOUS ATTEMPT HAD THESE ISSUES - FIX THEM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        for idx, issue in enumerate(verification_issues, 1):
            feedback_section += f"{idx}. {issue}\n"

        feedback_section += """
IMPORTANT: Address ALL of the above issues in your regeneration. The previous image did not meet requirements. This is a retry - make sure to correct these specific problems.

"""

        return base_prompt + feedback_section

    def _format_verification_prompt(self, shot_details: Dict[str, Any]) -> str:
        """Format verification prompt for Gemini 2.5 Pro."""
        first_frame = shot_details.get("first_frame", "")
        characters = shot_details.get("characters", [])
        location = shot_details.get("location", "")

        prompt = f"""
You are a quality control expert for AI-generated images used in video production.

Analyze this generated image and verify it matches the required specifications.

EXPECTED FIRST FRAME DESCRIPTION:
{first_frame}

EXPECTED CHARACTERS: {", ".join(characters)}
EXPECTED LOCATION: {location}

Verification Criteria:
1. COMPOSITION: Does the image composition match the description?
2. CHARACTERS: Are all specified characters present and visually distinguishable?
3. LOCATION: Does the setting/environment match the described location?
4. VISUAL QUALITY: Is the image clear, well-lit, and cinematic?
5. CONSISTENCY: Are there any hallucinations, artifacts, or unexpected elements?

Respond ONLY with valid JSON in this exact format:
{{
  "approved": true or false,
  "confidence": 0.0 to 1.0,
  "issues": ["list of specific issues found", "empty array if approved"],
  "recommendation": "approve" or "regenerate" or "manual_review"
}}

Provide your verification now:
"""
        return prompt

    def _parse_verification_response(self, response_text: str) -> Dict[str, Any]:
        """Parse verification JSON response."""
        try:
            result = json.loads(response_text)

            if "approved" not in result:
                result["approved"] = False
            if "confidence" not in result:
                result["confidence"] = 0.0
            if "issues" not in result:
                result["issues"] = []
            if "recommendation" not in result:
                result["recommendation"] = "manual_review"

            return result

        except json.JSONDecodeError:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')

            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

            logger.warning("Could not parse verification JSON, using fallback")
            return {
                "approved": False,
                "confidence": 0.0,
                "issues": ["Could not parse verification response"],
                "recommendation": "manual_review"
            }
