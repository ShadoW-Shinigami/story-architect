"""
Agent 9: Child Verification with Regeneration
Verifies child shot images using Gemini 2.5 Pro multimodal capabilities.
Can regenerate failed shots with feedback for up to 3 attempts.
Checks both accuracy to description and consistency with parent shot.
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
from core.validators import ChildShotsOutput, VerificationResult
from core.image_utils import save_image_with_metadata


class ChildVerificationAgent(BaseAgent):
    """Agent for verifying and regenerating child shot images."""

    def __init__(self, gemini_client, config, session_dir: Path):
        """Initialize Child Verification Agent."""
        super().__init__(gemini_client, config, "agent_9")
        self.session_dir = Path(session_dir)
        self.assets_dir = self.session_dir / "assets"
        self.child_shots_dir = self.assets_dir / "child_shots"
        self.max_verification_retries = config.get("max_retries", 3)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.consistency_threshold = config.get("consistency_threshold", 0.6)
        self.soft_failure_mode = config.get("soft_failure_mode", True)

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary")

        required_keys = ["child_shots", "parent_shots", "scene_breakdown", "shot_breakdown", "shot_grouping", "character_grids"]
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
        """Verify (and regenerate if needed) child shot images."""
        logger.info(f"{self.agent_name}: Verifying child shot images...")

        child_shots = input_data["child_shots"]
        parent_shots = input_data["parent_shots"]
        scene_breakdown = input_data["scene_breakdown"]
        shot_breakdown = input_data["shot_breakdown"]
        shot_grouping = input_data["shot_grouping"]
        character_grids = input_data["character_grids"]

        # Create lookups
        shots_by_id = {
            shot["shot_id"]: shot
            for shot in shot_breakdown.get("shots", [])
        }

        # Build comprehensive image lookup including BOTH parent and child shots
        # This is critical for grandchildren (child-of-child) to find their immediate parent
        parent_images_by_id = {
            parent["shot_id"]: self.session_dir / parent["image_path"]
            for parent in parent_shots
        }

        # Add child shots to lookup (for grandchildren support)
        for child in child_shots:
            child_id = child["shot_id"]
            child_path = self.session_dir / child["image_path"]
            parent_images_by_id[child_id] = child_path

        grids_by_chars = {
            tuple(sorted(grid["characters"])): self.session_dir / grid["grid_path"]
            for grid in character_grids
        }

        # Find parent_id for each child
        child_to_parent = self._map_child_to_parent(shot_grouping)

        # Verify each child shot
        verified_shots = []

        for child_shot in child_shots:
            shot_id = child_shot["shot_id"]

            try:
                logger.info(f"Verifying child shot: {shot_id}")

                # Get shot details
                shot_details = shots_by_id.get(shot_id)
                if not shot_details:
                    logger.warning(f"Shot details not found for {shot_id}")
                    continue

                # Get parent shot ID
                parent_id = child_to_parent.get(shot_id)
                parent_image_path = parent_images_by_id.get(parent_id) if parent_id else None

                # Verify with regeneration capability
                verification_result = self._verify_and_regenerate(
                    child_shot,
                    shot_details,
                    scene_breakdown,
                    parent_image_path,
                    grids_by_chars
                )

                # Update child shot data
                child_shot["verification_status"] = verification_result["status"]
                child_shot["final_verification"] = verification_result["final_result"]
                child_shot["verification_history"] = verification_result["history"]
                child_shot["attempts"] = verification_result["attempts"]

                # Update image path if regenerated
                if "new_image_path" in verification_result:
                    child_shot["image_path"] = verification_result["new_image_path"]
                    # CRITICAL: Update lookup so grandchildren use the NEW regenerated path
                    new_full_path = self.session_dir / verification_result["new_image_path"]
                    parent_images_by_id[shot_id] = new_full_path
                    logger.debug(f"Updated parent_images_by_id with regenerated path for {shot_id}")

                verified_shots.append(child_shot)

                status = verification_result["status"]
                logger.info(f"✓ Verification for {shot_id}: {status}")

            except Exception as e:
                logger.error(f"Verification failed for {shot_id}: {str(e)}")

                if self.soft_failure_mode:
                    child_shot["verification_status"] = "soft_failure"
                    child_shot["final_verification"] = {
                        "approved": False,
                        "confidence": 0.0,
                        "issues": [str(e)],
                        "recommendation": "manual_review"
                    }
                    verified_shots.append(child_shot)
                    logger.warning(f"Soft failure for {shot_id}, continuing...")
                else:
                    raise

        # Prepare output
        output = {
            "child_shots": verified_shots,
            "total_child_shots": len(verified_shots),
            "metadata": {
                "session_id": self.session_dir.name,
                "verified_at": datetime.now().isoformat(),
                "total_verified": sum(1 for s in verified_shots if s["verification_status"] == "verified"),
                "total_soft_failures": sum(1 for s in verified_shots if s["verification_status"] == "soft_failure")
            }
        }

        logger.info(
            f"{self.agent_name}: Verified {len(verified_shots)} child shots "
            f"({output['metadata']['total_verified']} approved, "
            f"{output['metadata']['total_soft_failures']} soft failures)"
        )

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

    def _verify_and_regenerate(
        self,
        child_shot: Dict[str, Any],
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        parent_image_path: Optional[Path],
        grids_by_chars: Dict[tuple, Path]
    ) -> Dict[str, Any]:
        """Verify child image and regenerate with feedback if verification fails."""
        from google.genai import types

        shot_id = child_shot["shot_id"]
        image_path = self.session_dir / child_shot["image_path"]

        verification_history = []
        best_result = None
        best_confidence = 0.0
        current_image_path = image_path

        for attempt in range(self.max_verification_retries):
            try:
                logger.debug(f"Verification attempt {attempt + 1}/{self.max_verification_retries}")

                # Load child image
                if not current_image_path.exists():
                    raise FileNotFoundError(f"Image not found: {current_image_path}")

                child_image = Image.open(current_image_path)

                # Load parent image if available
                parent_image = None
                if parent_image_path and parent_image_path.exists():
                    parent_image = Image.open(parent_image_path)

                # Verify
                verification_prompt = self._format_verification_prompt(shot_details, has_parent=parent_image is not None)

                contents = [child_image]
                if parent_image:
                    contents.append(parent_image)
                contents.append(verification_prompt)

                response = self.client.client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=contents,
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

                    if attempt > 0:
                        try:
                            return_data["new_image_path"] = str(current_image_path.relative_to(self.session_dir))
                        except ValueError:
                            return_data["new_image_path"] = str(current_image_path)

                    return return_data

                # Not approved - regenerate for next attempt
                if attempt < self.max_verification_retries - 1:
                    logger.warning(f"{shot_id}: Not approved, regenerating...")

                    # Regenerate with feedback
                    new_image_path = self._regenerate_child_shot(
                        shot_id,
                        shot_details,
                        scene_breakdown,
                        parent_image_path,
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

        if current_image_path != image_path:
            try:
                return_data["new_image_path"] = str(current_image_path.relative_to(self.session_dir))
            except ValueError:
                return_data["new_image_path"] = str(current_image_path)

        return return_data

    def _regenerate_child_shot(
        self,
        shot_id: str,
        shot_details: Dict[str, Any],
        scene_breakdown: Dict[str, Any],
        parent_image_path: Optional[Path],
        grids_by_chars: Dict[tuple, Path],
        verification_issues: List[str],
        attempt_number: int
    ) -> Path:
        """Regenerate child shot with feedback from verification."""
        from google.genai import types

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
        parent_image = Image.open(parent_image_path)

        # Get shot components
        first_frame = shot_details.get("first_frame", "")
        characters = shot_details.get("characters", [])
        location_name = shot_details.get("location", "")

        # Get descriptions
        character_descriptions = self._get_character_physical_descriptions(characters, scene_breakdown)

        # Find grid if needed
        char_combo = tuple(sorted(characters))
        grid_path = grids_by_chars.get(char_combo)

        # Format prompt with FEEDBACK
        prompt = self._format_regeneration_prompt(
            shot_id,
            first_frame,
            character_descriptions,
            location_name,
            verification_issues
        )

        # Prepare contents
        contents = [parent_image]
        if grid_path and grid_path.exists():
            grid_image = Image.open(grid_path)
            contents.append(grid_image)
        contents.append(prompt)

        # Generate
        response = self.client.client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="16:9"),
                temperature=0.8,
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
        image_filename = f"{shot_id}_child_retry{attempt_number}.png"
        new_image_path = self.child_shots_dir / image_filename

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

    def _format_regeneration_prompt(
        self,
        shot_id: str,
        first_frame: str,
        character_descriptions: List[Dict[str, str]],
        location: str,
        verification_issues: List[str]
    ) -> str:
        """Format prompt with verification feedback for child regeneration."""

        # Load Agent 8's prompt template (for child generation)
        agent8_prompt_path = Path("prompts/agent_8_prompt.txt")
        if not agent8_prompt_path.exists():
            raise ValueError("Agent 8 prompt template not found")

        with open(agent8_prompt_path, 'r', encoding='utf-8') as f:
            generation_template = f.read()

        # Format character descriptions
        char_desc_text = ""
        for idx, char in enumerate(character_descriptions, 1):
            char_desc_text += f"""
CHARACTER {idx}:
PHYSICAL TRAITS: {char['physical_description']}

When editing the parent shot, identify this character by these exact physical traits.

"""

        # Format base prompt
        base_prompt = generation_template.format(
            shot_id=shot_id,
            parent_id="unknown",  # Not critical for regeneration
            first_frame=first_frame,
            character_descriptions=char_desc_text,
            location=location
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
IMPORTANT: Address ALL of the above issues in your regeneration. This is a retry - correct these specific problems while maintaining visual continuity with the parent shot.

"""

        return base_prompt + feedback_section

    def _format_verification_prompt(self, shot_details: Dict[str, Any], has_parent: bool) -> str:
        """Format verification prompt."""
        first_frame = shot_details.get("first_frame", "")
        characters = shot_details.get("characters", [])
        location = shot_details.get("location", "")

        prompt = f"""
You are a quality control expert for AI-generated video sequences.

Analyze this child shot image{"and compare it with the parent shot image (second image provided)" if has_parent else ""}.

EXPECTED CHILD SHOT DESCRIPTION:
{first_frame}

EXPECTED CHARACTERS: {", ".join(characters)}
EXPECTED LOCATION: {location}

Verification Criteria:
1. COMPOSITION: Does the child shot match its description?
2. CHARACTERS: Are all specified characters present and visually correct?
3. LOCATION: Does the setting match the described location?
4. VISUAL QUALITY: Is the image clear and cinematic?
"""

        if has_parent:
            prompt += """
5. CONSISTENCY: Does it maintain visual consistency with the parent shot?
6. CONTINUITY: Is there smooth visual continuity from parent to child?
"""

        prompt += """

Respond ONLY with valid JSON in this exact format:
{
  "approved": true or false,
  "confidence": 0.0 to 1.0,
  "issues": ["list of specific issues", "empty if approved"],
  "recommendation": "approve" or "regenerate" or "manual_review"
}

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

            logger.warning("Could not parse verification JSON")
            return {
                "approved": False,
                "confidence": 0.0,
                "issues": ["Could not parse verification response"],
                "recommendation": "manual_review"
            }
