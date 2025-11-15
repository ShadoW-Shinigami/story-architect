"""
Image Utilities
Helper functions for Phase 2 image generation pipeline.
"""

import base64
import json
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from loguru import logger


def extract_character_combinations(shot_breakdown: Dict[str, Any]) -> List[Tuple[str, ...]]:
    """
    Analyze Agent 3 shot breakdown to find unique character combinations.
    Only returns combinations that actually appear in shots (max 3-4 per shot).

    Args:
        shot_breakdown: Agent 3 output JSON

    Returns:
        List of unique character combinations as tuples
        Example: [("Alice",), ("Alice", "Bob"), ("Alice", "Bob", "Charlie")]
    """
    combinations = set()

    shots = shot_breakdown.get("shots", [])
    logger.debug(f"Extracting character combinations from {len(shots)} shots")

    for shot in shots:
        characters_in_shot = shot.get("characters", [])

        # Only process shots with 1-4 characters (user constraint)
        if 1 <= len(characters_in_shot) <= 4:
            # Create sorted tuple for deduplication (order-independent)
            combo = tuple(sorted(characters_in_shot))
            combinations.add(combo)
            logger.debug(f"Shot {shot.get('shot_id', 'UNKNOWN')}: Added combo {combo}")

    result = sorted(list(combinations), key=lambda x: (len(x), x))

    logger.info(
        f"Found {len(result)} unique character combinations needed for grids: {result}"
    )

    return result


def create_character_grid(
    character_images: List[Image.Image],
    character_names: List[str],
    output_size: Tuple[int, int] = (1920, 1080),
    padding: int = 40
) -> Image.Image:
    """
    Create a 16:9 grid of character images for reference.

    Args:
        character_images: List of PIL Image objects (should be 1024x1024 each)
        character_names: List of character names (for labeling)
        output_size: Output dimensions (default 1920x1080 for 16:9)
        padding: Padding between images in pixels

    Returns:
        PIL Image object with character grid
    """
    num_chars = len(character_images)

    if num_chars == 0:
        raise ValueError("No character images provided")

    # Calculate optimal grid layout
    if num_chars == 1:
        cols, rows = 1, 1
    elif num_chars == 2:
        cols, rows = 2, 1
    elif num_chars == 3:
        cols, rows = 3, 1
    elif num_chars == 4:
        cols, rows = 2, 2
    else:
        # For more than 4, use square-ish layout
        cols = int(num_chars ** 0.5) + (1 if num_chars % int(num_chars ** 0.5) else 0)
        rows = (num_chars + cols - 1) // cols

    # Create blank canvas (white background)
    canvas = Image.new('RGB', output_size, color='white')

    # Calculate cell size with padding
    available_width = output_size[0] - padding * (cols + 1)
    available_height = output_size[1] - padding * (rows + 1)

    cell_width = available_width // cols
    cell_height = available_height // rows

    # Ensure cells are square (use smaller dimension)
    cell_size = min(cell_width, cell_height)

    # Center the grid
    total_grid_width = cols * cell_size + (cols + 1) * padding
    total_grid_height = rows * cell_size + (rows + 1) * padding
    offset_x = (output_size[0] - total_grid_width) // 2
    offset_y = (output_size[1] - total_grid_height) // 2

    # Paste images
    for idx, (img, name) in enumerate(zip(character_images, character_names)):
        row = idx // cols
        col = idx % cols

        # Resize image to fit cell
        img_resized = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)

        # Calculate position
        x = offset_x + padding + col * (cell_size + padding)
        y = offset_y + padding + row * (cell_size + padding)

        canvas.paste(img_resized, (x, y))

        # Optional: Add character name label below image
        # (You can uncomment this if you want text labels)
        # try:
        #     draw = ImageDraw.Draw(canvas)
        #     font = ImageFont.load_default()
        #     text_bbox = draw.textbbox((0, 0), name, font=font)
        #     text_width = text_bbox[2] - text_bbox[0]
        #     text_x = x + (cell_size - text_width) // 2
        #     text_y = y + cell_size + 5
        #     draw.text((text_x, text_y), name, fill='black', font=font)
        # except Exception as e:
        #     logger.warning(f"Could not add text label: {e}")

    logger.info(
        f"Created character grid: {num_chars} characters in {cols}x{rows} layout"
    )

    return canvas


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert PIL Image to base64 string.

    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.

    Args:
        base64_string: Base64 encoded image string

    Returns:
        PIL Image object
    """
    img_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_bytes))


def save_image_with_metadata(
    image: Image.Image,
    file_path: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save image with optional metadata.

    Args:
        image: PIL Image object
        file_path: Path to save image
        metadata: Optional metadata to save alongside image

    Returns:
        Path to saved image
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save image (format auto-detected from .png extension)
    image.save(file_path)

    # Save metadata if provided
    if metadata:
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    logger.debug(f"Saved image to {file_path}")

    return file_path


def get_character_images_for_combination(
    combination: Tuple[str, ...],
    character_images_dict: Dict[str, Image.Image]
) -> List[Image.Image]:
    """
    Get character images for a specific combination.

    Args:
        combination: Tuple of character names (e.g., ("Alice", "Bob"))
        character_images_dict: Dict mapping character names to PIL Images

    Returns:
        List of PIL Images in same order as combination
    """
    images = []

    for char_name in combination:
        if char_name in character_images_dict:
            images.append(character_images_dict[char_name])
        else:
            logger.warning(f"Character image not found: {char_name}")

    return images


def verify_image_quality(image: Image.Image) -> Dict[str, Any]:
    """
    Basic quality checks for generated images.

    Args:
        image: PIL Image object

    Returns:
        Dict with quality metrics
    """
    # Check dimensions
    width, height = image.size

    # Check if image is too small
    min_size = 512
    is_sufficient_size = width >= min_size and height >= min_size

    # Check if image is blank (all one color)
    extrema = image.convert("L").getextrema()
    is_blank = extrema[0] == extrema[1]

    # Calculate aspect ratio
    aspect_ratio = width / height if height > 0 else 0

    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "is_sufficient_size": is_sufficient_size,
        "is_blank": is_blank,
        "quality_ok": is_sufficient_size and not is_blank
    }


def create_comparison_grid(
    images: List[Image.Image],
    labels: List[str],
    grid_size: Tuple[int, int] = (2, 2)
) -> Image.Image:
    """
    Create a comparison grid of images (useful for verification displays).

    Args:
        images: List of PIL Images
        labels: List of labels for each image
        grid_size: (cols, rows) grid dimensions

    Returns:
        PIL Image with comparison grid
    """
    cols, rows = grid_size
    padding = 20

    # Use first image dimensions as reference
    if images:
        ref_width, ref_height = images[0].size
    else:
        ref_width, ref_height = 1024, 1024

    # Calculate canvas size
    canvas_width = cols * ref_width + (cols + 1) * padding
    canvas_height = rows * ref_height + (rows + 1) * padding

    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')

    # Paste images
    for idx, img in enumerate(images[:cols * rows]):
        row = idx // cols
        col = idx % cols

        # Resize image if needed
        img_resized = img.resize((ref_width, ref_height), Image.Resampling.LANCZOS)

        # Calculate position
        x = padding + col * (ref_width + padding)
        y = padding + row * (ref_height + padding)

        canvas.paste(img_resized, (x, y))

    return canvas
