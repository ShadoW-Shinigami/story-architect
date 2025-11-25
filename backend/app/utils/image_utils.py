"""
Async Image Utilities
Helper functions for Phase 2 image generation pipeline.
"""

import asyncio
import base64
import json
import re
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from loguru import logger


def extract_character_combinations(shot_breakdown: Dict[str, Any]) -> List[Tuple[str, ...]]:
    """
    Analyze Agent 3 shot breakdown to find unique character combinations.
    Only returns combinations that actually appear in shots (max 3-4 per shot).
    """
    combinations = set()

    shots = shot_breakdown.get("shots", [])
    logger.debug(f"Extracting character combinations from {len(shots)} shots")

    for shot in shots:
        characters_in_shot = shot.get("characters", [])
        if 1 <= len(characters_in_shot) <= 4:
            combo = tuple(sorted(characters_in_shot))
            combinations.add(combo)
            logger.debug(f"Shot {shot.get('shot_id', 'UNKNOWN')}: Added combo {combo}")

    result = sorted(list(combinations), key=lambda x: (len(x), x))
    logger.info(f"Found {len(result)} unique character combinations: {result}")
    return result


def create_character_grid(
    character_images: List[Image.Image],
    character_names: List[str],
    output_size: Tuple[int, int] = (1920, 1080),
    padding: int = 40
) -> Image.Image:
    """Create a 16:9 grid of character images for reference."""
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
        cols = int(num_chars ** 0.5) + (1 if num_chars % int(num_chars ** 0.5) else 0)
        rows = (num_chars + cols - 1) // cols

    canvas = Image.new('RGB', output_size, color='white')

    available_width = output_size[0] - padding * (cols + 1)
    available_height = output_size[1] - padding * (rows + 1)

    cell_width = available_width // cols
    cell_height = available_height // rows
    cell_size = min(cell_width, cell_height)

    total_grid_width = cols * cell_size + (cols + 1) * padding
    total_grid_height = rows * cell_size + (rows + 1) * padding
    offset_x = (output_size[0] - total_grid_width) // 2
    offset_y = (output_size[1] - total_grid_height) // 2

    for idx, (img, name) in enumerate(zip(character_images, character_names)):
        row = idx // cols
        col = idx % cols

        img_resized = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)

        x = offset_x + padding + col * (cell_size + padding)
        y = offset_y + padding + row * (cell_size + padding)

        canvas.paste(img_resized, (x, y))

    logger.info(f"Created character grid: {num_chars} characters in {cols}x{rows} layout")
    return canvas


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    img_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_bytes))


async def save_image_with_metadata(
    image: Image.Image,
    file_path: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """Save image with optional metadata (async)."""
    def _save():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(file_path)
        if metadata:
            metadata_path = file_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        return file_path

    result = await asyncio.to_thread(_save)
    logger.debug(f"Saved image to {file_path}")
    return result


def verify_image_quality(image: Image.Image) -> Dict[str, Any]:
    """Basic quality checks for generated images."""
    width, height = image.size
    min_size = 512
    is_sufficient_size = width >= min_size and height >= min_size

    extrema = image.convert("L").getextrema()
    is_blank = extrema[0] == extrema[1]

    aspect_ratio = width / height if height > 0 else 0

    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "is_sufficient_size": is_sufficient_size,
        "is_blank": is_blank,
        "quality_ok": is_sufficient_size and not is_blank
    }


def slugify(text: str) -> str:
    """Convert text to filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text if text else "unnamed"


def get_aspect_ratio_from_image(image_path: Path) -> str:
    """Detect aspect ratio from image file."""
    with Image.open(image_path) as img:
        width, height = img.size

    ratio = width / height

    if 0.9 <= ratio <= 1.1:
        return "1:1"
    elif 1.7 <= ratio <= 1.85:
        return "16:9"
    elif 0.5 <= ratio <= 0.6:
        return "9:16"
    elif 1.25 <= ratio <= 1.4:
        return "4:3"
    else:
        return "16:9" if width > height else "9:16"


def optimize_image_for_base64(
    image_path: Path,
    max_size_mb: float = 7.0,
    output_format: str = "JPEG",
    quality: int = 85
) -> Tuple[str, int]:
    """Optimize image for base64 encoding, returning (base64_str, final_size_bytes)."""
    with Image.open(image_path) as img:
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        buffer = BytesIO()
        img.save(buffer, format=output_format, quality=quality, optimize=True)
        img_bytes = buffer.getvalue()

        current_quality = quality
        while len(img_bytes) > max_size_mb * 1024 * 1024 and current_quality > 30:
            current_quality -= 10
            buffer = BytesIO()
            img.save(buffer, format=output_format, quality=current_quality, optimize=True)
            img_bytes = buffer.getvalue()

        base64_str = base64.b64encode(img_bytes).decode('utf-8')
        return base64_str, len(img_bytes)


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
