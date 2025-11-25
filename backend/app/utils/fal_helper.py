"""
Async FAL AI integration helper for image generation and editing.
"""

import asyncio
import os
import time
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from io import BytesIO
import httpx
from PIL import Image
from loguru import logger

try:
    import fal_client
    FAL_AVAILABLE = True
except ImportError:
    FAL_AVAILABLE = False
    logger.warning("fal_client not installed. Install with: pip install fal-client")


def _check_fal_authentication() -> bool:
    """Check if FAL_KEY is configured."""
    fal_key = os.getenv("FAL_KEY")
    if not fal_key or fal_key == "your_fal_api_key_here":
        logger.error("FAL_KEY not configured")
        return False
    return True


def _get_fal_params_from_size(width: int, height: int) -> Tuple[str, str]:
    """Determine aspect_ratio and resolution based on dimensions."""
    max_dim = max(width, height)
    if max_dim >= 3000:
        resolution = "4K"
    elif max_dim >= 1500:
        resolution = "2K"
    else:
        resolution = "1K"

    ratio = width / height

    if 0.9 <= ratio <= 1.1:
        aspect_ratio = "1:1"
    elif 1.7 <= ratio <= 1.85:
        aspect_ratio = "16:9"
    elif 1.25 <= ratio <= 1.4:
        aspect_ratio = "4:3"
    elif 1.45 <= ratio <= 1.55:
        aspect_ratio = "3:2"
    elif 2.0 <= ratio <= 2.4:
        aspect_ratio = "21:9"
    elif 0.5 <= ratio <= 0.6:
        aspect_ratio = "9:16"
    elif 0.6 <= ratio <= 0.7:
        aspect_ratio = "2:3"
    elif 0.7 <= ratio <= 0.8:
        aspect_ratio = "3:4"
    elif 0.8 <= ratio <= 0.9:
        aspect_ratio = "4:5"
    else:
        aspect_ratio = "16:9" if width > height else "9:16" if height > width else "1:1"

    return aspect_ratio, resolution


def resolve_dimensions_from_config(config: Dict[str, Any], default_aspect: str = "16:9") -> Tuple[int, int]:
    """
    Calculate pixel width and height based on config settings.
    Supports both 'resolution/aspect_ratio' (Nano Banana) and 'width/height' (Seedream/Legacy) formats.

    Args:
        config: Agent or global configuration dictionary
        default_aspect: Default aspect ratio if not found (16:9 or 1:1)

    Returns:
        Tuple of (width, height)
    """
    # 1. Check for explicit legacy width/height (Seedream style)
    width = config.get("width")
    height = config.get("height")
    if width and height and isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height

    # 2. Parse resolution and aspect ratio (Nano Banana / Gemini style)
    resolution = config.get("resolution", "2K")
    aspect_ratio = config.get("aspect_ratio", default_aspect)

    # Base dimension from resolution
    if resolution == "4K":
        base = 3840
    elif resolution == "2K":
        base = 1920
    else:  # 1K
        base = 1024

    # Calculate dims based on aspect ratio
    if aspect_ratio == "1:1":
        if resolution == "2K":
            base = 2048
        return base, base

    elif aspect_ratio == "16:9":
        if resolution == "4K":
            return 3840, 2160
        if resolution == "2K":
            return 1920, 1080
        return 1024, 576

    elif aspect_ratio == "9:16":
        if resolution == "4K":
            return 2160, 3840
        if resolution == "2K":
            return 1080, 1920
        return 576, 1024

    elif aspect_ratio == "4:3":
        return base, int(base * 3 / 4)

    elif aspect_ratio == "3:2":
        return base, int(base * 2 / 3)

    elif aspect_ratio == "21:9":
        return base, int(base * 9 / 21)

    # Fallback
    return base, int(base * 9 / 16)


async def download_image_from_url(url: str) -> Image.Image:
    """Download an image from a URL asynchronously."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))


async def upload_image_to_fal(image_path: Path) -> str:
    """Upload a local image to FAL (runs in thread pool)."""
    def _upload():
        return fal_client.upload_file(str(image_path))

    for attempt in range(3):
        try:
            url = await asyncio.to_thread(_upload)
            logger.info(f"Uploaded {image_path.name} to FAL")
            return url
        except Exception as e:
            if attempt < 2:
                logger.warning(f"Upload attempt {attempt + 1} failed: {e}, retrying...")
                await asyncio.sleep(2 ** attempt)
            else:
                raise


async def generate_with_fal_text_to_image(
    prompt: str,
    model: str = "fal-ai/nano-banana-pro",
    width: int = 1024,
    height: int = 1024,
    num_images: int = 1,
    enable_safety_checker: bool = True,
) -> Tuple[Image.Image, int]:
    """Generate an image from text using FAL AI (async)."""
    if not FAL_AVAILABLE:
        raise ImportError("fal_client not installed")

    if not _check_fal_authentication():
        raise Exception("FAL_KEY not configured")

    logger.info(f"Generating image with FAL: {model}")

    def _generate():
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    logger.info(f"FAL progress: {log.get('message', '')}")

        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "output_format": "png",
        }

        if "nano-banana" in model:
            aspect_ratio, resolution = _get_fal_params_from_size(width, height)
            arguments.update({
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
            })
        else:
            arguments.update({
                "image_size": {"height": height, "width": width},
                "max_images": num_images,
                "enable_safety_checker": enable_safety_checker,
            })

        result = fal_client.subscribe(
            model,
            arguments=arguments,
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        if not result.get("images"):
            raise Exception("No images returned from FAL")

        return result["images"][0]["url"], result.get("seed", 0)

    for attempt in range(3):
        try:
            image_url, seed = await asyncio.to_thread(_generate)
            image = await download_image_from_url(image_url)
            logger.info(f"Successfully generated image with FAL (seed: {seed})")
            return image, seed
        except Exception as e:
            if attempt < 2:
                logger.warning(f"FAL generation attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(3)
            else:
                raise


async def generate_with_fal_edit(
    prompt: str,
    image_paths: List[Path],
    model: str = "fal-ai/nano-banana-pro/edit",
    width: int = 1920,
    height: int = 1080,
    num_images: int = 1,
) -> Tuple[Image.Image, int]:
    """Edit/transform images using FAL AI (async)."""
    if not FAL_AVAILABLE:
        raise ImportError("fal_client not installed")

    if not _check_fal_authentication():
        raise Exception("FAL_KEY not configured")

    logger.info(f"Editing image with FAL: {model}")

    # Upload images
    image_urls = []
    for image_path in image_paths:
        url = await upload_image_to_fal(image_path)
        image_urls.append(url)

    def _edit():
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    logger.info(f"FAL progress: {log.get('message', '')}")

        arguments = {
            "prompt": prompt,
            "num_images": num_images,
            "output_format": "png",
            "image_urls": image_urls,
        }

        if "nano-banana" in model:
            aspect_ratio, resolution = _get_fal_params_from_size(width, height)
            arguments.update({
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
            })
        else:
            arguments.update({
                "image_size": {"height": height, "width": width},
            })

        result = fal_client.subscribe(
            model,
            arguments=arguments,
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        if not result.get("images"):
            raise Exception("No images returned from FAL")

        return result["images"][0]["url"], result.get("seed", 0)

    for attempt in range(3):
        try:
            image_url, seed = await asyncio.to_thread(_edit)
            image = await download_image_from_url(image_url)
            logger.info(f"Successfully edited image with FAL (seed: {seed})")
            return image, seed
        except Exception as e:
            if attempt < 2:
                logger.warning(f"FAL edit attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(3)
            else:
                raise


def is_fal_available() -> bool:
    """Check if fal_client is installed and authenticated."""
    return FAL_AVAILABLE and _check_fal_authentication()
