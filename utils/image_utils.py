"""
Image utilities for processing images for video generation.
Includes aspect ratio detection and image optimization for base64 encoding.
"""

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
from loguru import logger


def detect_aspect_ratio(width: int, height: int) -> str:
    """
    Detect the closest standard aspect ratio for video generation.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Closest aspect ratio string: "16:9", "9:16", or "1:1"
    """
    ratio = width / height
    
    # Define aspect ratios and their numeric values
    aspect_ratios = {
        "16:9": 16/9,   # ~1.778 (landscape)
        "1:1": 1.0,     # 1.0 (square)
        "9:16": 9/16    # ~0.5625 (portrait)
    }
    
    # Find closest aspect ratio
    closest = min(aspect_ratios.items(), key=lambda x: abs(x[1] - ratio))
    
    logger.debug(f"Image ratio {ratio:.3f} ({width}x{height}) -> closest: {closest[0]} ({closest[1]:.3f})")
    return closest[0]


def get_aspect_ratio_from_image(image_path: Path) -> str:
    """
    Get aspect ratio from image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Aspect ratio string: "16:9", "9:16", or "1:1"
    """
    with Image.open(image_path) as img:
        return detect_aspect_ratio(img.width, img.height)


def optimize_image_for_base64(
    image_path: Path,
    max_size_mb: float = 7.0,
    output_format: str = "JPEG",
    quality: int = 85
) -> Tuple[str, int]:
    """
    Optimize image for base64 encoding by compressing (reducing quality only).
    NEVER reduces resolution - only adjusts compression quality.
    
    Args:
        image_path: Path to original image
        max_size_mb: Maximum size in megabytes (default 7MB)
        output_format: Output format ("JPEG" or "PNG")
        quality: Initial JPEG quality (1-100, default 85)
        
    Returns:
        Tuple of (base64_string, final_size_bytes)
    """
    max_size_bytes = int(max_size_mb * 1024 * 1024)
    
    # Open image
    with Image.open(image_path) as img:
        original_dimensions = f"{img.width}x{img.height}"
        
        # Convert RGBA to RGB if saving as JPEG
        if output_format.upper() == "JPEG" and img.mode in ("RGBA", "LA", "P"):
            # Create white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = background
        
        # Save to bytes with initial quality
        buffer = BytesIO()
        img.save(buffer, format=output_format, quality=quality, optimize=True)
        img_bytes = buffer.getvalue()
        original_size = len(img_bytes)
        
        logger.debug(f"Original image: {original_dimensions}, size: {original_size / (1024*1024):.2f} MB")
        
        # If already under limit, return as-is
        if original_size <= max_size_bytes:
            b64_string = base64.b64encode(img_bytes).decode('utf-8')
            logger.debug(f"Image within size limit, no optimization needed")
            return b64_string, original_size
        
        # Need to compress - reduce quality only (never resize)
        logger.info(f"Image exceeds {max_size_mb}MB limit, compressing (quality reduction only)...")
        
        current_quality = quality - 5
        min_quality = 30  # Lower minimum for more compression range
        
        while current_quality >= min_quality:
            buffer = BytesIO()
            img.save(buffer, format=output_format, quality=current_quality, optimize=True)
            img_bytes = buffer.getvalue()
            
            if len(img_bytes) <= max_size_bytes:
                b64_string = base64.b64encode(img_bytes).decode('utf-8')
                logger.info(
                    f"Compressed image: {original_size / (1024*1024):.2f} MB -> "
                    f"{len(img_bytes) / (1024*1024):.2f} MB (quality={current_quality}, resolution={original_dimensions})"
                )
                return b64_string, len(img_bytes)
            
            current_quality -= 5
        
        # If we get here, even minimum quality exceeds limit
        # Return the minimum quality version (never resize)
        buffer = BytesIO()
        img.save(buffer, format=output_format, quality=min_quality, optimize=True)
        img_bytes = buffer.getvalue()
        b64_string = base64.b64encode(img_bytes).decode('utf-8')
        
        logger.warning(
            f"Could not compress image below {max_size_mb}MB limit. "
            f"Using minimum quality ({min_quality}). "
            f"Final size: {len(img_bytes) / (1024*1024):.2f} MB, resolution: {original_dimensions}"
        )
        return b64_string, len(img_bytes)


def image_to_base64(image_path: Path, format: str = "JPEG") -> str:
    """
    Convert image to base64 string without size optimization.
    
    Args:
        image_path: Path to image file
        format: Output format ("JPEG" or "PNG")
        
    Returns:
        Base64 encoded string
    """
    with Image.open(image_path) as img:
        # Convert to RGB if needed for JPEG
        if format.upper() == "JPEG" and img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = background
        
        buffer = BytesIO()
        img.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

