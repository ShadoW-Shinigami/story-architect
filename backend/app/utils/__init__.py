"""
Utility module exports for Story Architect backend.
"""

from app.utils.image_utils import (
    extract_character_combinations,
    create_character_grid,
    image_to_base64,
    base64_to_image,
    save_image_with_metadata,
    verify_image_quality,
    slugify,
    get_aspect_ratio_from_image,
    optimize_image_for_base64,
)

from app.utils.fal_helper import (
    generate_with_fal_text_to_image,
    generate_with_fal_edit,
    upload_image_to_fal,
    download_image_from_url,
    is_fal_available,
    resolve_dimensions_from_config,
)

from app.utils.vertex_veo_helper import (
    AsyncVertexVeoClient,
    is_vertex_available,
)

from app.utils.audio_analyzer import (
    AsyncAudioAnalyzer,
    is_whisperx_available,
)

from app.utils.ffmpeg_builder import (
    AsyncFFmpegBuilder,
)

__all__ = [
    # Image utilities
    "extract_character_combinations",
    "create_character_grid",
    "image_to_base64",
    "base64_to_image",
    "save_image_with_metadata",
    "verify_image_quality",
    "slugify",
    "get_aspect_ratio_from_image",
    "optimize_image_for_base64",
    # FAL helpers
    "generate_with_fal_text_to_image",
    "generate_with_fal_edit",
    "upload_image_to_fal",
    "download_image_from_url",
    "is_fal_available",
    "resolve_dimensions_from_config",
    # Vertex AI Veo
    "AsyncVertexVeoClient",
    "is_vertex_available",
    # Audio analysis
    "AsyncAudioAnalyzer",
    "is_whisperx_available",
    # FFmpeg
    "AsyncFFmpegBuilder",
]
