"""
Async Audio Analysis Module for Agent 11
Uses WhisperX to extract word-level timestamps and silence detection
from video files for intelligent video editing.
"""

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    logger.warning("WhisperX not installed. Audio analysis will be limited.")


class AsyncAudioAnalyzer:
    """
    Async audio analyzer using WhisperX.

    Extracts:
    - Word-level timestamps
    - Speech start/end times
    - Silence regions
    - Dialogue content
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AudioAnalyzer with WhisperX configuration.

        Args:
            config: Agent 11 configuration dictionary
        """
        self.config = config
        self.model_name = config.get("whisperx_model", "base.en")
        self.device = config.get("whisperx_device", "cpu")
        self.batch_size = config.get("whisperx_batch_size", 8)
        self.compute_type = config.get("whisperx_compute_type", "int8")

        self.model = None

        if WHISPERX_AVAILABLE:
            logger.info(f"WhisperX available: {self.model_name} on {self.device}")
        else:
            logger.warning("WhisperX not available - audio analysis will use fallback")

    async def _ensure_model_loaded(self):
        """Lazy-load WhisperX model."""
        if self.model is not None:
            return

        if not WHISPERX_AVAILABLE:
            return

        def _load():
            return whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type
            )

        logger.info(f"Loading WhisperX model: {self.model_name}")
        self.model = await asyncio.to_thread(_load)
        logger.info("WhisperX model loaded successfully")

    async def analyze_video_batch(
        self,
        video_files: List[Dict[str, Any]],
        session_dir: Path
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple video files for audio content (async).

        Args:
            video_files: List of video metadata dicts with 'shot_id' and 'video_path'
            session_dir: Session directory for resolving relative paths

        Returns:
            Dictionary mapping shot_id to audio metadata
        """
        await self._ensure_model_loaded()

        results = {}

        for i, video_info in enumerate(video_files, 1):
            shot_id = video_info.get("shot_id")
            video_path = session_dir / video_info.get("video_path")

            logger.info(f"[{i}/{len(video_files)}] Analyzing audio: {shot_id}")

            try:
                metadata = await self.analyze_video(video_path, shot_id)
                results[shot_id] = metadata
            except Exception as e:
                logger.error(f"Failed to analyze {shot_id}: {e}")
                results[shot_id] = await self._create_fallback_metadata(video_path, shot_id)

        return results

    async def analyze_video(self, video_path: Path, shot_id: str) -> Dict[str, Any]:
        """
        Analyze a single video file for audio content (async).

        Args:
            video_path: Path to video file
            shot_id: Shot identifier

        Returns:
            Audio metadata dictionary
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get video duration
        duration = await self._get_video_duration(video_path)

        # Check for audio track
        has_audio = await self._check_audio_track(video_path)

        if not has_audio:
            logger.warning(f"{shot_id}: No audio track detected")
            return {
                "shot_id": shot_id,
                "video_path": str(video_path),
                "duration": duration,
                "has_audio": False,
                "dialogue": None
            }

        # If WhisperX not available, return fallback
        if not WHISPERX_AVAILABLE or self.model is None:
            logger.warning(f"{shot_id}: WhisperX not available, using fallback")
            return await self._create_fallback_metadata(video_path, shot_id)

        # Extract audio to temporary WAV file
        audio_path = video_path.parent / f"{video_path.stem}_temp.wav"
        await self._extract_audio(video_path, audio_path)

        try:
            # Run WhisperX transcription in thread pool
            def _transcribe():
                audio = whisperx.load_audio(str(audio_path))
                result = self.model.transcribe(audio, batch_size=self.batch_size)

                # Perform word-level alignment
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"],
                    device=self.device
                )
                aligned = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False
                )
                return aligned

            result = await asyncio.to_thread(_transcribe)

            # Extract dialogue metadata
            dialogue_metadata = self._extract_dialogue_metadata(result, duration)

            # Clean up temp file
            await asyncio.to_thread(audio_path.unlink)

            return {
                "shot_id": shot_id,
                "video_path": str(video_path),
                "duration": duration,
                "has_audio": True,
                "dialogue": dialogue_metadata
            }

        except Exception as e:
            logger.error(f"WhisperX analysis failed for {shot_id}: {e}")
            if audio_path.exists():
                await asyncio.to_thread(audio_path.unlink)
            return await self._create_fallback_metadata(video_path, shot_id)

    def _extract_dialogue_metadata(
        self,
        whisperx_result: Dict[str, Any],
        video_duration: float
    ) -> Dict[str, Any]:
        """Extract structured dialogue metadata from WhisperX result."""
        segments = whisperx_result.get("segments", [])

        if not segments:
            return {
                "has_speech": False,
                "speech_start": None,
                "speech_end": None,
                "silence_before": video_duration,
                "silence_after": 0.0,
                "words": [],
                "transcript": ""
            }

        # Extract all words
        words = []
        for segment in segments:
            for word_info in segment.get("words", []):
                words.append({
                    "word": word_info.get("word", "").strip(),
                    "start": word_info.get("start", 0.0),
                    "end": word_info.get("end", 0.0)
                })

        if not words:
            return {
                "has_speech": False,
                "speech_start": None,
                "speech_end": None,
                "silence_before": video_duration,
                "silence_after": 0.0,
                "words": [],
                "transcript": ""
            }

        # Calculate speech boundaries
        speech_start = words[0]["start"]
        speech_end = words[-1]["end"]

        # Calculate silence regions
        silence_before = speech_start
        silence_after = max(0.0, video_duration - speech_end)

        # Build transcript
        transcript = " ".join([w["word"] for w in words])

        return {
            "has_speech": True,
            "speech_start": round(speech_start, 2),
            "speech_end": round(speech_end, 2),
            "silence_before": round(silence_before, 2),
            "silence_after": round(silence_after, 2),
            "words": words,
            "transcript": transcript
        }

    async def _extract_audio(self, video_path: Path, audio_path: Path):
        """Extract audio track from video using FFmpeg (async)."""
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(audio_path)
        ]

        def _run():
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

        result = await asyncio.to_thread(_run)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")

    async def _get_video_duration(self, video_path: Path) -> float:
        """Get video duration using FFprobe (async)."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]

        def _run():
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        result = await asyncio.to_thread(_run)

        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")

        return float(result.stdout.strip())

    async def _check_audio_track(self, video_path: Path) -> bool:
        """Check if video file has an audio track (async)."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]

        def _run():
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        result = await asyncio.to_thread(_run)
        return "audio" in result.stdout.lower()

    async def _create_fallback_metadata(
        self,
        video_path: Path,
        shot_id: str
    ) -> Dict[str, Any]:
        """Create fallback metadata when WhisperX analysis fails."""
        try:
            duration = await self._get_video_duration(video_path)
        except:
            duration = 5.0

        return {
            "shot_id": shot_id,
            "video_path": str(video_path),
            "duration": duration,
            "has_audio": False,
            "dialogue": None
        }

    async def save_metadata(self, metadata: Dict[str, Any], output_path: Path):
        """Save audio metadata to JSON file (async)."""
        def _save():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        await asyncio.to_thread(_save)
        logger.info(f"Audio metadata saved to: {output_path}")


def is_whisperx_available() -> bool:
    """Check if WhisperX is installed."""
    return WHISPERX_AVAILABLE
