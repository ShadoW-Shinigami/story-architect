"""
Async FFmpeg Command Builder for Agent 11
Constructs complex FFmpeg filter graphs for intelligent video editing.
"""

import asyncio
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger


class AsyncFFmpegBuilder:
    """
    Async builder for FFmpeg commands for intelligent video editing.

    Handles:
    - Trimming shots based on EDL
    - Audio offset for cinematic J/L cuts
    - Concatenation with transitions
    - Master timeline assembly
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FFmpegBuilder with configuration.

        Args:
            config: Agent 11 configuration dictionary
        """
        self.config = config
        self.ffmpeg_path = config.get("ffmpeg_path", "ffmpeg")
        self.ffprobe_path = config.get("ffprobe_path", "ffprobe")

        # Encoding settings
        self.output_codec = config.get("output_codec", "libx264")
        self.output_preset = config.get("output_preset", "medium")
        self.output_crf = config.get("output_crf", 23)
        self.audio_codec = config.get("audio_codec", "aac")
        self.audio_bitrate = config.get("audio_bitrate", "192k")

        # Output settings
        self.output_resolution = config.get("output_resolution", "1920x1080")
        self.output_fps = config.get("output_fps", 24)

    async def _get_output_params(self, reference_video: Path) -> Dict[str, Any]:
        """
        Get output encoding parameters, auto-detecting from reference video if needed.

        Args:
            reference_video: Path to reference video for auto-detection

        Returns:
            Dictionary with resolution, fps, etc.
        """
        params = {}

        # Get video info if we need to auto-detect anything
        if (self.output_resolution == "auto" or
            self.output_fps == "auto" or
            (isinstance(self.output_fps, str) and self.output_fps.lower() == "auto")):

            video_info = await self.get_video_info(reference_video)
            video_stream = next(
                (s for s in video_info.get('streams', []) if s.get('codec_type') == 'video'),
                None
            )

            if not video_stream:
                logger.warning("No video stream found, using default values")
                params['width'] = 1920
                params['height'] = 1080
                params['fps'] = 24
            else:
                # Resolution
                if self.output_resolution == "auto":
                    params['width'] = video_stream.get('width', 1920)
                    params['height'] = video_stream.get('height', 1080)
                    logger.info(f"Auto-detected resolution: {params['width']}x{params['height']}")
                else:
                    w, h = self.output_resolution.split('x')
                    params['width'] = int(w)
                    params['height'] = int(h)

                # FPS
                if isinstance(self.output_fps, str) and self.output_fps.lower() == "auto":
                    fps_str = video_stream.get('r_frame_rate', '24/1')
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        params['fps'] = int(float(num) / float(den))
                    else:
                        params['fps'] = int(float(fps_str))
                    logger.info(f"Auto-detected fps: {params['fps']}")
                else:
                    params['fps'] = self.output_fps
        else:
            # Use configured values
            if self.output_resolution != "auto":
                w, h = self.output_resolution.split('x')
                params['width'] = int(w)
                params['height'] = int(h)
            else:
                params['width'] = 1920
                params['height'] = 1080

            params['fps'] = self.output_fps if self.output_fps != "auto" else 24

        return params

    async def verify_ffmpeg(self):
        """Verify FFmpeg and FFprobe are installed (async)."""
        for tool, path in [("FFmpeg", self.ffmpeg_path), ("FFprobe", self.ffprobe_path)]:
            try:
                def _check():
                    return subprocess.run(
                        [path, "-version"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                result = await asyncio.to_thread(_check)
                if result.returncode == 0:
                    logger.info(f"{tool} verified at: {path}")
                else:
                    raise RuntimeError(f"{tool} failed verification")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError(
                    f"{tool} not found at '{path}'. "
                    f"Please install FFmpeg. Error: {e}"
                )

    async def get_video_duration(self, video_path: Path) -> float:
        """Get video duration using FFprobe (async)."""
        cmd = [
            self.ffprobe_path,
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

    async def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """Get comprehensive video information (async)."""
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,duration",
            "-show_entries", "format=duration,size,bit_rate",
            "-of", "json",
            str(video_path)
        ]

        def _run():
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        result = await asyncio.to_thread(_run)

        if result.returncode != 0:
            raise RuntimeError(f"FFprobe failed: {result.stderr}")

        return json.loads(result.stdout)

    async def edit_shot(
        self,
        video_path: Path,
        output_path: Path,
        trim_start: float = 0.0,
        trim_end: Optional[float] = None
    ) -> Path:
        """
        Edit a single shot: simple trim only (async).

        Args:
            video_path: Input video path
            output_path: Output video path
            trim_start: Seconds to trim from start
            trim_end: End timestamp (None = keep to end)

        Returns:
            Path to edited video
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Trimming shot: {video_path.name} ({trim_start}s to {trim_end if trim_end else 'end'}s)")

        duration = await self.get_video_duration(video_path)

        if trim_end is None:
            trim_end = duration

        # Validate
        trim_start = max(0.0, trim_start)
        trim_end = min(duration, trim_end)

        if trim_start >= trim_end:
            raise ValueError(f"Invalid trim values: start={trim_start}, end={trim_end}")

        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-ss", str(trim_start),
            "-to", str(trim_end),
            "-c:v", self.output_codec,
            "-preset", self.output_preset,
            "-crf", str(self.output_crf),
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
            "-y",
            str(output_path)
        ]

        await self._execute_ffmpeg(cmd, f"trimming shot {video_path.name}")

        logger.info(f"Shot trimmed: {output_path}")
        return output_path

    async def concatenate_simple(
        self,
        video_files: List[Path],
        output_path: Path,
        add_fade_transitions: bool = False,
        fade_duration: float = 1.0
    ) -> Path:
        """
        Simple concatenation of video files (async).

        Args:
            video_files: List of video file paths
            output_path: Output video file path
            add_fade_transitions: Whether to add fade transitions
            fade_duration: Fade duration in seconds

        Returns:
            Path to output video
        """
        if not video_files:
            raise ValueError("No video files provided for concatenation")

        logger.info(f"Concatenating {len(video_files)} videos...")

        # Create concat file
        concat_file = output_path.parent / f"{output_path.stem}_concat.txt"

        def _write_concat():
            with open(concat_file, 'w', encoding='utf-8') as f:
                for video in video_files:
                    if not video.exists():
                        raise FileNotFoundError(f"Video not found: {video}")
                    f.write(f"file '{str(video.absolute())}'\n")

        await asyncio.to_thread(_write_concat)

        try:
            if add_fade_transitions and len(video_files) > 1:
                output = await self._concatenate_with_fades(
                    video_files,
                    output_path,
                    fade_duration
                )
            else:
                cmd = [
                    self.ffmpeg_path,
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_file),
                    "-c:v", self.output_codec,
                    "-preset", self.output_preset,
                    "-crf", str(self.output_crf),
                    "-c:a", self.audio_codec,
                    "-b:a", self.audio_bitrate,
                    "-y",
                    str(output_path)
                ]

                await self._execute_ffmpeg(cmd, f"concatenation of {len(video_files)} videos")
                output = output_path

            # Clean up
            await asyncio.to_thread(concat_file.unlink)

            logger.info(f"Concatenation complete: {output}")
            return output

        except Exception as e:
            if concat_file.exists():
                await asyncio.to_thread(concat_file.unlink)
            raise

    async def _concatenate_with_fades(
        self,
        video_files: List[Path],
        output_path: Path,
        fade_duration: float
    ) -> Path:
        """Concatenate videos with crossfade transitions (async)."""
        if len(video_files) == 1:
            def _copy():
                shutil.copy2(video_files[0], output_path)

            await asyncio.to_thread(_copy)
            return output_path

        # Build filter_complex
        filter_parts = []
        input_args = []

        for video in video_files:
            input_args.extend(["-i", str(video)])

        # Build xfade chain
        for i in range(len(video_files) - 1):
            if i == 0:
                filter_parts.append(
                    f"[0:v][1:v]xfade=transition=fade:duration={fade_duration}:offset=0[v01]"
                )
            else:
                prev_label = f"v0{i}"
                curr_label = f"v0{i+1}"
                filter_parts.append(
                    f"[{prev_label}][{i+1}:v]xfade=transition=fade:duration={fade_duration}:offset=0[{curr_label}]"
                )

        # Audio concat
        audio_inputs = "".join([f"[{i}:a]" for i in range(len(video_files))])
        filter_parts.append(f"{audio_inputs}concat=n={len(video_files)}:v=0:a=1[aout]")

        filter_complex = ";".join(filter_parts)
        final_video_label = f"v0{len(video_files)-1}"

        cmd = [
            self.ffmpeg_path,
            *input_args,
            "-filter_complex", filter_complex,
            "-map", f"[{final_video_label}]",
            "-map", "[aout]",
            "-c:v", self.output_codec,
            "-preset", self.output_preset,
            "-crf", str(self.output_crf),
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
            "-y",
            str(output_path)
        ]

        await self._execute_ffmpeg(cmd, f"crossfade concatenation of {len(video_files)} videos")
        return output_path

    async def edit_scene_with_edl(
        self,
        edl_shots: List[Dict[str, Any]],
        session_dir: Path,
        output_path: Path
    ) -> Path:
        """
        Edit a scene using an Edit Decision List (async).

        Args:
            edl_shots: List of shot edit instructions from EDL
            session_dir: Session directory for resolving paths
            output_path: Output scene video path

        Returns:
            Path to edited scene video
        """
        logger.info(f"Editing scene with {len(edl_shots)} shots...")

        temp_dir = output_path.parent / "temp_edited_shots"

        def _mkdir():
            temp_dir.mkdir(exist_ok=True)

        await asyncio.to_thread(_mkdir)

        edited_shots = []

        try:
            # Step 1: Trim each shot
            for i, shot_edit in enumerate(edl_shots):
                shot_id = shot_edit.get("shot_id")
                video_rel_path = shot_edit.get("video_path")
                video_path = session_dir / video_rel_path

                trim_start = shot_edit.get("trim_start", 0.0)
                trim_end = shot_edit.get("trim_end")

                edited_shot_path = temp_dir / f"edited_{i:03d}_{shot_id}.mp4"

                await self.edit_shot(
                    video_path,
                    edited_shot_path,
                    trim_start,
                    trim_end
                )

                edited_shots.append(edited_shot_path)

            # Step 2: Build transitions list
            transitions = []
            for i in range(len(edl_shots) - 1):
                next_shot = edl_shots[i + 1]
                edit_type = next_shot.get("edit_type", "hard_cut")
                audio_offset = next_shot.get("audio_start_offset", 0.0)

                if edit_type == "j_cut" and audio_offset < 0:
                    transitions.append({
                        "type": "j_cut",
                        "audio_advance": abs(audio_offset)
                    })
                elif edit_type == "l_cut" and audio_offset > 0:
                    transitions.append({
                        "type": "l_cut",
                        "audio_extend": abs(audio_offset)
                    })
                else:
                    transitions.append({"type": "hard_cut"})

            # Step 3: Check for J/L cuts
            has_jl_cuts = any(t.get("type") in ["j_cut", "l_cut"] for t in transitions)

            if has_jl_cuts:
                await self._concatenate_with_timeline_audio(edited_shots, transitions, output_path)
            else:
                await self.concatenate_simple(edited_shots, output_path)

            # Clean up temp files
            for shot in edited_shots:
                if shot.exists():
                    await asyncio.to_thread(shot.unlink)

            def _rmdir():
                if temp_dir.exists():
                    temp_dir.rmdir()

            await asyncio.to_thread(_rmdir)

            logger.info(f"Scene editing complete: {output_path}")
            return output_path

        except Exception as e:
            # Clean up on error
            for shot in edited_shots:
                if shot.exists():
                    await asyncio.to_thread(shot.unlink)
            if temp_dir.exists():
                await asyncio.to_thread(temp_dir.rmdir)
            raise

    async def _concatenate_with_timeline_audio(
        self,
        video_files: List[Path],
        transitions: List[Dict[str, Any]],
        output_path: Path
    ) -> Path:
        """Concatenate with proper J/L cut support (async)."""
        if len(video_files) == 1:
            def _copy():
                shutil.copy2(video_files[0], output_path)

            await asyncio.to_thread(_copy)
            return output_path

        # Get durations
        durations = []
        for vf in video_files:
            dur = await self.get_video_duration(vf)
            durations.append(dur)

        # Calculate timeline positions
        video_positions = [0.0]
        audio_positions = [0.0]

        current_video_time = 0.0

        for i, duration in enumerate(durations):
            if i < len(transitions):
                trans = transitions[i]
                trans_type = trans.get("type", "hard_cut")

                if trans_type == "j_cut":
                    audio_advance = trans.get("audio_advance", 0.0)
                    next_video_pos = current_video_time + duration
                    next_audio_pos = next_video_pos - audio_advance
                elif trans_type == "l_cut":
                    audio_extend = trans.get("audio_extend", 0.0)
                    next_video_pos = current_video_time + duration
                    next_audio_pos = next_video_pos - audio_extend
                else:
                    next_video_pos = current_video_time + duration
                    next_audio_pos = next_video_pos

                video_positions.append(next_video_pos)
                audio_positions.append(next_audio_pos)
                current_video_time = next_video_pos

        # Build filter
        input_args = []
        for vf in video_files:
            input_args.extend(["-i", str(vf)])

        filter_parts = []

        # Video concat
        video_concat = "".join([f"[{i}:v]" for i in range(len(video_files))])
        video_concat += f"concat=n={len(video_files)}:v=1:a=0[vout]"
        filter_parts.append(video_concat)

        # Position and mix audio
        audio_labels = []
        for i in range(len(video_files)):
            audio_delay_ms = int(max(0, audio_positions[i]) * 1000)

            if audio_delay_ms > 0:
                filter_parts.append(f"[{i}:a]adelay={audio_delay_ms}|{audio_delay_ms}[a{i}]")
                audio_labels.append(f"[a{i}]")
            else:
                audio_labels.append(f"[{i}:a]")

        audio_mix = "".join(audio_labels)
        audio_mix += f"amix=inputs={len(video_files)}:duration=longest:dropout_transition=0[amixed]"
        filter_parts.append(audio_mix)

        # Build audio enhancement chain
        audio_enhancements = []
        current_label = "[amixed]"

        # [A] Dynamic Range Compression (make quiet parts louder, loud parts quieter)
        if self.config.get("audio_compression", True):
            compression_ratio = self.config.get("compression_ratio", 3)
            audio_enhancements.append(
                f"{current_label}acompressor=threshold=-18dB:ratio={compression_ratio}:attack=20:release=250[acomp]"
            )
            current_label = "[acomp]"
            logger.debug("Audio enhancement: Dynamic range compression enabled")

        # [C] Dialogue Enhancement EQ (boost voice frequencies)
        if self.config.get("dialogue_enhancement", True):
            eq_boost = self.config.get("eq_voice_boost_db", 3)
            audio_enhancements.append(
                f"{current_label}equalizer=f=1000:t=h:width=1000:g={eq_boost},highpass=f=100,lowpass=f=8000[aeq]"
            )
            current_label = "[aeq]"
            logger.debug(f"Audio enhancement: Dialogue EQ with +{eq_boost}dB boost")

        # [A] Loudness Normalization (streaming standard)
        if self.config.get("normalize_loudness", True):
            target_loudness = self.config.get("target_loudness", -16)
            audio_enhancements.append(
                f"{current_label}loudnorm=I={target_loudness}:TP=-1.5:LRA=11[aout]"
            )
            current_label = "[aout]"
            logger.debug(f"Audio enhancement: Loudness normalization to {target_loudness} LUFS")
        else:
            # Rename final label to aout if no normalization
            if current_label != "[amixed]":
                audio_enhancements.append(f"{current_label}acopy[aout]")
            else:
                # No enhancements at all, rename amixed to aout
                filter_parts[-1] = filter_parts[-1].replace("[amixed]", "[aout]")

        # Add all audio enhancements to filter
        if audio_enhancements:
            filter_parts.extend(audio_enhancements)

        filter_complex = ";".join(filter_parts)

        cmd = [
            self.ffmpeg_path,
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", self.output_codec,
            "-preset", self.output_preset,
            "-crf", str(self.output_crf),
            "-c:a", self.audio_codec,
            "-b:a", self.audio_bitrate,
            "-y",
            str(output_path)
        ]

        await self._execute_ffmpeg(cmd, "concatenation with audio mixing")
        return output_path

    async def _execute_ffmpeg(self, cmd: List[str], operation_desc: str):
        """Execute FFmpeg command with error handling (async)."""
        logger.debug(f"Executing FFmpeg: {' '.join(cmd)}")

        def _run():
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

        result = await asyncio.to_thread(_run)

        if result.returncode != 0:
            logger.error(f"FFmpeg failed during {operation_desc}")
            logger.error(f"Error: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")

        logger.debug(f"FFmpeg {operation_desc} completed successfully")
