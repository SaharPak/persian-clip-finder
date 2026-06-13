"""Clip cutting, 9:16 vertical reframing, and subtitle burn-in with FFmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .config import CLIP_DIR, SUBTITLE_DIR
from .subtitles import build_ass

VERTICAL_W, VERTICAL_H = 1080, 1920


def _has_ass_filter(ffmpeg: str) -> bool:
    """True if this ffmpeg build was compiled with libass (ass/subtitles)."""
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
        ).stdout
        return any(line.split()[1:2] == ["ass"] for line in out.splitlines())
    except Exception:
        return False


@lru_cache(maxsize=1)
def _ensure_ffmpeg() -> str:
    """Return an ffmpeg that supports libass (needed for subtitle burn-in).

    Prefers the system ffmpeg when it has libass; otherwise falls back to the
    bundled static build from the ``static-ffmpeg`` package.
    """
    system = shutil.which("ffmpeg")
    if system and _has_ass_filter(system):
        return system

    try:
        from static_ffmpeg import run

        static_path, _ = run.get_or_fetch_platform_executables_else_raise()
        if _has_ass_filter(static_path):
            return static_path
    except Exception:
        pass

    if system:
        return system
    raise RuntimeError("FFmpeg not found. Install it or `pip install static-ffmpeg`.")


@lru_cache(maxsize=1)
def _ensure_ffprobe() -> str | None:
    probe = shutil.which("ffprobe")
    if probe:
        return probe
    try:
        from static_ffmpeg import run

        _, static_probe = run.get_or_fetch_platform_executables_else_raise()
        return static_probe
    except Exception:
        return None


def _probe_dimensions(source: str) -> tuple[int, int]:
    """Return (width, height) of the source video, defaulting to 1280x720."""
    probe = _ensure_ffprobe()
    if not probe:
        return 1280, 720
    try:
        out = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                source,
            ],
            capture_output=True,
            text=True,
        ).stdout
        stream = json.loads(out)["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception:
        return 1280, 720


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (code {proc.returncode}):\n{proc.stderr[-2000:]}"
        )


def _escape_for_filter(path: str) -> str:
    """Escape a path for use inside an ffmpeg filtergraph (ass=...)."""
    p = str(Path(path).resolve())
    p = p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return p


def generate_clip(
    source: str,
    start: float,
    end: float,
    out_name: str,
    segments: list[dict] | None = None,
    vertical: bool = False,
    font: str = "Geeza Pro",
) -> str:
    """Cut [start, end] from source.

    If ``segments`` (clip-relative) are given, burn them as RTL Persian
    subtitles. If ``vertical`` is True, reframe to 1080x1920 (9:16) for
    Shorts/Reels. Returns the path to the generated MP4.
    """
    ffmpeg = _ensure_ffmpeg()
    out_path = CLIP_DIR / f"{out_name}.mp4"
    duration = max(0.1, end - start)

    if vertical:
        out_w, out_h = VERTICAL_W, VERTICAL_H
    else:
        out_w, out_h = _probe_dimensions(source)

    filters: list[str] = []
    if vertical:
        filters.append(
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase"
        )
        filters.append(f"crop={out_w}:{out_h}")

    if segments:
        ass_path = build_ass(
            segments,
            SUBTITLE_DIR / f"{out_name}.ass",
            width=out_w,
            height=out_h,
            font=font,
        )
        filters.append(f"ass={_escape_for_filter(ass_path)}")

    cmd = [ffmpeg, "-y", "-ss", f"{start}", "-i", source, "-t", f"{duration}"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    _run(cmd)
    return str(out_path)
