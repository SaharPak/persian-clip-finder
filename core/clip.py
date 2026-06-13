"""Clip cutting and subtitle burn-in with FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .config import CLIP_DIR


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
        # Usable for cutting, but subtitle burn-in will fail.
        return system
    raise RuntimeError("FFmpeg not found. Install it or `pip install static-ffmpeg`.")


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
    ass_path: str | None = None,
) -> str:
    """Cut [start, end] from source. Optionally burn an ASS subtitle file.

    Returns the path to the generated MP4.
    """
    ffmpeg = _ensure_ffmpeg()
    out_path = CLIP_DIR / f"{out_name}.mp4"
    duration = max(0.1, end - start)

    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start}",
        "-i",
        source,
        "-t",
        f"{duration}",
    ]

    if ass_path:
        vf = f"ass={_escape_for_filter(ass_path)}"
        cmd += ["-vf", vf]

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
