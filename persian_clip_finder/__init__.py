"""Persian Clip Finder — turn long Persian videos into ready-to-post short clips.

Public entry points re-exported here:

* :func:`transcribe`     – run faster-whisper on a video file
* :func:`find_highlights` – ask an LLM to score and rank candidate moments
* :func:`generate_clip`  – cut a clip with FFmpeg, smart-crop, burn subtitles
* :func:`process_video`  – the end-to-end "Tech Immigrants" workflow
* :func:`run_cli`        – ``python -m persian_clip_finder`` entry point
"""

from __future__ import annotations

# Re-exports are best-effort: heavy optional deps (faster-whisper, yt-dlp,
# opencv, …) are imported lazily inside each module, so importing the
# package should never fail in a minimal environment.
try:
    from .clip import generate_clip
except Exception:  # noqa: BLE001
    generate_clip = None  # type: ignore
try:
    from .highlights import Highlight, find_highlights
except Exception:  # noqa: BLE001
    Highlight = None  # type: ignore
    find_highlights = None  # type: ignore
try:
    from .transcribe import Transcript, transcribe
except Exception:  # noqa: BLE001
    Transcript = None  # type: ignore
    transcribe = None  # type: ignore
try:
    from .workflow import WorkflowConfig, process_video
except Exception:  # noqa: BLE001
    WorkflowConfig = None  # type: ignore
    process_video = None  # type: ignore

__version__ = "0.2.0"

__all__ = [
    "Transcript",
    "Highlight",
    "WorkflowConfig",
    "transcribe",
    "find_highlights",
    "generate_clip",
    "process_video",
    "run_cli",
]


def run_cli(argv: list[str] | None = None) -> int:
    """Programmatic entry point: ``run_cli(['process', '--input', 'x.mp4'])``."""
    from .cli import main

    return main(argv)
