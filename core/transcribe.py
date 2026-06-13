"""Transcription with Whisper large-v3 via faster-whisper."""

from __future__ import annotations

from functools import lru_cache

from faster_whisper import WhisperModel

from .config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL


def _resolve_device() -> tuple[str, str]:
    """Pick a sensible device / compute type combo."""
    device = WHISPER_DEVICE
    compute = WHISPER_COMPUTE_TYPE

    if device == "auto":
        try:
            import torch  # noqa: WPS433 (optional dependency)

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"

    return device, compute


@lru_cache(maxsize=1)
def _load_model() -> WhisperModel:
    device, compute = _resolve_device()
    return WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)


def transcribe(video_path: str, language: str = "fa", progress=None) -> dict:
    """Transcribe a media file.

    Returns a dict with the detected ``language`` and a list of ``segments``,
    each: {"start": float, "end": float, "text": str}.
    """
    model = _load_model()
    segments_iter, info = model.transcribe(
        video_path,
        language=language,
        vad_filter=True,
        beam_size=5,
    )

    total = info.duration or 0
    segments: list[dict] = []
    for seg in segments_iter:
        segments.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
        )
        if progress is not None and total:
            progress(min(seg.end / total, 1.0))

    if progress is not None:
        progress(1.0)

    return {"language": info.language, "segments": segments}


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS (or MM:SS when under an hour)."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def transcript_to_text(segments: list[dict]) -> str:
    """Render segments as a readable timestamped transcript."""
    lines = []
    for seg in segments:
        ts = format_timestamp(seg["start"])
        lines.append(f"[{ts}] {seg['text']}")
    return "\n".join(lines)
