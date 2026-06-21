"""Transcription with faster-whisper.

This module exposes a single, stable function :func:`transcribe` and a small
data class :class:`Transcript` for downstream consumers.

Why a class?
------------
Down-stream code (segmentation, subtitle generation, highlight scoring) all
needs the same shape: a list of segments with optional word-level timestamps.
Wrapping the result in a dataclass:

* makes the JSON-on-disk schema explicit,
* makes the field names stable (no dict-key typos),
* gives us a single place to add validation/normalisation later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:  # noqa: BLE001
    # WhisperModel is only needed for :func:`transcribe`. The package still
    # imports cleanly so the CLI, the segmentation logic and the tests can
    # run without the heavy speech-recognition dependency installed.
    WhisperModel = None  # type: ignore

from .config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Word:
    """A single word with start/end timestamps in seconds."""

    word: str
    start: float
    end: float

    @classmethod
    def from_raw(cls, raw: dict) -> "Word":
        return cls(
            word=str(raw.get("word", "")).strip(),
            start=float(raw.get("start", 0.0)),
            end=float(raw.get("end", 0.0)),
        )


@dataclass
class Segment:
    """A transcript segment (one or more sentences) with optional word info."""

    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: dict, idx: int) -> "Segment":
        words = [Word.from_raw(w) for w in (raw.get("words") or [])]
        return cls(
            id=idx,
            start=round(float(raw.get("start", 0.0)), 2),
            end=round(float(raw.get("end", 0.0)), 2),
            text=str(raw.get("text", "")).strip(),
            words=words,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [asdict(w) for w in self.words],
        }


@dataclass
class Transcript:
    """The full transcript of a single media file."""

    video_id: str
    language: str
    duration: float
    segments: list[Segment]
    source: str = ""  # original file path, if known

    # ---- (de)serialisation ---- #

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "language": self.language,
            "duration": round(float(self.duration or 0.0), 2),
            "source": self.source,
            "segments": [s.to_dict() for s in self.segments],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str | Path, indent: int = 2) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(indent=indent), encoding="utf-8")
        return p

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        return cls(
            video_id=str(data.get("video_id", "")),
            language=str(data.get("language", "fa")),
            duration=float(data.get("duration", 0.0) or 0.0),
            source=str(data.get("source", "")),
            segments=[
                Segment.from_raw(s, idx)
                for idx, s in enumerate(data.get("segments", []), start=1)
            ],
        )

    @classmethod
    def load(cls, path: str | Path) -> "Transcript":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---- convenience for callers that want plain dicts ---- #

    def to_segments_dicts(self) -> list[dict]:
        """Return segments as plain dicts (back-compat for the old API)."""
        return [s.to_dict() for s in self.segments]


# --------------------------------------------------------------------------- #
# Whisper wrapper
# --------------------------------------------------------------------------- #


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


def _video_id_for(path: str | Path) -> str:
    """Stable, filesystem-friendly id for a video path."""
    import hashlib

    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def transcribe(
    video_path: str,
    language: str = "fa",
    progress: Optional[Callable[[float], None]] = None,
    word_timestamps: bool = True,
) -> Transcript:
    """Transcribe a media file with faster-whisper.

    Returns a :class:`Transcript`. Word-level timestamps are requested by
    default; if the backend cannot produce them (e.g. model not built with
    the alignment feature) we silently fall back to segment-level timestamps
    so the rest of the pipeline still works.
    """
    model = _load_model()
    segments_iter, info = model.transcribe(
        str(video_path),
        language=language,
        vad_filter=True,
        beam_size=5,
        word_timestamps=word_timestamps,
    )

    total = info.duration or 0
    segments: list[Segment] = []
    for idx, seg in enumerate(segments_iter, start=1):
        words: list[Word] = []
        if getattr(seg, "words", None):
            words = [
                Word(w.word.strip(), float(w.start), float(w.end))
                for w in seg.words
                if getattr(w, "word", None) is not None
            ]
        segments.append(
            Segment(
                id=idx,
                start=round(float(seg.start), 2),
                end=round(float(seg.end), 2),
                text=str(seg.text).strip(),
                words=words,
            )
        )
        if progress is not None and total:
            progress(min(seg.end / total, 1.0))

    if progress is not None:
        progress(1.0)

    return Transcript(
        video_id=_video_id_for(video_path),
        language=info.language or language,
        duration=float(info.duration or 0.0),
        segments=segments,
        source=str(video_path),
    )


# --------------------------------------------------------------------------- #
# Backwards-compat helpers (used by the existing app.py)
# --------------------------------------------------------------------------- #


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS (or MM:SS when under an hour)."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def transcript_to_text(segments: Iterable[dict]) -> str:
    """Render segments as a readable timestamped transcript."""
    lines = []
    for seg in segments:
        ts = format_timestamp(seg["start"])
        lines.append(f"[{ts}] {seg['text']}")
    return "\n".join(lines)
