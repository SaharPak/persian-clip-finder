"""End-to-end "Tech Immigrants Repurposing" workflow.

The workflow glues together transcription, segmentation, highlight detection,
subtitle generation, multi-aspect video export, and per-clip social-post
metadata. The output layout matches the brief:

    outputs/
      <video-title-date>/
        transcript.json
        highlights.json
        clips/
          clip_001/
            clip_001_reels.mp4
            clip_001_square.mp4
            clip_001_youtube.mp4
            clip_001_reels.ass
            clip_001_transcript.txt
            clip_001_metadata.json
            clip_001_social_posts.md
          clip_002/
            ...

The single public entry point is :func:`process_video`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

from .clip import (
    export_clip_batch,
    resolve_aspects,
)
from .highlights import (
    Highlight,
    enrich_social_posts,
    find_highlights,
)
from .subtitles import SubtitleStyle
from .transcribe import Transcript, transcribe


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class WorkflowConfig:
    """All knobs the user might want to tweak in one place."""

    language: str = "fa"
    provider: Optional[str] = None  # claude | gpt | offline | None=auto
    top_k: int = 8
    aspects: tuple[str, ...] = ("reels", "square", "youtube")
    subtitle_mode: str = "phrase_highlight"   # static | word | phrase_highlight
    subtitle_position: str = "lower"          # bottom | lower | middle | top
    font: str = "Geeza Pro"
    crop_mode: str = "auto"                   # auto | single | split2 | grid3 | fit
    min_duration: float = 20.0
    target_min: float = 45.0
    target_max: float = 75.0
    max_duration: float = 120.0
    word_timestamps: bool = True
    burn_subtitles: bool = True
    save_transcript_json: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


# Slug-friendly: keep ASCII letters/digits/underscore/hyphen and any
# non-ASCII letter (so Persian passes through), but normalise whitespace
# and common punctuation to a single dash.
_SLUG_RE = re.compile(r"[\s\\/\.\,!?\(\)\[\]\{\}\:\;\|\"\'<>@#$%^&*+]+")


def _safe_slug(text: str, fallback: str = "video") -> str:
    s = (text or "").strip()
    s = _SLUG_RE.sub("-", s)
    s = s.strip("-").lower()
    return s or fallback


def _now_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def _build_subtitle_style(cfg: WorkflowConfig) -> SubtitleStyle:
    return SubtitleStyle(
        mode=cfg.subtitle_mode,
        font=cfg.font,
        subtitle_position=cfg.subtitle_position,
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Per-clip artifacts
# --------------------------------------------------------------------------- #


def write_clip_transcript(
    clip_dir: Path, full_segments: list[dict], start: float, end: float
) -> Path:
    """Write a plain-text transcript slice for a single clip."""
    lines = []
    for seg in full_segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        s = max(seg["start"], start) - start
        e = min(seg["end"], end) - start
        ts = f"[{_fmt(s)}-{_fmt(e)}]"
        lines.append(f"{ts} {seg['text'].strip()}")
    out = clip_dir / "clip_transcript.txt"
    _write_text(out, "\n".join(lines))
    return out


def _fmt(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def write_clip_metadata(clip_dir: Path, h: Highlight, *, guest: str, topic: str, video_title: str) -> Path:
    payload = {
        **h.to_dict(),
        "guest": guest,
        "topic": topic,
        "video_title": video_title,
    }
    out = clip_dir / "clip_metadata.json"
    _write_text(out, json.dumps(payload, ensure_ascii=False, indent=2))
    return out


def write_clip_social_posts(clip_dir: Path, h: Highlight) -> Path:
    md = [
        f"# {h.title or 'کلیپ'}",
        "",
        f"_شروع: {_fmt(h.start)} · پایان: {_fmt(h.end)} · مدت: {_fmt(h.duration)}_",
        "",
        "## یوتیوب Shorts",
        h.title or "",
        "",
        "## اینستاگرام / LinkedIn",
        h.caption_instagram or h.caption_linkedin or "",
        "",
        "## لینکدین",
        h.caption_linkedin or "",
        "",
        "## X (توییتر)",
        "```",
        h.post_x or "",
        "```",
        "",
        "## تلگرام",
        h.post_telegram or "",
        "",
        "## هشتگ‌ها",
        " ".join("#" + t for t in h.hashtags) if h.hashtags else "",
        "",
        "## کامنت پین‌شده",
        h.pinned_comment or "",
        "",
        "## متن تامبنیل",
        h.thumbnail_text or "",
        "",
    ]
    out = clip_dir / "social_posts.md"
    _write_text(out, "\n".join(md))
    return out


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


@dataclass
class WorkflowResult:
    out_dir: Path
    transcript: Transcript
    highlights: list[Highlight]
    exports: list[dict] = field(default_factory=list)
    social_posts: dict[str, Path] = field(default_factory=dict)


ProgressCb = Optional[Callable[[str, float], None]]


def process_video(
    source: str,
    out_root: str | Path,
    *,
    cfg: Optional[WorkflowConfig] = None,
    guest: str = "",
    topic: str = "",
    video_title: Optional[str] = None,
    progress: ProgressCb = None,
) -> WorkflowResult:
    """Run the full Tech-Immigrants pipeline.

    Parameters
    ----------
    source
        Path to a local MP4 (or anything yt-dlp / faster-whisper can read).
    out_root
        Directory under which the per-video output folder is created.
    cfg
        :class:`WorkflowConfig`; if None, sensible defaults are used.
    guest
        Name of the speaker (used in social posts).
    topic
        Topic / title of the live (used in social posts and folder name).
    video_title
        Display title for the video; defaults to the source file name.
    progress
        Optional callback ``(stage: str, percent: float)`` for UIs.
    """
    cfg = cfg or WorkflowConfig()
    source_path = Path(source)
    title = video_title or source_path.stem

    slug = _safe_slug(f"{topic}-{title}" if topic else title)
    out_dir = Path(out_root) / f"{slug}-{_now_id()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_root = out_dir / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    # 1) Transcribe
    if progress:
        progress("transcribe", 0.0)
    transcript = transcribe(
        str(source_path),
        language=cfg.language,
        progress=lambda p: progress("transcribe", p) if progress else None,
        word_timestamps=cfg.word_timestamps,
    )
    if progress:
        progress("transcribe", 1.0)

    if cfg.save_transcript_json:
        transcript.save(out_dir / "transcript.json")

    # 2) Find highlights
    if progress:
        progress("highlights", 0.0)
    provider = cfg.provider
    if provider is None:
        # auto
        from .highlights import _auto_provider
        provider = _auto_provider()
    try:
        highlights = find_highlights(
            transcript.to_segments_dicts(),
            provider=provider,
            top_k=cfg.top_k,
        )
    except RuntimeError as e:
        # Missing API key etc. – fall back to offline ranker with a clear log.
        print(f"[workflow] LLM unavailable ({e}); using offline ranker.")
        highlights = find_highlights(
            transcript.to_segments_dicts(),
            provider="offline",
            top_k=cfg.top_k,
        )
    if progress:
        progress("highlights", 1.0)

    # 3) Enrich social posts
    highlights = enrich_social_posts(
        highlights, guest=guest, topic=topic, video_title=title
    )

    # 4) Persist highlights.json
    _write_text(
        out_dir / "highlights.json",
        json.dumps([h.to_dict() for h in highlights], ensure_ascii=False, indent=2),
    )

    # 5) Batch export
    if progress:
        progress("export", 0.0)
    style = _build_subtitle_style(cfg) if cfg.burn_subtitles else None
    segments_dicts = transcript.to_segments_dicts()
    exports = export_clip_batch(
        source=str(source_path),
        highlights=[h.to_dict() for h in highlights],
        out_root=clips_root,
        aspects=resolve_aspects(cfg.aspects),
        full_segments=segments_dicts if cfg.burn_subtitles else None,
        crop_mode=cfg.crop_mode,
        style=style,
    )
    if progress:
        progress("export", 1.0)

    # 6) Per-clip metadata + social_posts.md
    social_paths: dict[str, Path] = {}
    for i, h in enumerate(highlights, start=1):
        clip_name = f"clip_{i:03d}"
        clip_dir = clips_root / clip_name
        if not clip_dir.exists():
            clip_dir.mkdir(parents=True, exist_ok=True)
        write_clip_transcript(clip_dir, segments_dicts, h.start, h.end)
        write_clip_metadata(clip_dir, h, guest=guest, topic=topic, video_title=title)
        social_paths[clip_name] = write_clip_social_posts(clip_dir, h)

    return WorkflowResult(
        out_dir=out_dir,
        transcript=transcript,
        highlights=highlights,
        exports=exports,
        social_posts=social_paths,
    )
