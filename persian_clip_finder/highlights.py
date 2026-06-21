"""Highlight detection for Tech-Immigrants-style Persian content.

The public entry point is :func:`find_highlights`, which returns a list of
``Highlight`` objects, each carrying:

* time bounds (``start``, ``end``)
* a multi-dimensional score breakdown
* rich social-post metadata (Persian titles, captions, hashtags, …)

Two backends are supported out of the box:

* ``claude`` – Anthropic Claude (default if ``ANTHROPIC_API_KEY`` is set)
* ``gpt``    – OpenAI GPT-4o / GPT-4 family

If no LLM API key is available, :func:`find_highlights_offline` produces a
heuristic ranking from pause density and length — useful so the rest of the
pipeline (segmentation, export, UI) still works.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from .config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    GPT_MODEL,
    OPENAI_API_KEY,
)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


SCORE_KEYS = (
    "hook_strength",
    "clarity",
    "standalone_value",
    "educational_value",
    "emotional_value",
    "shareability",
    "audience_fit_for_tech_immigrants",
)


@dataclass
class Scores:
    """Per-axis 1-10 scores for a candidate moment."""

    hook_strength: int = 0
    clarity: int = 0
    standalone_value: int = 0
    educational_value: int = 0
    emotional_value: int = 0
    shareability: int = 0
    audience_fit_for_tech_immigrants: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Scores":
        kwargs = {k: int(round(float(data.get(k, 0) or 0))) for k in SCORE_KEYS}
        return cls(**kwargs)

    def overall(self) -> float:
        """Weighted overall score in 0-10. Weights biased toward hook/clarity."""
        weights = {
            "hook_strength": 1.5,
            "clarity": 1.4,
            "standalone_value": 1.3,
            "educational_value": 1.2,
            "emotional_value": 1.0,
            "shareability": 1.2,
            "audience_fit_for_tech_immigrants": 1.4,
        }
        total_w = sum(weights.values())
        s = sum(getattr(self, k) * w for k, w in weights.items()) / total_w
        return round(min(10.0, max(0.0, s)), 2)


@dataclass
class Highlight:
    """A selected moment with full social-post metadata."""

    start: float
    end: float
    title: str
    hook: str
    reason: str
    scores: Scores = field(default_factory=Scores)
    tags: list[str] = field(default_factory=list)
    caption_instagram: str = ""
    caption_linkedin: str = ""
    post_x: str = ""
    post_telegram: str = ""
    hashtags: list[str] = field(default_factory=list)
    pinned_comment: str = ""
    thumbnail_text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def overall_score(self) -> float:
        return self.scores.overall()

    @property
    def score(self) -> int:
        """Back-compat single score (0-100)."""
        return int(round(self.overall_score * 10))

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "title": self.title,
            "hook": self.hook,
            "reason": self.reason,
            "score": self.score,
            "scores": self.scores.to_dict(),
            "tags": list(self.tags),
            "caption_instagram": self.caption_instagram,
            "caption_linkedin": self.caption_linkedin,
            "post_x": self.post_x,
            "post_telegram": self.post_telegram,
            "hashtags": list(self.hashtags),
            "pinned_comment": self.pinned_comment,
            "thumbnail_text": self.thumbnail_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Highlight":
        scores = Scores.from_dict(data.get("scores", {}) or {})
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            title=str(data.get("title", "")).strip(),
            hook=str(data.get("hook", "")).strip(),
            reason=str(data.get("reason", "")).strip(),
            scores=scores,
            tags=list(data.get("tags", []) or []),
            caption_instagram=str(data.get("caption_instagram", "")).strip(),
            caption_linkedin=str(data.get("caption_linkedin", "")).strip(),
            post_x=str(data.get("post_x", "")).strip(),
            post_telegram=str(data.get("post_telegram", "")).strip(),
            hashtags=list(data.get("hashtags", []) or []),
            pinned_comment=str(data.get("pinned_comment", "")).strip(),
            thumbnail_text=str(data.get("thumbnail_text", "")).strip(),
        )


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = (
    "You are an expert short-form video editor for Persian (Farsi) "
    "Tech-Immigrants content — long-form live conversations and podcasts "
    "aimed at Iranian / Persian-speaking software engineers, founders, and "
    "tech workers thinking about moving abroad or working internationally. "
    "You find the moments that work best as standalone YouTube Shorts, "
    "Instagram Reels, LinkedIn clips, X posts, and Telegram posts."
)


USER_TEMPLATE = """Below is a timestamped transcript of a Persian Tech-Immigrants
video. Each line is formatted as: [start_seconds-end_seconds] text

TASK:
Select the {top_k} strongest moments for short-form clips (target 30-90s each).

Look for moments that are:
- practical career advice
- migration / settlement insight
- controversial or debate-worthy opinion
- emotional or relatable moment
- clear explanation of a technical concept
- a useful quote from the guest
- a strong hook for an Iranian / Persian-speaking tech audience
- a good standalone educational clip

For each selected moment return STRICT JSON with this exact shape (no
markdown, no commentary, no questions):

{{
  "highlights": [
    {{
      "start": 12.0,
      "end": 48.5,
      "title": "Persian title for the clip",
      "hook": "Single-line Persian hook (max 90 chars)",
      "reason": "One-line Persian reason why this clip is strong",
      "tags": ["career", "migration"],
      "scores": {{
        "hook_strength": 8,
        "clarity": 7,
        "standalone_value": 8,
        "educational_value": 9,
        "emotional_value": 6,
        "shareability": 7,
        "audience_fit_for_tech_immigrants": 9
      }}
    }}
  ]
}}

Rules:
- All scores are integers 1-10.
- start and end MUST be in seconds and fall within the transcript.
- Titles, hooks, reasons and tags MUST be in Persian (Farsi).
- Tags can include English technical terms when natural.
- Each highlight should be a self-contained clip, ideally 30-90 seconds.
- Return at least 1 highlight. Never ask for more content; work with what is
  provided, even if it is short.

TRANSCRIPT:
{transcript}
"""


# --------------------------------------------------------------------------- #
# Transcript rendering
# --------------------------------------------------------------------------- #


def _build_transcript(segments: list[dict], max_chars: int = 45000) -> str:
    lines = [f"[{s['start']}-{s['end']}] {s['text']}" for s in segments]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


# --------------------------------------------------------------------------- #
# JSON extraction / normalisation
# --------------------------------------------------------------------------- #


def _extract_json(raw: str) -> dict:
    """Best-effort JSON extraction from a model response."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _normalize_highlights(data: dict, segments: list[dict]) -> list[Highlight]:
    items = data.get("highlights", data if isinstance(data, list) else [])
    out: list[Highlight] = []
    for h in items:
        try:
            start = float(h["start"])
            end = float(h["end"])
        except (KeyError, ValueError, TypeError):
            continue
        if end <= start:
            continue
        # Clamp to the actual transcript range
        if segments:
            t_end = max((float(s["end"]) for s in segments), default=end)
            t_start = min((float(s["start"]) for s in segments), default=start)
            start = max(start, t_start)
            end = min(end, t_end)
        scores = Scores.from_dict(h.get("scores", {}) or {})
        out.append(
            Highlight(
                start=start,
                end=end,
                title=str(h.get("title", "")).strip(),
                hook=str(h.get("hook", "")).strip(),
                reason=str(h.get("reason", "")).strip(),
                scores=scores,
                tags=[str(t).strip() for t in (h.get("tags") or []) if str(t).strip()],
            )
        )
    out.sort(key=lambda x: x.overall_score, reverse=True)
    return out


# --------------------------------------------------------------------------- #
# LLM backends
# --------------------------------------------------------------------------- #


def _call_claude(prompt: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _call_gpt(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _auto_provider() -> str:
    if ANTHROPIC_API_KEY:
        return "claude"
    if OPENAI_API_KEY:
        return "gpt"
    return "offline"


def find_highlights(
    segments: list[dict],
    provider: Optional[str] = None,
    top_k: int = 8,
) -> list[Highlight]:
    """Return up to ``top_k`` highlights sorted by overall score.

    If ``provider`` is None we auto-select: claude > gpt > offline. The
    "offline" branch is a heuristic ranker that does not call any LLM.
    """
    if not segments:
        return []

    provider = (provider or _auto_provider()).lower()

    if provider == "offline":
        return find_highlights_offline(segments, top_k=top_k)

    if provider == "claude":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        raw = _call_claude(
            USER_TEMPLATE.format(top_k=top_k, transcript=_build_transcript(segments))
        )
    elif provider == "gpt":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        raw = _call_gpt(
            USER_TEMPLATE.format(top_k=top_k, transcript=_build_transcript(segments))
        )
    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Use 'claude', 'gpt', or 'offline'."
        )

    return _normalize_highlights(_extract_json(raw), segments)[:top_k]


def find_highlights_offline(
    segments: list[dict], top_k: int = 8
) -> list[Highlight]:
    """Heuristic fallback that does not require an LLM.

    Strategy: pick the segments with the densest text + the longest gaps around
    them. We then synthesize Highlight objects with empty social-post fields
    and a score derived from segment length and proximity to a pause.
    """
    if not segments:
        return []
    from .segmentation import generate_candidates

    candidates = generate_candidates(segments)
    out: list[Highlight] = []
    for c in candidates[:top_k]:
        first_words = c.text.split()[:8]
        title = " ".join(first_words).strip() or f"کلیپ از {int(c.start)} ثانیه"
        if len(title) > 60:
            title = title[:60].rstrip() + "…"
        out.append(
            Highlight(
                start=c.start,
                end=c.end,
                title=title,
                hook="",
                reason=c.reason or "انتخاب بر اساس مکث و طول پنجره",
                scores=Scores(
                    hook_strength=int(round(c.confidence * 6 + 2)),
                    clarity=int(round(c.confidence * 6 + 2)),
                    standalone_value=int(round(c.confidence * 6 + 2)),
                    educational_value=int(round(c.confidence * 5 + 2)),
                    emotional_value=int(round(c.confidence * 4 + 2)),
                    shareability=int(round(c.confidence * 5 + 2)),
                    audience_fit_for_tech_immigrants=5,
                ),
                tags=["auto-detected"],
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Social-post metadata enrichment
# --------------------------------------------------------------------------- #


def enrich_social_posts(
    highlights: list[Highlight],
    *,
    guest: str = "",
    topic: str = "",
    video_title: str = "",
) -> list[Highlight]:
    """Fill in Instagram/LinkedIn/X/Telegram fields for highlights that lack them.

    The LLM is asked for the title + hook; the rest of the social-post fields
    can be templated from those. This is intentionally deterministic so it
    also works in the offline / no-LLM fallback.
    """
    for h in highlights:
        if not h.caption_instagram:
            h.caption_instagram = _build_instagram_caption(h, guest, topic)
        if not h.caption_linkedin:
            h.caption_linkedin = _build_linkedin_post(h, guest, topic)
        if not h.post_x:
            h.post_x = _build_x_post(h)
        if not h.post_telegram:
            h.post_telegram = _build_telegram_post(h, guest, topic)
        if not h.hashtags:
            h.hashtags = _build_hashtags(h, topic)
        if not h.pinned_comment:
            h.pinned_comment = _build_pinned_comment(h, video_title)
        if not h.thumbnail_text:
            h.thumbnail_text = _build_thumbnail_text(h)
    return highlights


def _build_instagram_caption(h: Highlight, guest: str, topic: str) -> str:
    parts: list[str] = []
    if h.hook:
        parts.append(f"🔥 {h.hook}")
    if h.title:
        parts.append(f"\n{h.title}")
    if guest:
        parts.append(f"\nاز {guest}")
    if topic:
        parts.append(f" — {topic}")
    if h.reason:
        parts.append(f"\n\n{h.reason}")
    parts.append("\n\n📌 ذخیره کن تا بعداً ببینی.")
    if h.hashtags:
        parts.append("\n" + " ".join("#" + t for t in h.hashtags))
    return "".join(parts).strip()


def _build_linkedin_post(h: Highlight, guest: str, topic: str) -> str:
    lines: list[str] = []
    if h.hook:
        lines.append(h.hook)
    if h.title:
        lines.append("")
        lines.append(h.title)
    body = h.reason or h.hook
    if body:
        lines.append("")
        lines.append(body)
    if guest:
        lines.append("")
        lines.append(f"— از گفت‌وگوی «{topic or 'تک ایمیگرنتس'}» با {guest}")
    lines.append("")
    lines.append("#TechImmigrants #PersianTech")
    return "\n".join(lines).strip()


def _build_x_post(h: Highlight) -> str:
    pieces = [p for p in (h.hook, h.title) if p]
    text = " · ".join(pieces)
    if len(text) > 230:
        text = text[:227].rstrip() + "…"
    tags = " ".join("#" + t for t in h.hashtags[:3])
    if tags:
        text = f"{text}\n{tags}"
    return text.strip()


def _build_telegram_post(h: Highlight, guest: str, topic: str) -> str:
    parts: list[str] = []
    if h.hook:
        parts.append(f"<b>{h.hook}</b>")
    if h.title:
        parts.append("")
        parts.append(h.title)
    if h.reason:
        parts.append("")
        parts.append(h.reason)
    if guest or topic:
        parts.append("")
        header = "🎙 گفت‌وگوی تک ایمیگرنتس"
        if topic:
            header += f" — {topic}"
        if guest:
            header += f" با {guest}"
        parts.append(header)
    if h.hashtags:
        parts.append("")
        parts.append(" ".join("#" + t for t in h.hashtags))
    return "\n".join(parts).strip()


def _build_hashtags(h: Highlight, topic: str) -> list[str]:
    base = ["تک_ایمیگرنتس", "مهاجرت_تک", "فارسی", "پادکست_فارسی"]
    for tag in h.tags:
        t = tag.strip().lstrip("#")
        if t and t not in base:
            base.append(t)
    if topic:
        topic_tag = topic.strip().replace(" ", "_")
        if topic_tag and topic_tag not in base:
            base.append(topic_tag)
    return base[:10]


def _build_pinned_comment(h: Highlight, video_title: str) -> str:
    if h.reason:
        return h.reason
    if video_title:
        return f"این کلیپ از {video_title} استخراج شده است."
    return "نظر شما چیست؟ در کامنت‌ها بنویسید."


def _build_thumbnail_text(h: Highlight) -> str:
    text = h.hook or h.title
    if len(text) > 40:
        text = text[:40].rstrip() + "…"
    return text
