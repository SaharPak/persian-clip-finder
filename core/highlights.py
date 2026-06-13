"""Find the best short-clip moments with an LLM (Claude or GPT-4o)."""

from __future__ import annotations

import json
import re

from .config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    GPT_MODEL,
    OPENAI_API_KEY,
)

SYSTEM_PROMPT = (
    "You are an expert short-form video editor for Persian (Farsi) podcasts "
    "and livestreams. You find the moments that work best as standalone "
    "YouTube Shorts and Instagram Reels (ideally 20-60 seconds each)."
)

USER_TEMPLATE = """Below is a timestamped transcript of a Persian video.
Each line is formatted as: [start_seconds-end_seconds] text

TASKS:
1. Segment the video into its main topics (internally).
2. Select the TOP 20 most interesting standalone moments for Shorts/Reels.

Look especially for:
- Strong opinions / hot takes
- Emotional moments
- Storytelling
- Career advice
- AI insights
- Funny moments

Rules:
- Each highlight should be a self-contained clip, ideally 20-60 seconds.
- start and end MUST be in seconds (numbers) and fall within the transcript.
- title and reason MUST be written in Persian (Farsi).
- score is an integer 1-100 (how strong the clip is).
- Return STRICTLY valid JSON, no markdown, no commentary.

Output JSON shape:
{{
  "highlights": [
    {{
      "start": 12.0,
      "end": 48.5,
      "title": "عنوان فارسی",
      "reason": "دلیل فارسی",
      "score": 87
    }}
  ]
}}

TRANSCRIPT:
{transcript}
"""


def _build_transcript(segments: list[dict], max_chars: int = 45000) -> str:
    lines = [f"[{s['start']}-{s['end']}] {s['text']}" for s in segments]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


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


def _normalize(data: dict) -> list[dict]:
    highlights = data.get("highlights", data if isinstance(data, list) else [])
    out: list[dict] = []
    for h in highlights:
        try:
            out.append(
                {
                    "start": float(h["start"]),
                    "end": float(h["end"]),
                    "title": str(h.get("title", "")).strip(),
                    "reason": str(h.get("reason", "")).strip(),
                    "score": int(round(float(h.get("score", 0)))),
                }
            )
        except (KeyError, ValueError, TypeError):
            continue
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


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


def find_highlights(segments: list[dict], provider: str = "claude") -> list[dict]:
    """Return up to 20 highlights, sorted by score (descending)."""
    if not segments:
        return []

    prompt = USER_TEMPLATE.format(transcript=_build_transcript(segments))

    provider = provider.lower()
    if provider == "claude":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        raw = _call_claude(prompt)
    elif provider == "gpt":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        raw = _call_gpt(prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return _normalize(_extract_json(raw))[:20]
