"""Candidate-segment generation for short-clip extraction.

The highlight-detection step is LLM-based and works better when given a small
number of well-bounded candidate windows instead of the full transcript. This
module produces those candidates from the transcript by:

* respecting pause boundaries (gaps between segments),
* enforcing a target duration window,
* padding the edges with a small context margin,
* avoiding cuts in the middle of a sentence where possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class SegmentationConfig:
    """Tunable parameters for candidate generation."""

    min_duration: float = 20.0  # a useful clip should be at least 20s
    target_min: float = 30.0    # soft lower bound for "short clip" range
    target_max: float = 90.0    # soft upper bound for "short clip" range
    max_duration: float = 120.0 # hard ceiling
    min_pause: float = 0.7      # seconds of silence = a "boundary"
    context_pad: float = 1.0    # seconds of context added around the window

    def clamp(self, duration: float) -> float:
        """Clamp a candidate duration into [min_duration, max_duration]."""
        return max(self.min_duration, min(self.max_duration, duration))


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class CandidateSegment:
    """A short candidate clip with metadata for the LLM scorer."""

    start: float
    end: float
    text: str
    estimated_topic: str = ""
    reason: str = ""
    confidence: float = 0.0  # 0..1, computed heuristically
    segment_ids: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "text": self.text,
            "estimated_topic": self.estimated_topic,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "segment_ids": list(self.segment_ids),
        }


# --------------------------------------------------------------------------- #
# Core algorithm
# --------------------------------------------------------------------------- #


def _pauses(segments: list[dict]) -> list[float]:
    """Return the gap before each segment (in seconds)."""
    pauses: list[float] = []
    for i, seg in enumerate(segments):
        if i == 0:
            pauses.append(0.0)
            continue
        prev = segments[i - 1]
        pauses.append(max(0.0, float(seg["start"]) - float(prev["end"])))
    return pauses


def _confidence(duration: float, cfg: SegmentationConfig) -> float:
    """Heuristic confidence that this is a good short clip.

    Peaks in [target_min, target_max]; falls off outside it.
    """
    if duration < cfg.min_duration:
        return 0.1
    if duration > cfg.max_duration:
        return 0.1
    if cfg.target_min <= duration <= cfg.target_max:
        # Centre of the target window = 1.0
        mid = (cfg.target_min + cfg.target_max) / 2
        span = (cfg.target_max - cfg.target_min) / 2
        return round(max(0.4, 1.0 - abs(duration - mid) / span), 3)
    if duration < cfg.target_min:
        return round(0.4 + 0.5 * (duration - cfg.min_duration) /
                     max(0.1, cfg.target_min - cfg.min_duration), 3)
    # duration > target_max but < max_duration
    return round(0.4 + 0.5 * (cfg.max_duration - duration) /
                 max(0.1, cfg.max_duration - cfg.target_max), 3)


def _looks_like_sentence_end(text: str) -> bool:
    """True if the segment text appears to end at a sentence boundary."""
    t = (text or "").strip()
    if not t:
        return False
    # Persian full stop "؟", "!" and ASCII ".", "!", "?" are common sentence ends.
    return t[-1] in ".!?؟!"


def _join_text(segs: list[dict]) -> str:
    return " ".join(s.get("text", "").strip() for s in segs if s.get("text"))


def generate_candidates(
    segments: list[dict],
    cfg: Optional[SegmentationConfig] = None,
) -> list[CandidateSegment]:
    """Build candidate clip windows from a flat list of transcript segments.

    The algorithm:
    1. Walk through segments; when a pause >= ``cfg.min_pause`` is observed,
       mark it as a "boundary".
    2. From each boundary, accumulate following segments until the running
       window reaches ``cfg.target_max`` (or we run out of segments).
    3. Snap the end to the nearest sentence boundary if possible.
    4. Skip windows shorter than ``cfg.min_duration``.

    This intentionally produces *more* candidates than the final clip count;
    the LLM highlight scorer picks the strongest ones.
    """
    cfg = cfg or SegmentationConfig()
    if not segments:
        return []

    pauses = _pauses(segments)
    boundaries = [i for i, p in enumerate(pauses) if p >= cfg.min_pause]
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries
    if boundaries[-1] != len(segments):
        boundaries.append(len(segments))

    candidates: list[CandidateSegment] = []
    seen: set[tuple[int, int]] = set()

    for b_start, b_end in zip(boundaries[:-1], boundaries[1:]):
        # Try a few start points within this pause-bounded "topic" region.
        # The first start is exactly at the boundary; later ones start
        # a few segments in (so we cover the middle of long topics too).
        starts_to_try = [b_start]
        mid = b_start + max(1, (b_end - b_start) // 2 - 1)
        if mid not in starts_to_try and mid > b_start:
            starts_to_try.append(mid)

        for s_idx in starts_to_try:
            window: list[dict] = []
            seg_ids: list[int] = []
            for j in range(s_idx, b_end):
                seg = segments[j]
                window.append(seg)
                seg_ids.append(int(seg.get("id", j + 1)))
                dur = float(seg["end"]) - float(window[0]["start"])
                if dur >= cfg.target_max:
                    break

            if not window:
                continue
            win_start = float(window[0]["start"])
            win_end = float(window[-1]["end"])
            duration = win_end - win_start

            # Snap end to a sentence boundary if we are within 4 seconds of one
            # and the result still respects cfg.target_min.
            if duration > cfg.target_min:
                for k in range(len(window) - 1, 0, -1):
                    if _looks_like_sentence_end(window[k].get("text", "")):
                        snap_end = float(window[k]["end"])
                        if snap_end - win_start >= cfg.min_duration:
                            window = window[: k + 1]
                            seg_ids = seg_ids[: k + 1]
                            win_end = snap_end
                            duration = win_end - win_start
                        break

            if duration < cfg.min_duration:
                continue
            if duration > cfg.max_duration:
                # Hard trim to cfg.max_duration from the start.
                end_time = win_start + cfg.max_duration
                trimmed = [s for s in window if float(s["end"]) <= end_time]
                if not trimmed:
                    continue
                window = trimmed
                seg_ids = seg_ids[: len(window)]
                win_end = float(window[-1]["end"])
                duration = win_end - win_start

            # Apply context padding
            pad_start = max(0.0, win_start - cfg.context_pad)
            pad_end = win_end + cfg.context_pad
            key = (round(pad_start, 2), round(pad_end, 2))
            if key in seen:
                continue
            seen.add(key)

            text = _join_text(window)
            reason = (
                f"پنجره‌ای حدود {int(duration)} ثانیه‌ای پس از مکث {round(pauses[s_idx], 1)} ثانیه‌ای"
                if pauses[s_idx] >= cfg.min_pause
                else f"پنجره‌ای حدود {int(duration)} ثانیه‌ای در میانه‌ی یک موضوع"
            )
            candidates.append(
                CandidateSegment(
                    start=round(pad_start, 2),
                    end=round(pad_end, 2),
                    text=text,
                    estimated_topic="",  # filled in by the LLM scorer
                    reason=reason,
                    confidence=_confidence(duration, cfg),
                    segment_ids=seg_ids,
                )
            )

    # Sort by descending confidence, then by duration descending
    candidates.sort(key=lambda c: (c.confidence, c.duration), reverse=True)
    return candidates


def candidates_to_dicts(candidates: Iterable[CandidateSegment]) -> list[dict]:
    return [c.to_dict() for c in candidates]
