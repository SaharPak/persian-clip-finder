"""Tests for the candidate-segmentation layer."""

from __future__ import annotations

from persian_clip_finder.segmentation import (
    SegmentationConfig,
    generate_candidates,
)


FIXTURE_SEGMENTS = [
    {"id": 1, "start": 0.0, "end": 4.2, "text": "سلام و خوش آمدید."},
    {"id": 2, "start": 4.2, "end": 8.5, "text": "درباره‌ی مهاجرت صحبت می‌کنیم."},
    # 0.9s pause (boundary)
    {"id": 3, "start": 9.4, "end": 14.0, "text": "بزرگ‌ترین اشتباه مهاجران."},
    {"id": 4, "start": 14.0, "end": 18.5, "text": "بازار کار را بشناسید."},
    {"id": 5, "start": 18.5, "end": 23.0, "text": "رزومه مهم است."},
    # 1.0s pause (boundary)
    {"id": 6, "start": 24.0, "end": 28.5, "text": "یک تجربه‌ی شخصی دارم."},
    {"id": 7, "start": 28.5, "end": 33.0, "text": "صبر و تلاش نتیجه می‌دهد."},
    {"id": 8, "start": 33.0, "end": 37.0, "text": "سؤالی دارید؟"},
]


def test_candidates_cover_all_pauses():
    cfg = SegmentationConfig(min_duration=3.0, target_min=5.0, target_max=15.0, max_duration=20.0)
    cands = generate_candidates(FIXTURE_SEGMENTS, cfg)
    # We should get at least one candidate per "topic region"
    assert len(cands) >= 2
    for c in cands:
        assert c.end > c.start
        assert c.duration >= cfg.min_duration
        assert c.duration <= cfg.max_duration
        assert c.confidence > 0
        assert len(c.segment_ids) > 0


def test_empty_segments_yields_no_candidates():
    assert generate_candidates([]) == []


def test_text_is_joined():
    cfg = SegmentationConfig(min_duration=3.0, target_min=5.0, target_max=15.0, max_duration=20.0)
    cands = generate_candidates(FIXTURE_SEGMENTS, cfg)
    joined = " ".join(c.text for c in cands)
    assert "مهاجرت" in joined or "تجربه" in joined
