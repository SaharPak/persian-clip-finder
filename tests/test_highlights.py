"""Tests for the highlight scoring + social-post enrichment."""

from __future__ import annotations

from persian_clip_finder.highlights import (
    Highlight,
    Scores,
    enrich_social_posts,
    find_highlights_offline,
)


def test_scores_overall_is_weighted():
    s = Scores(
        hook_strength=10,
        clarity=10,
        standalone_value=10,
        educational_value=10,
        emotional_value=10,
        shareability=10,
        audience_fit_for_tech_immigrants=10,
    )
    assert s.overall() == 10.0


def test_scores_round_trip():
    s = Scores(hook_strength=7, clarity=5)
    d = s.to_dict()
    s2 = Scores.from_dict(d)
    assert s2.hook_strength == 7
    assert s2.clarity == 5


def test_highlight_round_trip():
    h = Highlight(
        start=10.0,
        end=40.0,
        title="نمونه",
        hook="این یک تست است",
        reason="چون تست است",
        scores=Scores(hook_strength=8),
        tags=["test"],
    )
    d = h.to_dict()
    h2 = Highlight.from_dict(d)
    assert h2.title == h.title
    assert h2.start == h.start
    assert h2.end == h.end
    assert h2.overall_score == h.overall_score


def test_enrich_social_posts_fills_all_fields():
    h = Highlight(
        start=10.0,
        end=40.0,
        title="کلیپ تست",
        hook="یک شروع قوی",
        reason="چون کوتاه است",
        scores=Scores(hook_strength=8),
        tags=["career"],
    )
    out = enrich_social_posts([h], guest="مهمان", topic="مهاجرت")
    h2 = out[0]
    assert h2.caption_instagram
    assert h2.caption_linkedin
    assert h2.post_x
    assert h2.post_telegram
    assert h2.hashtags
    assert h2.pinned_comment
    assert h2.thumbnail_text


def test_offline_highlights_uses_segmentation():
    # Build a 100s transcript that has at least one ~30s window and a clear
    # pause boundary, so the candidate-segmentation layer has something
    # to chew on (its default minimum clip length is 20s).
    segments = []
    t = 0.0
    for i in range(20):
        dur = 5.0
        segments.append({"id": i + 1, "start": t, "end": t + dur, "text": f"جمله {i + 1}"})
        t += dur
    # Insert a clear pause between seg 10 and seg 11
    segments[10]["start"] = 65.0
    segments[10]["end"] = 70.0
    segments[11]["start"] = 71.0
    segments[11]["end"] = 76.0
    out = find_highlights_offline(segments, top_k=3)
    assert len(out) >= 1
    for h in out:
        assert h.end > h.start
        assert h.overall_score > 0
