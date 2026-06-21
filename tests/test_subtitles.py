"""Tests for ASS subtitle generation."""

from __future__ import annotations

from pathlib import Path

from persian_clip_finder.subtitles import (
    RLE,
    PDF,
    SubtitleStyle,
    build_ass,
    segments_in_window,
)


def test_segments_in_window_time_shifts():
    segs = [
        {"start": 0.0, "end": 5.0, "text": "intro"},
        {"start": 5.0, "end": 10.0, "text": "middle"},
        {"start": 12.0, "end": 18.0, "text": "outro"},
    ]
    out = segments_in_window(segs, 4.0, 11.0)
    assert len(out) == 2
    # First seg [0,5] clipped to [4,5] -> clip-relative [0,1]
    assert out[0]["start"] == 0.0
    assert out[0]["end"] == 1.0
    # Second seg [5,10] fully inside -> clip-relative [1,6]
    assert out[1]["start"] == 1.0
    assert out[1]["end"] == 6.0


def test_build_ass_static_mode(tmp_path: Path):
    segs = [
        {"start": 0.0, "end": 2.0, "text": "سلام"},
        {"start": 2.0, "end": 4.0, "text": "خداحافظ"},
    ]
    out = build_ass(segs, tmp_path / "out.ass", width=1080, height=1920)
    p = Path(out)
    content = p.read_text(encoding="utf-8")
    assert "[Script Info]" in content
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert "Dialogue:" in content
    # bidi markers
    assert RLE in content
    assert PDF in content


def test_build_ass_phrase_highlight(tmp_path: Path):
    segs = [{"start": 0.0, "end": 3.0, "text": "یک جمله‌ی نسبتاً طولانی برای تست"}]
    style = SubtitleStyle(mode="phrase_highlight", font="Geeza Pro")
    out = build_ass(segs, tmp_path / "out.ass", width=1080, height=1920, style=style)
    content = Path(out).read_text(encoding="utf-8")
    # The phrase_highlight path inserts a primary-colour override
    assert "\\1c" in content


def test_build_ass_word_mode_uses_words(tmp_path: Path):
    segs = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "یک دو",
            "words": [
                {"word": "یک", "start": 0.0, "end": 1.0},
                {"word": "دو", "start": 1.0, "end": 2.0},
            ],
        }
    ]
    style = SubtitleStyle(mode="word")
    out = build_ass(segs, tmp_path / "out.ass", width=720, height=1280, style=style)
    content = Path(out).read_text(encoding="utf-8")
    # Two word-level dialogue lines for two words
    assert content.count("Dialogue:") == 2
