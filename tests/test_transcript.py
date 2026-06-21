"""Tests for the Transcript / Word / Segment data model."""

from __future__ import annotations

import json
from pathlib import Path


from persian_clip_finder.transcribe import (
    Segment,
    Transcript,
    Word,
    format_timestamp,
    transcript_to_text,
)


FIXTURE = Path(__file__).parent / "fixtures" / "sample_transcript.json"


def test_transcript_loads_from_fixture():
    t = Transcript.load(FIXTURE)
    assert t.video_id == "fixture-001"
    assert t.language == "fa"
    assert t.duration == 180.0
    assert len(t.segments) == 8


def test_transcript_roundtrip_json(tmp_path: Path):
    t = Transcript.load(FIXTURE)
    out = tmp_path / "t.json"
    t.save(out)
    assert json.loads(out.read_text(encoding="utf-8")) == t.to_dict()


def test_segments_have_words():
    t = Transcript.load(FIXTURE)
    seg1 = t.segments[0]
    assert isinstance(seg1, Segment)
    assert len(seg1.words) > 0
    assert isinstance(seg1.words[0], Word)


def test_to_segments_dicts_matches_legacy_shape():
    t = Transcript.load(FIXTURE)
    dicts = t.to_segments_dicts()
    assert isinstance(dicts, list)
    assert {"id", "start", "end", "text", "words"}.issubset(dicts[0].keys())


def test_format_timestamp_zero_pads():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3661) == "01:01:01"


def test_transcript_to_text_contains_persian():
    t = Transcript.load(FIXTURE)
    text = transcript_to_text(t.to_segments_dicts())
    assert "تک ایمیگرنتس" in text
