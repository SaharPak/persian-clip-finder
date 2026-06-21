"""Tests for the public clip API that don't require FFmpeg."""

from __future__ import annotations

from persian_clip_finder.clip import ASPECT_ALIASES, resolve_aspects


def test_resolve_aspects_aliases():
    out = resolve_aspects(["9x16", "1x1", "vertical", "16x9"])
    assert "reels" in out
    assert "square" in out
    assert "youtube" in out
    # No duplicates
    assert len(out) == len(set(out))


def test_resolve_aspects_unknown_falls_back_to_reels():
    out = resolve_aspects(["bogus"])
    assert out == ["reels"]


def test_aspect_aliases_canonical_keys():
    # All values must be valid template keys (TEMPLATES keys); the
    # templates module is the single source of truth.
    from persian_clip_finder.templates import TEMPLATES

    for k in ASPECT_ALIASES.values():
        assert k in TEMPLATES, f"alias maps to unknown template {k!r}"
