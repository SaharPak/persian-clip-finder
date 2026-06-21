"""Tests for config.yaml loading."""

from __future__ import annotations

import os
from pathlib import Path

from persian_clip_finder import config


def test_config_has_expected_keys():
    status = config.llm_status()
    assert "anthropic" in status
    assert "openai" in status


def test_load_yaml_config_missing(tmp_path: Path):
    assert config.load_yaml_config(tmp_path / "nope.yaml") == {}


def test_load_yaml_config_happy(tmp_path: Path):
    try:
        import yaml  # noqa: F401
    except ImportError:
        # Without PyYAML, the loader returns {} - that's a documented
        # contract, so the test trivially passes.
        assert config.load_yaml_config(tmp_path / "x.yaml") == {}
        return

    p = tmp_path / "config.yaml"
    p.write_text(
        "language: en\n"
        "top_k: 12\n"
        "aspects: [9x16, 1x1]\n",
        encoding="utf-8",
    )
    data = config.load_yaml_config(p)
    assert data["language"] == "en"
    assert data["top_k"] == 12
    assert data["aspects"] == ["9x16", "1x1"]


def test_get_with_default():
    assert config.get("definitely_not_a_real_key", "fallback") == "fallback"
