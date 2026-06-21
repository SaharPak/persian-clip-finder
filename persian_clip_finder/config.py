"""Shared configuration and paths.

Resolution order (highest priority first):

1. Environment variables (and ``.env`` via ``python-dotenv``)
2. Optional ``config.yaml`` in the project root
3. Built-in defaults

We intentionally keep the module *side-effect free* w.r.t. any heavy imports
so it can be loaded from tests, the CLI, the Streamlit app, and the workflow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    # python-dotenv is optional for running the package; the user can still
    # set environment variables manually. We silently continue.
    pass


# --------------------------------------------------------------------------- #
# Directories
# --------------------------------------------------------------------------- #


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads"
CLIP_DIR = DATA_DIR / "clips"
SUBTITLE_DIR = DATA_DIR / "subtitles"
OUTPUT_DIR = BASE_DIR / "outputs"

for _d in (DATA_DIR, DOWNLOAD_DIR, CLIP_DIR, SUBTITLE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# API keys
# --------------------------------------------------------------------------- #


ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()

# Loud and helpful messages if a caller tries to use an LLM without a key.
if not ANTHROPIC_API_KEY:
    os.environ.setdefault("ANTHROPIC_MISSING_WARNED", "1")
if not OPENAI_API_KEY:
    os.environ.setdefault("OPENAI_MISSING_WARNED", "1")


# --------------------------------------------------------------------------- #
# Whisper / LLM defaults (env-overridable)
# --------------------------------------------------------------------------- #


WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "auto")

CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
GPT_MODEL: str = os.getenv("GPT_MODEL", "gpt-4o")


# --------------------------------------------------------------------------- #
# config.yaml loading
# --------------------------------------------------------------------------- #


def load_yaml_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` (or ``config.local.yaml``) and return a dict.

    Returns an empty dict if the file is missing or PyYAML is not installed —
    config.yaml is always optional.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path).expanduser())
    else:
        candidates.append(BASE_DIR / "config.yaml")
        candidates.append(BASE_DIR / "config.local.yaml")

    for c in candidates:
        if not c.exists():
            continue
        try:
            import yaml  # type: ignore
        except ImportError:
            return {}
        try:
            data = yaml.safe_load(c.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def get(key: str, default: Any = None) -> Any:
    """Look up a setting in config.yaml, falling back to the provided default."""
    return load_yaml_config().get(key, default)


# --------------------------------------------------------------------------- #
# Convenience diagnostic
# --------------------------------------------------------------------------- #


def has_any_llm_key() -> bool:
    return bool(ANTHROPIC_API_KEY or OPENAI_API_KEY)


def llm_status() -> dict[str, bool]:
    return {
        "anthropic": bool(ANTHROPIC_API_KEY),
        "openai": bool(OPENAI_API_KEY),
    }
