"""Shared configuration and paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Directories ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOWNLOAD_DIR = DATA_DIR / "downloads"
CLIP_DIR = DATA_DIR / "clips"
SUBTITLE_DIR = DATA_DIR / "subtitles"

for _d in (DATA_DIR, DOWNLOAD_DIR, CLIP_DIR, SUBTITLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- API keys ------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# --- Whisper -------------------------------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")

# --- LLM models ----------------------------------------------------------
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")
