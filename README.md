# 🎬 Persian Clip Finder

A simple, single-machine MVP that finds the best short-clip moments in Persian
podcasts and livestreams. Give it a YouTube URL or an MP4, and it will:

1. Download the video (yt-dlp)
2. Transcribe it with **Whisper large-v3** (faster-whisper)
3. Ask **Claude Sonnet** or **GPT-4o** to find the top 20 Shorts/Reels moments
4. Cut any highlight to MP4 with **FFmpeg**
5. Burn in **RTL Persian subtitles** (ASS)

No auth. No database. Just a Streamlit app.

## Requirements

- Python 3.10–3.12 (3.12 recommended; ML wheels are not yet available for 3.14)
- FFmpeg **with libass** for subtitle burn-in. If your system FFmpeg lacks
  libass (many Homebrew builds do), the app automatically falls back to a
  bundled static build via the `static-ffmpeg` package — no action needed.
- An Anthropic and/or OpenAI API key

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/). A `.venv` (Python 3.12)
is created for you.

```bash
# create / reuse the virtual environment
uv venv --python 3.12 .venv

# install dependencies
uv pip install -r requirements.txt

# configure API keys
cp .env.example .env
#   then edit .env and add ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

> Plain `pip` works too: `source .venv/bin/activate && pip install -r requirements.txt`

## Run

```bash
uv run streamlit run app.py
# or: source .venv/bin/activate && streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## How to use

1. **Add a video** — paste a YouTube URL and download, or upload an MP4.
2. **Transcribe** — runs Whisper large-v3 (first run downloads the model).
3. **Find highlights** — pick Claude or GPT-4o in the sidebar, get a ranked table.
4. **Generate a clip** — choose a highlight, optionally burn Persian subtitles,
   and export/download the MP4.

## Notes

- The first transcription downloads the `large-v3` model (~3 GB) and runs on CPU
  by default (slow). With an NVIDIA GPU it uses CUDA + float16 automatically.
- Subtitles render right-to-left. The default font is **Geeza Pro** (bundled on
  macOS). Change it in the sidebar if your system uses a different Persian font
  (e.g. `Vazirmatn`, `Nazli`).
- All artifacts are written under `data/` (git-ignored).

## Project layout

```
app.py              Streamlit UI
core/
  config.py         paths, keys, model names
  download.py       yt-dlp download + upload handling
  transcribe.py     faster-whisper (large-v3)
  highlights.py     Claude / GPT-4o highlight detection
  subtitles.py      RTL Persian ASS generation
  clip.py           FFmpeg cut + subtitle burn-in
```
