# 🎬 Persian Clip Finder

Turn a long Persian YouTube live or local MP4 into a set of
ready-to-post short clips with:

- 🧠 **Whisper large-v3** transcription (Persian / Farsi) with optional
  word-level timestamps
- ✂️ **Candidate segmentation** based on pauses and target duration
- ✨ **LLM-based highlight detection** with a multi-axis score rubric tuned
  for the *Tech Immigrants* audience
- 🇮🇷 **RTL Persian captions** burned into the video (three styles:
  static / word / phrase-highlight)
- 📐 **Multiple aspect ratios** exported in one FFmpeg pass: 9:16 (Reels /
  Shorts / TikTok), 1:1 (LinkedIn / Instagram feed), 16:9 (YouTube)
- 📝 **Ready-to-post metadata** for every clip: Persian title, hook,
  Instagram / LinkedIn / X / Telegram copy, hashtags, pinned comment,
  thumbnail text

The default workflow is built around the **Tech Immigrants Repurposing
Mode**, but the same building blocks can be reused for any Persian
podcast / live stream.

## Why this exists

> This project helps Persian-speaking tech communities repurpose long
> educational live sessions into short, useful, shareable clips.

The Tech Immigrants YouTube live sessions are typically 60-120 minutes
long. A handful of well-chosen 30-90 second clips, each with a clear
Persian hook and platform-tailored metadata, drive more discovery and
conversions than the original long-form upload. This project automates
that extraction.

## Requirements

- Python **3.10 – 3.12** (3.12 recommended; ML wheels are not yet
  available for 3.14)
- **FFmpeg with libass** for subtitle burn-in. If your system FFmpeg
  lacks libass (many Homebrew builds do), the app automatically falls
  back to a bundled static build via the `static-ffmpeg` package.
- An **Anthropic** and/or **OpenAI** API key (optional — the app runs
  in an *offline* heuristic mode if no key is set).

## Installation

```bash
git clone https://github.com/SaharPak/persian-clip-finder
cd persian-clip-finder

# create / reuse the virtual environment (uv)
uv venv --python 3.12 .venv

# install dependencies
uv pip install -r requirements.txt

# configure API keys
cp .env.example .env
#   then edit .env and add ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

`pip` works too:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to use

### A. Streamlit UI (recommended for exploration)

```bash
uv run streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501). The
sidebar has a **Mode** toggle:

- **🚀 Tech Immigrants** – paste a YouTube URL or upload an MP4, set the
  guest name and topic, and get a folder full of clips + social-post
  copy.
- **🛠 Quick (per-clip)** – the original per-clip workflow; useful for
  one-off exports.

### B. Command-line interface (recommended for automation)

```bash
# process a YouTube URL
python -m persian_clip_finder process \
  --url "https://www.youtube.com/watch?v=XXXXXXXXXXX" \
  --guest "اسم مهمان" \
  --topic "مهاجرت کاری به اروپا" \
  --clips 8 \
  --formats 9x16,1x1,16x9 \
  --subtitle-mode phrase_highlight \
  --output outputs/

# process a local MP4
python -m persian_clip_finder process \
  --input ./my-video.mp4 \
  --guest "اسم مهمان" \
  --topic "موضوع لایو" \
  --clips 5

# just transcribe (writes <input>.transcript.json)
python -m persian_clip_finder transcribe --input ./my-video.mp4

# version
python -m persian_clip_finder version
```

You can also point the CLI at a config file with `--config
path/to/config.yaml`. CLI flags override config values.

### C. Programmatic API

```python
from persian_clip_finder import process_video, WorkflowConfig

cfg = WorkflowConfig(
    language="fa",
    top_k=8,
    aspects=("9x16", "1x1", "16x9"),
    subtitle_mode="phrase_highlight",
    font="Geeza Pro",
)

result = process_video(
    source="my-video.mp4",
    out_root="outputs/",
    cfg=cfg,
    guest="اسم مهمان",
    topic="موضوع",
    video_title="Tech Immigrants – DevOps",
)
print(result.out_dir)        # outputs/tech-immigrants-devops-2025-12-08_1430
print(len(result.exports))   # number of mp4 files written
```

## Output folder structure

```
outputs/
└── <video-title-date>/
    ├── transcript.json       # full transcript with word timestamps
    ├── highlights.json       # ranked list of all detected moments
    └── clips/
        ├── clip_001/
        │   ├── clip_001_reels.mp4      # 9:16 (Reels / Shorts / TikTok)
        │   ├── clip_001_square.mp4     # 1:1 (LinkedIn / Instagram feed)
        │   ├── clip_001_youtube.mp4    # 16:9 (YouTube)
        │   ├── clip_001_reels.ass      # burned-in captions source
        │   ├── clip_001_transcript.txt # plain-text transcript slice
        │   ├── clip_001_metadata.json  # scores + raw highlight
        │   └── clip_001_social_posts.md  # ready-to-post copy
        ├── clip_002/
        └── …
```

### `social_posts.md` example

```markdown
# چرا مهاجرت کاری به اروپا سخت‌تر شده؟

_شروع: 00:12 · پایان: 01:04 · مدت: 00:52_

## یوتیوب Shorts
چرا مهاجرت کاری به اروپا سخت‌تر شده؟

## اینستاگرام / LinkedIn
🔥 چرا مهاجرت کاری به اروپا سخت‌تر شده؟
…

## X (توییتر)
چرا مهاجرت کاری به اروپا سخت‌تر شده؟ · …

## تلگرام
<b>چرا مهاجرت کاری به اروپا سخت‌تر شده؟</b>
…

## هشتگ‌ها
#تک_ایمیگرنتس #مهاجرت_تک #فارسی
```

## Subtitle modes

| Mode | Description |
|---|---|
| `static`           | One ASS line per transcript segment. Fast, conservative. |
| `word`             | One ASS line per word (only useful when word timestamps are present). Looks "karaoke" style. |
| `phrase_highlight` | One line per segment; the last 1-3 words are highlighted in yellow. Best for short-form hooks. |

Default is `phrase_highlight` (most "Shorts-y" feel).

## Templates & smart cropping

| Template key | Output | When to use |
|---|---|---|
| `reels`     | 1080×1920 (9:16) | Instagram Reels / TikTok / YouTube Shorts |
| `short`     | 1080×1920 (9:16) | YouTube Shorts (alias of `reels`) |
| `square`    | 1080×1080 (1:1)  | LinkedIn / Instagram feed |
| `youtube`   | 1920×1080 (16:9) | YouTube horizontal |
| `original`  | source dims      | Pass-through (no crop) |

The **crop / layout** strategy is face-detection-based (OpenCV Haar
cascades) and chooses between:

- `auto` – 1 speaker is centered, 2 are stacked top/bottom, 3 are
  arranged in stacked bands.
- `single` / `split2` / `grid3` – force a specific layout.
- `fit` – no crop; letterbox the whole frame.

## Configuration

`config.yaml` (or `config.local.yaml`) is read on top of the
environment, and the CLI flags override both:

```yaml
provider: claude          # claude | gpt | offline
language: fa
whisper_model: large-v3
top_k: 8
aspects: [9x16, 1x1, 16x9]
subtitle_mode: phrase_highlight
subtitle_position: lower   # bottom | lower | middle | top
font: Geeza Pro            # change on Linux (e.g. Vazirmatn)
crop_mode: auto
```

`.env` is loaded with `python-dotenv` (optional):

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
WHISPER_MODEL=large-v3
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto
```

If **no LLM API key is set**, the pipeline runs in *offline* mode: it
uses a heuristic ranker based on pause density and length, so you can
still produce clips end-to-end.

## Tests

The test suite is fully offline and uses synthetic transcripts. No real
video or model download is required.

```bash
# With pytest
python -m pytest tests/ -q

# Without pytest (the test runner is bundled in tests/run_tests.py)
python tests/run_tests.py
```

## Project layout

```
app.py                              Streamlit UI (Tech Immigrants + Quick modes)
persian_clip_finder/                The Python package
  __init__.py                       Re-exports of the public API
  __main__.py                       `python -m persian_clip_finder …` entry
  cli.py                            argparse commands (version, transcribe, process)
  config.py                         paths, env + config.yaml loading
  download.py                       yt-dlp download + upload handling
  transcribe.py                     faster-whisper, Transcript/Word/Segment dataclasses
  segmentation.py                   pause-based candidate generation
  highlights.py                     LLM scoring + social-post enrichment
  subtitles.py                      RTL Persian ASS (static / word / phrase_highlight)
  templates.py                      output templates + crop/caption options
  layout.py                         face detection + speaker clustering
  clip.py                           FFmpeg cut, template reframing, layout, subtitle burn
  workflow.py                       end-to-end Tech Immigrants pipeline
tests/                              offline tests + sample fixture transcript
config.yaml                         defaults
requirements.txt                    pinned dependency set
AUDIT.md                            re-audit & implementation plan
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: faster_whisper` | `pip install faster-whisper` (and ensure Python 3.10-3.12). |
| Subtitles render as boxes | The default font (`Geeza Pro`) is macOS-only. On Linux, set `font: Vazirmatn` in `config.yaml` (and have the font installed). |
| `ffmpeg: command not found` and the static fallback fails | `pip install static-ffmpeg` and re-run. |
| The LLM returns empty highlights | Your transcript is too short, or the API key is missing/wrong. Try `--provider offline` to confirm the rest of the pipeline works. |
| `RuntimeError: ANTHROPIC_API_KEY is not set` | Add the key to `.env`, or pass `--provider offline`. |
| Word-level timestamps are empty | The Whisper model build you installed may not include the alignment feature; the pipeline silently falls back to segment-level timestamps. |

## Roadmap

- [ ] WhisperX backend for higher-quality word alignment
- [ ] Speaker diarization (pyannote-audio)
- [ ] Auto-generated thumbnails (Pillow + a Persian font)
- [ ] Telegram-channel poster (telethon) for fully-automatic publishing
- [ ] Optional vertical-video safe-area preview in the UI

## License

MIT — see `LICENSE` (if present in your fork).
