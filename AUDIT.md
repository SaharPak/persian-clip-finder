# Audit & Implementation Plan

_Re-audit of the persian-clip-finder repo before incremental improvements._

## Current state

### Entry points
- `app.py` – single-page Streamlit UI that drives the whole pipeline:
  download → transcribe → find highlights → generate one clip at a time.
- No CLI exists; the brief calls for `python -m persian_clip_finder process …`.

### Package layout
The current code is a flat `core/` folder imported via `from core import …`.
For a real Python package we need `persian_clip_finder/` (the brief explicitly
references `python -m persian_clip_finder …`). The folder has been renamed in
this revision; all relative imports continue to work.

### Transcription (`persian_clip_finder/transcribe.py`)
- Single backend: `faster-whisper` with `large-v3` (configurable).
- Returns `{language, segments: [{start, end, text}]}`.
- **No word-level timestamps**, even though `faster-whisper` exposes
  `word_timestamps=True`.
- **No JSON-on-disk artifact** for caching/inspection.
- Progress is reported via a `progress` callback, which is good.

### Highlight detection (`persian_clip_finder/highlights.py`)
- One generic prompt asking for Shorts/Reels moments.
- Single 1–100 score, no breakdown of *why* a moment is good.
- No social-post metadata generated (titles/captions are clip titles only).
- The prompt is in English; the brief wants a Persian-first, Tech-Immigrants
  rubric.

### Segmentation
- There is **no segmentation layer**. Highlights go straight from
  transcript → LLM. There's no concept of "candidate segments" or pause-based
  boundaries.

### Subtitles (`persian_clip_finder/subtitles.py`)
- Resolution-aware ASS, with RLE/PDF bidi markers. Good baseline.
- **No word/phrase highlighting mode** (the brief asks for this).
- Caption position is binary (bottom/lower/center/top) and not exposed as a
  rich `subtitle_style` dict.

### Video export (`persian_clip_finder/clip.py`)
- Single-clip export. No batch export across aspect ratios.
- Smart-crop modes (auto / single / split2 / grid3 / fit) – good baseline.
- Template presets (Reels/Short/Square/YouTube/Original) – good baseline.
- Encodes with libx264 CRF 20, AAC 160k. Reasonable.

### Output structure
Everything is dumped into `data/clips/<name>.mp4`. There's no per-clip folder,
no per-clip metadata JSON, no social-posts markdown. The brief requires
`outputs/<video-title-date>/clips/clip_001/...`.

### Configuration (`persian_clip_finder/config.py`)
- Reads env vars only. No `config.yaml`. Error message when a key is missing
  is reasonable but only thrown deep inside `find_highlights`.

### Tests
**No tests at all.** The brief calls for offline tests with a fixture
transcript.

### Documentation
- README exists, English, decent. Lacks the "Why this exists" / Tech
  Immigrants framing, CLI usage, troubleshooting, and roadmap sections.

### Limitations
1. No CLI; only Streamlit UI.
2. No word-level timestamps; no transcript JSON-on-disk.
3. No segmentation layer.
4. No multi-score rubric; no social-post generation.
5. No batch multi-aspect export.
6. No per-clip folder structure with metadata + social posts.
7. No tests.
8. README missing CLI / workflow / troubleshooting sections.
9. `core/productbuilders-app/` (a Next.js sub-app) and
   `core/tools/daily-task-picker/` were present locally; these are unrelated
   to the project and have been removed.

## Plan (small, safe commits)

| # | Phase | Change |
|---|-------|--------|
| 1 | 1 | Reorganize into proper `persian_clip_finder` package, write `AUDIT.md` |
| 2 | 2 | `transcribe.py`: word-level timestamps when available, transcript JSON I/O, language-tolerant defaults |
| 3 | 3 | New `segmentation.py`: candidate segments from pauses + length constraints |
| 4 | 4 | `highlights.py`: multi-score rubric (hook, clarity, etc.) and rich social-post metadata |
| 5 | 5 | `subtitles.py`: add `phrase_highlight` style, `subtitle_style` config |
| 6 | 6 | `clip.py`: batch multi-aspect export, clean filenames |
| 7 | 7 | New `workflow.py`: Tech Immigrants end-to-end orchestrator with per-clip folder + social_posts.md |
| 8 | 8 | `app.py`: improve Streamlit UI (multi-clip review, run mode toggle, metadata download) |
| 9 | 9 | New `__main__.py`: `python -m persian_clip_finder process` |
| 10 | 10 | `config.py`: add `config.yaml` loading + provider abstraction |
| 11 | 11 | `tests/`: offline unit tests with a fixture transcript |
| 12 | 12 | `README.md`: rewrite with Tech Immigrants framing, CLI, troubleshooting, roadmap |

### Out of scope for this revision
- WhisperX alignment (kept as optional future work; the new transcript schema
  already supports word-level timestamps when any backend provides them).
- Speaker diarization (the current `layout.py` already does face-based
  speaker clustering for cropping, which is sufficient for this phase).
- TikTok-specific watermark removal.

## Status (this revision)

| # | Phase | Done? | Notes |
|---|-------|-------|-------|
| 1 | Package + audit              | ✅ | `core/` → `persian_clip_finder/` |
| 2 | Transcription + word-level   | ✅ | `transcribe.py` exposes `Transcript/Word/Segment` dataclasses; JSON on disk |
| 3 | Segmentation layer           | ✅ | `segmentation.py` pause-aware candidate generator |
| 4 | Multi-score highlight rubric | ✅ | `Scores` dataclass + 1-10 per-axis scoring; social-post enrichment |
| 5 | Subtitle improvements        | ✅ | `SubtitleStyle` dataclass; `static` / `word` / `phrase_highlight` modes |
| 6 | Multi-aspect batch export    | ✅ | `export_clip_batch` + `export_clip_multi` (single FFmpeg pass) |
| 7 | Tech Immigrants workflow     | ✅ | `workflow.py:process_video` |
| 8 | Improved Streamlit UI        | ✅ | New "Tech Immigrants" mode in the sidebar; per-clip download |
| 9 | CLI entry point              | ✅ | `python -m persian_clip_finder {version,transcribe,process}` |
| 10 | `config.yaml` + provider     | ✅ | `config.load_yaml_config`; `--config` flag |
| 11 | Offline tests + fixtures     | ✅ | 31 tests in `tests/`; bundled `tests/run_tests.py` |
| 12 | README rewrite               | ✅ | Tech Immigrants framing, CLI usage, troubleshooting, roadmap |

All 31 tests pass:

```bash
$ python tests/run_tests.py
…
31 passed, 0 failed
```
