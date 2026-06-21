"""Persian Clip Finder — Streamlit UI.

Two modes:

1. **Quick mode** – the original per-clip workflow. Good for one-off exports.
2. **Tech Immigrants mode** – the new end-to-end pipeline. Drop a YouTube URL
   or a local MP4, fill in guest + topic, and get a folder full of
   per-aspect clips with social-post metadata.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

from persian_clip_finder import config
from persian_clip_finder.clip import (
    ASPECT_ALIASES,
    generate_clip,
    resolve_aspects,
)
from persian_clip_finder.download import download_youtube, save_upload
from persian_clip_finder.highlights import (
    Highlight,
    enrich_social_posts,
    find_highlights,
)
from persian_clip_finder.subtitles import (
    CAPTION_POSITIONS,
    SubtitleStyle,
    segments_in_window,
)
from persian_clip_finder.templates import CROP_MODES, TEMPLATES
from persian_clip_finder.transcribe import (
    format_timestamp,
    transcribe,
    transcript_to_text,
)
from persian_clip_finder.workflow import WorkflowConfig, process_video


st.set_page_config(page_title="Persian Clip Finder", page_icon="🎬", layout="wide")

# --- Session defaults ----------------------------------------------------
for key, default in {
    "video_path": None,
    "video_title": None,
    "segments": None,
    "transcript": None,
    "highlights": None,
    "last_clip": None,
    "workflow_result": None,
}.items():
    st.session_state.setdefault(key, default)


# --- Sidebar -------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio(
    "Mode",
    ["🚀 Tech Immigrants", "🛠 Quick (per-clip)"],
    index=0,
)

provider_label = st.sidebar.radio(
    "Highlight model",
    ["Claude Sonnet", "GPT-4o", "Offline (heuristic)"],
    index=0,
)
provider = {
    "Claude Sonnet": "claude",
    "GPT-4o": "gpt",
    "Offline (heuristic)": "offline",
}[provider_label]

language = st.sidebar.text_input("Transcription language", value="fa")
sub_font = st.sidebar.text_input("Subtitle font", value="Geeza Pro")
n_clips = st.sidebar.slider("Target number of clips", 1, 20, 8)
formats_csv = st.sidebar.text_input(
    "Output formats (comma-separated)", value="9x16,1x1,16x9"
)
sub_mode = st.sidebar.selectbox(
    "Subtitle mode",
    ["phrase_highlight", "static", "word"],
    index=0,
)
sub_pos = st.sidebar.selectbox(
    "Subtitle position",
    list(CAPTION_POSITIONS.keys()),
    index=1,
)

claude_ok = bool(config.ANTHROPIC_API_KEY)
gpt_ok = bool(config.OPENAI_API_KEY)
st.sidebar.markdown("**API keys**")
st.sidebar.write(f"- Anthropic: {'✅' if claude_ok else '❌ missing'}")
st.sidebar.write(f"- OpenAI: {'✅' if gpt_ok else '❌ missing'}")
st.sidebar.caption("Set keys in a .env file (see .env.example).")


# --- Header --------------------------------------------------------------
st.title("🎬 Persian Clip Finder")
st.caption(
    "Repurpose long Persian Tech Immigrants sessions into Shorts, Reels, "
    "TikTok, LinkedIn, X and Telegram clips with RTL Persian captions."
)


# ============================================================================
# Mode 1: Tech Immigrants end-to-end
# ============================================================================
if mode.startswith("🚀"):
    st.header("Tech Immigrants Repurposing Mode")

    col1, col2 = st.columns(2)
    with col1:
        url = st.text_input("YouTube URL", key="ti_url")
    with col2:
        upload = st.file_uploader(
            "…or upload an MP4",
            type=["mp4", "mov", "mkv", "m4v"],
            key="ti_upload",
        )

    col_g, col_t = st.columns(2)
    guest = col_g.text_input("Guest name", key="ti_guest")
    topic = col_t.text_input("Topic / live title", key="ti_topic")

    if st.button(
        "🚀 Run Tech Immigrants pipeline",
        type="primary",
        use_container_width=True,
    ) and (url or upload is not None):
        try:
            if url.strip():
                with st.spinner("Downloading…"):
                    info = download_youtube(url.strip())
                st.session_state.video_path = info["path"]
                st.session_state.video_title = info["title"]
            else:
                info = save_upload(upload)
                st.session_state.video_path = info["path"]
                st.session_state.video_title = info["title"]
            st.session_state.workflow_result = None

            cfg = WorkflowConfig(
                language=language.strip() or "fa",
                provider=provider,
                top_k=int(n_clips),
                aspects=tuple(
                    a.strip() for a in formats_csv.split(",") if a.strip()
                ),
                subtitle_mode=sub_mode,
                subtitle_position=sub_pos,
                font=sub_font.strip() or "Geeza Pro",
            )

            progress = st.progress(0.0, text="Starting…")

            def _on_progress(stage: str, p: float) -> None:
                label = {
                    "transcribe": "Transcribing…",
                    "highlights": "Finding highlights…",
                    "export": "Exporting clips…",
                }.get(stage, stage)
                progress.progress(min(1.0, p), text=f"{label} ({int(p*100)}%)")

            result = process_video(
                source=st.session_state.video_path,
                out_root="outputs/",
                cfg=cfg,
                guest=guest,
                topic=topic,
                video_title=st.session_state.video_title,
                progress=_on_progress,
            )
            st.session_state.workflow_result = result
            progress.empty()
            st.success(
                f"Done. {len(result.exports)} files in {result.out_dir}"
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Pipeline failed: {e}")

    result = st.session_state.workflow_result
    if result is not None:
        st.subheader("📊 Clips")
        rows = []
        for h in result.highlights:
            rows.append(
                {
                    "#": len(rows) + 1,
                    "Start": format_timestamp(h.start),
                    "End": format_timestamp(h.end),
                    "Score": h.score,
                    "Title": h.title,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("📁 Outputs")
        st.write(f"All artifacts under `{result.out_dir}`")
        for exp in result.exports:
            clip = result.out_dir / "clips" / exp["clip_name"]
            with st.expander(
                f"{exp['clip_name']} — {exp['template']} ({format_timestamp(exp['start'])}–{format_timestamp(exp['end'])})",
                expanded=False,
            ):
                p = Path(exp["out_path"])
                if p.exists():
                    st.video(str(p))
                # Social posts
                social = result.social_posts.get(exp["clip_name"])
                if social and social.exists():
                    st.download_button(
                        "⬇️ social_posts.md",
                        social.read_bytes(),
                        file_name=social.name,
                        mime="text/markdown",
                    )
                meta = clip / "clip_metadata.json"
                if meta.exists():
                    st.download_button(
                        "⬇️ clip_metadata.json",
                        meta.read_bytes(),
                        file_name=meta.name,
                        mime="application/json",
                    )

        # Bulk download
        st.subheader("⬇️ Download everything")
        zip_target = Path(result.out_dir) / "clips"
        if st.button("📦 Zip clips/"):
            import zipfile

            zip_path = Path(result.out_dir) / "clips.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in zip_target.rglob("*"):
                    if p.is_file():
                        zf.write(p, arcname=p.relative_to(zip_target.parent))
            with open(zip_path, "rb") as f:
                st.download_button(
                    "⬇️ clips.zip",
                    f.read(),
                    file_name=zip_path.name,
                    mime="application/zip",
                )


# ============================================================================
# Mode 2: Quick per-clip workflow (kept from the original MVP)
# ============================================================================
else:
    st.header("Quick per-clip workflow")
    st.caption(
        "Useful for one-off exports. The Tech Immigrants mode above is "
        "preferred for batch repurposing."
    )

    # --- Step 1 & 2: Input + download ---
    st.subheader("1. Add a video")
    col_url, col_file = st.columns(2)
    with col_url:
        url = st.text_input("YouTube URL", key="q_url")
        if st.button("⬇️ Download from YouTube", use_container_width=True):
            if not url.strip():
                st.warning("Please paste a URL first.")
            else:
                try:
                    with st.spinner("Downloading with yt-dlp…"):
                        info = download_youtube(url.strip())
                    st.session_state.video_path = info["path"]
                    st.session_state.video_title = info["title"]
                    st.session_state.segments = None
                    st.session_state.highlights = None
                    st.success(f"Downloaded: {info['title']}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Download failed: {e}")
    with col_file:
        upload = st.file_uploader(
            "…or upload an MP4",
            type=["mp4", "mov", "mkv", "m4v"],
            key="q_upload",
        )
        if upload is not None and st.button(
            "📁 Use uploaded file", use_container_width=True
        ):
            try:
                info = save_upload(upload)
                st.session_state.video_path = info["path"]
                st.session_state.video_title = info["title"]
                st.session_state.segments = None
                st.session_state.highlights = None
                st.success(f"Loaded: {info['title']}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Upload failed: {e}")

    if st.session_state.video_path:
        st.info(f"Current video: **{st.session_state.video_title}**")
        st.video(st.session_state.video_path)

    # --- Step 2: Transcribe ---
    st.subheader("2. Transcribe (Whisper large-v3)")
    if not st.session_state.video_path:
        st.caption("Add a video first.")
    else:
        if st.button("📝 Transcribe", type="primary", key="q_transcribe"):
            try:
                bar = st.progress(0.0, text="Transcribing…")
                result = transcribe(
                    st.session_state.video_path,
                    language=language.strip() or "fa",
                    progress=lambda p: bar.progress(p, text=f"Transcribing… {int(p*100)}%"),
                )
                st.session_state.transcript = result
                st.session_state.segments = result.to_segments_dicts()
                st.session_state.highlights = None
                bar.empty()
                st.success(f"Transcribed {len(result.segments)} segments.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Transcription failed: {e}")

    if st.session_state.segments:
        with st.expander("View transcript", expanded=False):
            st.text_area(
                "Timestamped transcript",
                transcript_to_text(st.session_state.segments),
                height=300,
            )

    # --- Step 3: Highlights ---
    st.subheader("3. Find highlights")
    if not st.session_state.segments:
        st.caption("Transcribe a video first.")
    else:
        if st.button(
            f"✨ Find top moments with {provider_label}",
            type="primary",
            key="q_highlights",
        ):
            if provider == "claude" and not claude_ok:
                st.error("ANTHROPIC_API_KEY is not set.")
            elif provider == "gpt" and not gpt_ok:
                st.error("OPENAI_API_KEY is not set.")
            else:
                try:
                    with st.spinner(f"Asking {provider_label}…"):
                        highlights = find_highlights(
                            st.session_state.segments, provider=provider, top_k=n_clips
                        )
                    highlights = enrich_social_posts(
                        highlights,
                        guest="",
                        topic="",
                        video_title=st.session_state.video_title or "",
                    )
                    st.session_state.highlights = highlights
                    st.success(f"Found {len(highlights)} highlights.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Highlight detection failed: {e}")

    if st.session_state.highlights:
        rows = []
        for i, h in enumerate(st.session_state.highlights):
            rows.append(
                {
                    "#": i + 1,
                    "Start": format_timestamp(h.start),
                    "End": format_timestamp(h.end),
                    "Score": h.score,
                    "Title": h.title,
                    "Reason": h.reason,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Step 4: Generate a clip ---
    st.subheader("4. Generate a clip")
    if not st.session_state.highlights:
        st.caption("Find highlights first.")
    else:
        labels = [
            f"#{i+1} · {h.score} · {format_timestamp(h.start)}–"
            f"{format_timestamp(h.end)} · {h.title}"
            for i, h in enumerate(st.session_state.highlights)
        ]
        choice = st.selectbox(
            "Choose a highlight",
            range(len(labels)),
            format_func=lambda i: labels[i],
            key="q_choice",
        )
        col_t, col_c = st.columns(2)
        template = col_t.selectbox(
            "Template",
            list(TEMPLATES.keys()),
            format_func=lambda k: TEMPLATES[k]["label"],
            key="q_template",
        )
        crop_mode = col_c.selectbox(
            "Crop / layout",
            list(CROP_MODES.keys()),
            format_func=lambda k: CROP_MODES[k],
            key="q_crop",
        )
        col_b, col_p = st.columns(2)
        burn = col_b.checkbox("Burn Persian subtitles (RTL)", value=True)
        caption_pos = col_p.selectbox(
            "Caption position",
            list(CAPTION_POSITIONS.keys()),
            index=1,
            key="q_caption_pos",
        )

        if st.button("🎞️ Generate Clip", type="primary", key="q_generate"):
            h = st.session_state.highlights[choice]
            try:
                clip_segs = None
                if burn:
                    clip_segs = segments_in_window(
                        st.session_state.segments, h.start, h.end
                    )
                with st.spinner("Exporting clip with FFmpeg (detecting faces…)"):
                    out = generate_clip(
                        st.session_state.video_path,
                        h.start,
                        h.end,
                        out_name=f"clip_{choice+1}_{template}_{crop_mode}",
                        segments=clip_segs,
                        template=template,
                        crop_mode=crop_mode,
                        caption_pos=caption_pos,
                        font=sub_font.strip() or "Geeza Pro",
                    )
                st.session_state.last_clip = out
                st.success("Clip ready!")
            except Exception as e:  # noqa: BLE001
                st.error(f"Clip generation failed: {e}")

    if st.session_state.last_clip:
        st.subheader("Preview")
        st.video(st.session_state.last_clip)
        with open(st.session_state.last_clip, "rb") as f:
            st.download_button(
                "⬇️ Download clip",
                f,
                file_name=st.session_state.last_clip.split("/")[-1],
                mime="video/mp4",
            )
