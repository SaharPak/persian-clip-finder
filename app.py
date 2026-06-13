"""Persian Clip Finder — Streamlit MVP.

Given a YouTube URL or an uploaded MP4, find the best moments for short clips
from Persian podcasts / livestreams, then export them with burned-in subtitles.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core import config
from core.clip import generate_clip
from core.download import download_youtube, save_upload
from core.highlights import find_highlights
from core.subtitles import segments_in_window
from core.transcribe import (
    format_timestamp,
    transcribe,
    transcript_to_text,
)

st.set_page_config(page_title="Persian Clip Finder", page_icon="🎬", layout="wide")

# --- Session defaults ----------------------------------------------------
for key, default in {
    "video_path": None,
    "video_title": None,
    "segments": None,
    "highlights": None,
    "last_clip": None,
}.items():
    st.session_state.setdefault(key, default)


# --- Sidebar -------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

provider_label = st.sidebar.radio(
    "Highlight model",
    ["Claude Sonnet", "GPT-4o"],
    index=0,
)
provider = "claude" if provider_label == "Claude Sonnet" else "gpt"

language = st.sidebar.text_input("Transcription language", value="fa")
sub_font = st.sidebar.text_input("Subtitle font", value="Geeza Pro")

claude_ok = bool(config.ANTHROPIC_API_KEY)
gpt_ok = bool(config.OPENAI_API_KEY)
st.sidebar.markdown("**API keys**")
st.sidebar.write(f"- Anthropic: {'✅' if claude_ok else '❌ missing'}")
st.sidebar.write(f"- OpenAI: {'✅' if gpt_ok else '❌ missing'}")
st.sidebar.caption("Set keys in a .env file (see .env.example).")


# --- Header --------------------------------------------------------------
st.title("🎬 Persian Clip Finder")
st.caption(
    "Find the best Shorts/Reels moments in Persian podcasts & livestreams."
)


# === Step 1 & 2: Input + download =======================================
st.header("1. Add a video")
col_url, col_file = st.columns(2)

with col_url:
    url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
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
    upload = st.file_uploader("…or upload an MP4", type=["mp4", "mov", "mkv", "m4v"])
    if upload is not None and st.button("📁 Use uploaded file", use_container_width=True):
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


# === Step 3: Transcribe =================================================
st.header("2. Transcribe (Whisper large-v3)")

if not st.session_state.video_path:
    st.caption("Add a video first.")
else:
    if st.button("📝 Transcribe", type="primary"):
        try:
            bar = st.progress(0.0, text="Transcribing…")
            result = transcribe(
                st.session_state.video_path,
                language=language.strip() or "fa",
                progress=lambda p: bar.progress(p, text=f"Transcribing… {int(p*100)}%"),
            )
            st.session_state.segments = result["segments"]
            st.session_state.highlights = None
            bar.empty()
            st.success(f"Transcribed {len(result['segments'])} segments.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Transcription failed: {e}")

if st.session_state.segments:
    with st.expander("View transcript", expanded=False):
        st.text_area(
            "Timestamped transcript",
            transcript_to_text(st.session_state.segments),
            height=300,
        )


# === Step 4: Highlights =================================================
st.header("3. Find highlights")

if not st.session_state.segments:
    st.caption("Transcribe a video first.")
else:
    if st.button(f"✨ Find top moments with {provider_label}", type="primary"):
        if provider == "claude" and not claude_ok:
            st.error("ANTHROPIC_API_KEY is not set.")
        elif provider == "gpt" and not gpt_ok:
            st.error("OPENAI_API_KEY is not set.")
        else:
            try:
                with st.spinner(f"Asking {provider_label}…"):
                    st.session_state.highlights = find_highlights(
                        st.session_state.segments, provider=provider
                    )
                st.success(f"Found {len(st.session_state.highlights)} highlights.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Highlight detection failed: {e}")

if st.session_state.highlights:
    rows = []
    for i, h in enumerate(st.session_state.highlights):
        rows.append(
            {
                "#": i + 1,
                "Start": format_timestamp(h["start"]),
                "End": format_timestamp(h["end"]),
                "Score": h["score"],
                "Title": h["title"],
                "Reason": h["reason"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# === Step 5 & 6: Generate clip ==========================================
st.header("4. Generate a clip")

if not st.session_state.highlights:
    st.caption("Find highlights first.")
else:
    labels = [
        f"#{i+1} · {h['score']} · {format_timestamp(h['start'])}–"
        f"{format_timestamp(h['end'])} · {h['title']}"
        for i, h in enumerate(st.session_state.highlights)
    ]
    choice = st.selectbox("Choose a highlight", range(len(labels)),
                          format_func=lambda i: labels[i])
    col_a, col_b = st.columns(2)
    burn = col_a.checkbox("Burn Persian subtitles (RTL)", value=True)
    vertical = col_b.checkbox("9:16 vertical (Shorts/Reels)", value=False)

    if st.button("🎞️ Generate Clip", type="primary"):
        h = st.session_state.highlights[choice]
        try:
            clip_segs = None
            if burn:
                clip_segs = segments_in_window(
                    st.session_state.segments, h["start"], h["end"]
                )
            with st.spinner("Exporting clip with FFmpeg…"):
                out = generate_clip(
                    st.session_state.video_path,
                    h["start"],
                    h["end"],
                    out_name=f"clip_{choice+1}{'_vertical' if vertical else ''}",
                    segments=clip_segs,
                    vertical=vertical,
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
