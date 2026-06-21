"""Video acquisition: download from YouTube (yt-dlp) or save an upload."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

try:
    from yt_dlp import YoutubeDL  # type: ignore
except Exception:  # noqa: BLE001
    YoutubeDL = None  # type: ignore

from .config import DOWNLOAD_DIR


def _slug(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def download_youtube(url: str, progress_hook=None) -> dict:
    """Download a YouTube video as MP4. Returns dict with path and title."""
    out_tmpl = str(DOWNLOAD_DIR / f"{_slug(url)}.%(ext)s")
    opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": out_tmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if progress_hook is not None:
        opts["progress_hooks"] = [progress_hook]

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # After merge the real file is the mp4 next to the slug.
        path = Path(DOWNLOAD_DIR / f"{_slug(url)}.mp4")
        if not path.exists():
            # Fall back to whatever yt-dlp reports.
            path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")

    return {
        "path": str(path),
        "title": info.get("title", "video"),
        "duration": info.get("duration"),
    }


def save_upload(uploaded_file) -> dict:
    """Persist a Streamlit UploadedFile to disk and return its path."""
    digest = _slug(uploaded_file.name + str(uploaded_file.size))
    dest = DOWNLOAD_DIR / f"upload_{digest}.mp4"
    with open(dest, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)
    return {"path": str(dest), "title": uploaded_file.name, "duration": None}
