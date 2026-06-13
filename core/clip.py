"""Clip cutting, template reframing, face-aware layout, and subtitle burn-in."""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .config import CLIP_DIR, SUBTITLE_DIR
from .layout import detect_people
from .subtitles import build_ass
from .templates import CAPTION_POSITIONS, template_size


def _has_ass_filter(ffmpeg: str) -> bool:
    """True if this ffmpeg build was compiled with libass (ass/subtitles)."""
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
        ).stdout
        return any(line.split()[1:2] == ["ass"] for line in out.splitlines())
    except Exception:
        return False


@lru_cache(maxsize=1)
def _ensure_ffmpeg() -> str:
    """Return an ffmpeg that supports libass (needed for subtitle burn-in)."""
    system = shutil.which("ffmpeg")
    if system and _has_ass_filter(system):
        return system
    try:
        from static_ffmpeg import run

        static_path, _ = run.get_or_fetch_platform_executables_else_raise()
        if _has_ass_filter(static_path):
            return static_path
    except Exception:
        pass
    if system:
        return system
    raise RuntimeError("FFmpeg not found. Install it or `pip install static-ffmpeg`.")


@lru_cache(maxsize=1)
def _ensure_ffprobe() -> str | None:
    probe = shutil.which("ffprobe")
    if probe:
        return probe
    try:
        from static_ffmpeg import run

        _, static_probe = run.get_or_fetch_platform_executables_else_raise()
        return static_probe
    except Exception:
        return None


def _probe_dimensions(source: str) -> tuple[int, int]:
    """Return (width, height) of the source video, defaulting to 1280x720."""
    probe = _ensure_ffprobe()
    if not probe:
        return 1280, 720
    try:
        out = subprocess.run(
            [
                probe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "json", source,
            ],
            capture_output=True,
            text=True,
        ).stdout
        s = json.loads(out)["streams"][0]
        return int(s["width"]), int(s["height"])
    except Exception:
        return 1280, 720


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (code {proc.returncode}):\n{proc.stderr[-2000:]}"
        )


def _escape_for_filter(path: str) -> str:
    p = str(Path(path).resolve())
    return p.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _even(n: float) -> int:
    return int(n) - (int(n) % 2)


def _crop_rect(W: int, H: int, cx: float, cy: float, ar: float):
    """Largest crop of aspect ``ar`` (w/h) fitting in WxH, centered on (cx,cy)."""
    if W / H > ar:
        ch = H
        cw = ch * ar
    else:
        cw = W
        ch = cw / ar
    cw = min(cw, W)
    ch = min(ch, H)
    x = min(max(cx - cw / 2, 0), W - cw)
    y = min(max(cy - ch / 2, 0), H - ch)
    return _even(cw), _even(ch), _even(x), _even(y)


def _resolve_layout(
    source, start, end, template, crop_mode, W, H
) -> tuple[str, list[dict]]:
    """Decide the effective layout mode and the person regions to use."""
    if template == "original":
        return "none", []
    if crop_mode == "fit":
        return "fit", []

    people = detect_people(source, start, end)

    if crop_mode == "single":
        return "single", people
    if crop_mode == "split2":
        return ("split2", people[:2]) if len(people) >= 2 else ("single", people)
    if crop_mode == "grid3":
        if len(people) >= 3:
            return "grid3", people[:3]
        if len(people) >= 2:
            return "split2", people[:2]
        return "single", people

    # auto
    if len(people) >= 3:
        return "grid3", people[:3]
    if len(people) == 2:
        return "split2", people
    return "single", people


def _build_filter_complex(W, H, ow, oh, regions, mode) -> tuple[str, str]:
    """Return (filter_complex, output_label) producing a `[vbase]` stream."""
    if mode == "none":
        return "[0:v]setsar=1[vbase]", "vbase"

    if mode == "fit":
        return (
            f"[0:v]scale={ow}:{oh}:force_original_aspect_ratio=decrease,"
            f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[vbase]",
            "vbase",
        )

    if mode == "single":
        main = max(regions, key=lambda p: p["w"] * p["h"]) if regions else {
            "cx": W / 2,
            "cy": H / 2,
        }
        cw, ch, x, y = _crop_rect(W, H, main["cx"], main["cy"], ow / oh)
        return (
            f"[0:v]crop={cw}:{ch}:{x}:{y},scale={ow}:{oh},setsar=1[vbase]",
            "vbase",
        )

    # stacked: split2 / grid3
    k = 2 if mode == "split2" else 3
    tiles = regions[:k]
    tile_w = ow
    tile_h = _even(oh / k)
    ar = tile_w / tile_h

    parts = [f"[0:v]split={k}" + "".join(f"[s{i}]" for i in range(k))]
    for i, r in enumerate(tiles):
        cw, ch, x, y = _crop_rect(W, H, r["cx"], r["cy"], ar)
        parts.append(
            f"[s{i}]crop={cw}:{ch}:{x}:{y},scale={tile_w}:{tile_h},setsar=1[t{i}]"
        )
    parts.append(
        "".join(f"[t{i}]" for i in range(k)) + f"vstack=inputs={k}[vbase]"
    )
    return ";".join(parts), "vbase"


def generate_clip(
    source: str,
    start: float,
    end: float,
    out_name: str,
    segments: list[dict] | None = None,
    template: str = "reels",
    crop_mode: str = "auto",
    caption_pos: str = "lower",
    font: str = "Geeza Pro",
) -> str:
    """Cut [start, end] and render to a template with face-aware cropping.

    - ``template``: one of TEMPLATES keys (reels/short/square/youtube/original).
    - ``crop_mode``: auto/single/split2/grid3/fit (see CROP_MODES).
    - ``segments`` (clip-relative): burned in as RTL Persian subtitles.
    Returns the path to the generated MP4.
    """
    ffmpeg = _ensure_ffmpeg()
    out_path = CLIP_DIR / f"{out_name}.mp4"
    duration = max(0.1, end - start)

    W, H = _probe_dimensions(source)
    ow, oh = template_size(template, (W, H))
    ow, oh = _even(ow), _even(oh)

    mode, regions = _resolve_layout(source, start, end, template, crop_mode, W, H)
    fc, label = _build_filter_complex(W, H, ow, oh, regions, mode)

    map_label = label
    if segments:
        frac = CAPTION_POSITIONS.get(caption_pos, 0.14)
        ass_path = build_ass(
            segments,
            SUBTITLE_DIR / f"{out_name}.ass",
            width=ow,
            height=oh,
            font=font,
            margin_v=int(oh * frac),
        )
        fc += f";[{label}]ass={_escape_for_filter(ass_path)}[vout]"
        map_label = "vout"

    cmd = [
        ffmpeg, "-y", "-ss", f"{start}", "-i", source, "-t", f"{duration}",
        "-filter_complex", fc,
        "-map", f"[{map_label}]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    _run(cmd)
    return str(out_path)
