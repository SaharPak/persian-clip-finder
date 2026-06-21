"""Clip cutting, template reframing, face-aware layout, and subtitle burn-in.

Public entry points
-------------------

* :func:`generate_clip`     – cut a single clip with FFmpeg (back-compat).
* :func:`export_clip_multi` – cut the same moment into multiple aspect
  ratios in a single FFmpeg invocation (much faster than re-encoding once
  per aspect).
* :func:`export_clip_batch` – cut a list of highlights into a list of
  aspect ratios; the unit of work used by the Tech-Immigrants workflow.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from .config import CLIP_DIR, SUBTITLE_DIR
from .layout import detect_people
from .subtitles import SubtitleStyle, build_ass
from .templates import CAPTION_POSITIONS, template_size


# --------------------------------------------------------------------------- #
# FFmpeg discovery
# --------------------------------------------------------------------------- #


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


def ffmpeg_path() -> str:
    """Public helper for callers that want to know the resolved ffmpeg path."""
    return _ensure_ffmpeg()


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


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
    parts: list[str] = []
    for i, p in enumerate(tiles):
        cw, ch, x, y = _crop_rect(W, H, p["cx"], p["cy"], tile_w / tile_h)
        parts.append(
            f"[0:v]crop={cw}:{ch}:{x}:{y},scale={tile_w}:{tile_h},setsar=1[t{i}]"
        )
    parts.append(
        "".join(f"[t{i}]" for i in range(k)) + f"vstack=inputs={k}[vbase]"
    )
    return ";".join(parts), "vbase"


# --------------------------------------------------------------------------- #
# Public API – single clip (back-compat)
# --------------------------------------------------------------------------- #


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
    style: Optional[SubtitleStyle] = None,
) -> str:
    """Cut [start, end] and render to a template with face-aware cropping."""
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
        sub_style = style or SubtitleStyle(font=font, subtitle_position=caption_pos)
        sub_style.subtitle_position = caption_pos
        ass_path = build_ass(
            segments,
            SUBTITLE_DIR / f"{out_name}.ass",
            width=ow,
            height=oh,
            style=sub_style,
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


# --------------------------------------------------------------------------- #
# Public API – multi-aspect batch
# --------------------------------------------------------------------------- #


@dataclass
class AspectExport:
    """Result of exporting a single clip in a single aspect ratio."""

    template: str
    out_path: str
    duration: float


# Map short names used by the CLI to internal template keys.
ASPECT_ALIASES = {
    "9x16": "reels",
    "vertical": "reels",
    "short": "short",
    "shorts": "short",
    "1x1": "square",
    "square": "square",
    "16x9": "youtube",
    "youtube": "youtube",
    "horizontal": "youtube",
    "original": "original",
}


def resolve_aspects(aspects: Iterable[str]) -> list[str]:
    """Normalize a list of aspect aliases into TEMPLATES keys (deduped, ordered)."""
    out: list[str] = []
    for a in aspects:
        key = ASPECT_ALIASES.get(str(a).strip().lower())
        if key and key not in out:
            out.append(key)
    return out or ["reels"]


def export_clip_multi(
    source: str,
    start: float,
    end: float,
    out_dir: str | Path,
    aspects: Iterable[str],
    *,
    base_name: str = "clip",
    segments: list[dict] | None = None,
    crop_mode: str = "auto",
    style: Optional[SubtitleStyle] = None,
) -> list[AspectExport]:
    """Cut ``[start, end]`` once and re-encode into each requested aspect.

    This is significantly faster than calling :func:`generate_clip` once per
    aspect because we run ``-ss``/``-t`` once on the input and only fork the
    video filter graph for the per-aspect scaling.
    """
    ffmpeg = _ensure_ffmpeg()
    out_paths = _aspect_output_paths(out_dir, base_name, aspects)
    aspect_keys = list(out_paths.keys())
    if not aspect_keys:
        return []

    W, H = _probe_dimensions(source)
    duration = max(0.1, end - start)

    ass_path_for_aspect: dict[str, str] = {}
    for i, tpl in enumerate(aspect_keys):
        ow, oh = template_size(tpl, (W, H))
        ow, oh = _even(ow), _even(oh)
        mode, regions = _resolve_layout(source, start, end, tpl, crop_mode, W, H)
        _, label = _build_filter_complex(W, H, ow, oh, regions, mode)
        if segments is not None:
            ass_path = build_ass(
                segments,
                Path(out_paths[tpl]).with_suffix(".ass"),
                width=ow,
                height=oh,
                style=style,
            )
            ass_path_for_aspect[tpl] = ass_path

    filter_complex_parts: list[str] = [
        f"[0:v]split={len(aspect_keys)}" + "".join(f"[v{i}]" for i in range(len(aspect_keys)))
    ]
    for i, tpl in enumerate(aspect_keys):
        ow, oh = template_size(tpl, (W, H))
        ow, oh = _even(ow), _even(oh)
        mode, regions = _resolve_layout(source, start, end, tpl, crop_mode, W, H)
        base_fc, label = _build_filter_complex(W, H, ow, oh, regions, mode)
        # Strip leading "[0:v]" because we feed from [vi]
        if base_fc.startswith("[0:v]"):
            base_fc = base_fc[len("[0:v]"):]
        elif base_fc.startswith("[0:v]"):
            base_fc = base_fc[len("[0:v]"):]
        out_label = f"vp{i}"
        if segments is not None and tpl in ass_path_for_aspect:
            ass_path = ass_path_for_aspect[tpl]
            base_fc += f",ass={_escape_for_filter(ass_path)}"
        filter_complex_parts.append(f"[v{i}]{base_fc}[{out_label}]")

    cmd = [ffmpeg, "-y", "-ss", f"{start}", "-i", source, "-t", f"{duration}"]
    for i, tpl in enumerate(aspect_keys):
        cmd.extend(["-map", f"[vp{i}]", "-map", "0:a?"])
    cmd.extend(
        [
            "-filter_complex", ";".join(filter_complex_parts),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart",
        ]
    )
    for tpl in aspect_keys:
        cmd.append(str(out_paths[tpl]))

    _run(cmd)
    return [
        AspectExport(template=tpl, out_path=str(out_paths[tpl]), duration=duration)
        for tpl in aspect_keys
    ]


def _aspect_output_paths(
    out_dir: str | Path, base_name: str, aspects: Iterable[str]
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for tpl in aspects:
        paths[tpl] = out_dir / f"{base_name}_{tpl}.mp4"
    return paths


def export_clip_batch(
    source: str,
    highlights: list[dict],
    out_root: str | Path,
    aspects: Iterable[str] = ("reels", "square", "youtube"),
    *,
    full_segments: list[dict] | None = None,
    crop_mode: str = "auto",
    style: Optional[SubtitleStyle] = None,
    name_fn=None,
) -> list[dict]:
    """Cut every highlight into every aspect ratio. Returns a list of exports.

    Each returned dict has keys: ``clip_name``, ``template``, ``out_path``,
    ``start``, ``end``, ``duration``, ``ass_path``.
    """
    aspect_keys = resolve_aspects(aspects)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for i, h in enumerate(highlights, start=1):
        clip_name = (name_fn(i) if name_fn else f"clip_{i:03d}")
        clip_dir = out_root / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)
        # Slice segments relative to this clip's window for caption burn-in.
        rel_segs = (
            segments_in_window_local(full_segments, h["start"], h["end"])
            if full_segments else None
        )
        exports = export_clip_multi(
            source,
            h["start"],
            h["end"],
            out_dir=clip_dir,
            aspects=aspect_keys,
            base_name=clip_name,
            segments=rel_segs,
            crop_mode=crop_mode,
            style=style,
        )
        for e in exports:
            ass_path = clip_dir / f"{clip_name}_{e.template}.ass"
            results.append(
                {
                    "clip_name": clip_name,
                    "template": e.template,
                    "out_path": str(e.out_path),
                    "start": h["start"],
                    "end": h["end"],
                    "duration": e.duration,
                    "ass_path": str(ass_path) if ass_path.exists() else "",
                }
            )
    return results


def segments_in_window_local(
    segments: list[dict], start: float, end: float
) -> list[dict]:
    """Time-shift segments overlapping [start, end] to clip start."""
    out: list[dict] = []
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        s = max(seg["start"], start) - start
        e = min(seg["end"], end) - start
        if e <= s:
            continue
        out.append({"start": s, "end": e, "text": seg["text"]})
    return out
