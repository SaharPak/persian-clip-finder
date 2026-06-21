"""Generate RTL Persian subtitles in ASS format.

Two modes are supported:

* ``static`` – one line per segment, default style.
* ``word``   – one line per word, with each word timed individually
  (only useful when word-level timestamps are present).
* ``phrase_highlight`` – same density as ``static`` but key words inside the
  phrase are coloured (yellow by default). The current implementation
  highlights the **last 1-3 words** of each line, which is a strong visual
  hook for short-form captions.

The :class:`SubtitleStyle` dataclass exposes the same knobs that the UI and
the CLI tweak (font, size, position, highlight colour, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# Bidi control chars to force right-to-left embedding for Persian text.
RLE = "\u202b"  # Right-to-Left Embedding
PDF = "\u202c"  # Pop Directional Formatting


# --------------------------------------------------------------------------- #
# Public data model
# --------------------------------------------------------------------------- #


@dataclass
class SubtitleStyle:
    """A serialisable bundle of subtitle / caption settings."""

    mode: str = "static"             # static | word | phrase_highlight
    font: str = "Geeza Pro"
    font_size: int = 0               # 0 = auto from output height
    outline: int = 0                 # 0 = auto
    margin_h: int = 0                # 0 = auto
    margin_v: int = 0                # 0 = auto
    subtitle_position: str = "lower" # bottom | lower | middle | top
    highlight_color: str = "&H0000FFFF"  # BGR; default = yellow
    secondary_color: str = "&H00FF0000"  # BGR; default = blue
    max_chars_per_line: int = 40
    max_lines: int = 2

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "SubtitleStyle":
        valid_keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in valid_keys})


DEFAULT_STYLE = SubtitleStyle()


# --------------------------------------------------------------------------- #
# ASS header / helpers
# --------------------------------------------------------------------------- #


ASS_HEADER_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Persian,{font},{size},&H00FFFFFF,{secondary},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},1,2,{margin_h},{margin_h},{margin_v},1
Style: PersianHi,{font},{size},{highlight},&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},1,2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


# --------------------------------------------------------------------------- #
# Public helpers (re-used by clip.py)
# --------------------------------------------------------------------------- #


def segments_in_window(
    segments: list[dict], start: float, end: float
) -> list[dict]:
    """Return segments overlapping [start, end], time-shifted to clip start."""
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


# --------------------------------------------------------------------------- #
# ASS builders
# --------------------------------------------------------------------------- #


def _resolve_dims(width: int, height: int, style: SubtitleStyle) -> dict:
    if style.font_size <= 0:
        style.font_size = max(22, round(height * 0.05))
    if style.outline <= 0:
        style.outline = max(2, round(height * 0.004))
    if style.margin_h <= 0:
        style.margin_h = round(width * 0.05)
    if style.margin_v <= 0:
        # Subtitle position
        position_frac = {
            "bottom": 0.08,
            "lower": 0.14,
            "middle": 0.45,
            "top": 0.82,
        }.get(style.subtitle_position, 0.14)
        style.margin_v = round(height * position_frac)
    return {
        "size": style.font_size,
        "outline": style.outline,
        "margin_h": style.margin_h,
        "margin_v": style.margin_v,
    }


def _wrap_rtl(text: str) -> str:
    text = (text or "").replace("\n", " ").strip()
    return f"{RLE}{text}{PDF}"


def _static_dialogue(seg: dict, dims: dict, style: SubtitleStyle) -> str:
    text = _wrap_rtl(seg["text"])
    style_name = "Persian"
    if style.mode == "phrase_highlight":
        text = _highlight_phrase(text)
        # The highlight control tags use {\highlight&H00FFFF&} which changes the
        # secondary colour for the rest of the line. We keep the base style
        # name Persian and rely on the override.
    return (
        f"Dialogue: 0,{_ass_time(seg['start'])},{_ass_time(seg['end'])},"
        f"{style_name},0,0,0,,{text}"
    )


def _highlight_phrase(rle_text: str) -> str:
    """Wrap the last 1-3 words of an RLE-wrapped line with a colour override.

    The simplest reliable approach in ASS is to use ``\\an`` style override
    tags; the cleanest visual is achieved by changing the *secondary* colour
    for the trailing words. We approximate this by inserting a primary-colour
    override (yellow) just before the last 1-3 whitespace-separated tokens,
    then resetting it.
    """
    # Pull the Persian text out of the RLE/PDF wrapper, modify, and rewrap.
    inner = rle_text.strip(RLE + PDF)
    tokens = inner.split(" ")
    if len(tokens) < 2:
        return rle_text
    n = min(3, max(1, len(tokens) // 3))
    head = " ".join(tokens[:-n])
    tail = " ".join(tokens[-n:])
    # Primary colour override; 0x0000FFFF = yellow (ABGR -> &HAABBGGRR)
    highlighted = f"{head} {{\\1c&H0000FFFF&}}{tail}"
    return f"{RLE}{highlighted}{PDF}"


def _word_dialogues(seg: dict, style: SubtitleStyle) -> list[str]:
    """One ASS line per word. Falls back to static line if no words."""
    words = seg.get("words") or []
    if not words:
        return [_static_dialogue(seg, {}, style)]
    out: list[str] = []
    for w in words:
        token = (w.get("word") or "").strip()
        if not token:
            continue
        out.append(
            f"Dialogue: 0,{_ass_time(w['start'])},{_ass_time(w['end'])},"
            f"Persian,,0,0,0,,{_wrap_rtl(token)}"
        )
    return out


def build_ass(
    segments: list[dict],
    out_path: str | Path,
    width: int = 1280,
    height: int = 720,
    font: str = "Geeza Pro",
    font_size: int | None = None,
    margin_v: int | None = None,
    style: Optional[SubtitleStyle] = None,
) -> str:
    """Write an ASS subtitle file sized to the target output resolution.

    Back-compat: ``font`` and ``font_size`` may still be passed positionally.
    """
    out_path = Path(out_path)
    style = style or SubtitleStyle(font=font)
    if font and font != style.font:
        style.font = font
    if font_size is not None:
        style.font_size = int(font_size)
    if margin_v is not None:
        style.margin_v = int(margin_v)

    dims = _resolve_dims(width, height, style)
    header = ASS_HEADER_TEMPLATE.format(
        width=width,
        height=height,
        font=style.font,
        secondary=style.secondary_color,
        highlight=style.highlight_color,
        **dims,
    )
    lines = [header]
    for seg in segments:
        if style.mode == "word":
            lines.extend(_word_dialogues(seg, style))
        else:
            lines.append(_static_dialogue(seg, dims, style))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
