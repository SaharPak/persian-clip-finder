"""Generate RTL Persian subtitles in ASS format."""

from __future__ import annotations

from pathlib import Path

# Bidi control chars to force right-to-left embedding for Persian text.
RLE = "\u202b"  # Right-to-Left Embedding
PDF = "\u202c"  # Pop Directional Formatting

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Persian,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,40,40,60,1

[Events]
Format: Layer, Start, End, Style, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


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


def build_ass(
    segments: list[dict],
    out_path: str | Path,
    font: str = "Geeza Pro",
    size: int = 48,
) -> str:
    """Write an ASS subtitle file for the given (clip-relative) segments."""
    out_path = Path(out_path)
    lines = [ASS_HEADER.format(font=font, size=size)]
    for seg in segments:
        text = seg["text"].replace("\n", " ").strip()
        text = f"{RLE}{text}{PDF}"
        lines.append(
            f"Dialogue: 0,{_ass_time(seg['start'])},{_ass_time(seg['end'])},"
            f"Persian,,0,0,0,,{text}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
