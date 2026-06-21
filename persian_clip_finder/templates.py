"""Output templates (aspect/size) and crop-layout options."""

from __future__ import annotations

# Each template defines the output canvas. ``size`` of None means "use the
# source video's own dimensions" (Original).
TEMPLATES: dict[str, dict] = {
    "reels": {"label": "Instagram Reels / Story (9:16)", "size": (1080, 1920)},
    "short": {"label": "YouTube Short (9:16)", "size": (1080, 1920)},
    "square": {"label": "Square (1:1)", "size": (1080, 1080)},
    "youtube": {"label": "YouTube (16:9)", "size": (1920, 1080)},
    "original": {"label": "Original (keep full frame)", "size": None},
}

# Crop / layout strategies.
CROP_MODES: dict[str, str] = {
    "auto": "Auto — detect faces & arrange",
    "single": "Single speaker (center on main face)",
    "split2": "Two speakers (stacked top / bottom)",
    "grid3": "Three speakers (stacked)",
    "fit": "No crop (fit & pad)",
}

# Where captions sit, as a fraction of output height measured from the bottom.
CAPTION_POSITIONS: dict[str, float] = {
    "bottom": 0.08,
    "lower": 0.14,
    "center": 0.45,
    "top": 0.82,
}


def template_size(template: str, source_dims: tuple[int, int]) -> tuple[int, int]:
    """Resolve the output (width, height) for a template."""
    spec = TEMPLATES.get(template, TEMPLATES["reels"])
    return spec["size"] or source_dims
