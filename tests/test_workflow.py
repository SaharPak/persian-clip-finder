"""Tests for the workflow / CLI glue.

These tests do not need a real video. We test:

* the WorkflowConfig dataclass
* the file/directory creation logic of the workflow
* the CLI ``version`` subcommand
* the CLI help output
* the config.yaml loading
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


from persian_clip_finder.workflow import (
    WorkflowConfig,
    _safe_slug,
    write_clip_metadata,
    write_clip_social_posts,
    write_clip_transcript,
)
from persian_clip_finder.highlights import Highlight, Scores


def test_safe_slug():
    assert _safe_slug("Tech Immigrants / DevOps!") == "tech-immigrants-devops"
    assert _safe_slug("") == "video"
    assert _safe_slug("نوار فارسی") == "نوار-فارسی"


def test_workflow_config_to_dict():
    cfg = WorkflowConfig()
    d = cfg.to_dict()
    assert d["language"] == "fa"
    assert "reels" in d["aspects"]


def test_write_clip_artifacts(tmp_path: Path):
    segs = [
        {"start": 0.0, "end": 5.0, "text": "شروع"},
        {"start": 5.0, "end": 10.0, "text": "ادامه"},
        {"start": 10.0, "end": 15.0, "text": "پایان"},
    ]
    clip_dir = tmp_path / "clip_001"
    write_clip_transcript(clip_dir, segs, 4.0, 12.0)
    assert (clip_dir / "clip_transcript.txt").exists()
    text = (clip_dir / "clip_transcript.txt").read_text(encoding="utf-8")
    assert "ادامه" in text

    h = Highlight(
        start=4.0, end=12.0, title="تست", hook="شروع", reason="آزمایش"
    )
    write_clip_metadata(clip_dir, h, guest="مهمان", topic="موضوع", video_title="ویدیو")
    meta = json.loads((clip_dir / "clip_metadata.json").read_text(encoding="utf-8"))
    assert meta["title"] == "تست"
    assert meta["guest"] == "مهمان"

    write_clip_social_posts(clip_dir, h)
    md = (clip_dir / "social_posts.md").read_text(encoding="utf-8")
    assert "## یوتیوب Shorts" in md


def test_cli_version(capsys):
    from persian_clip_finder.cli import main

    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "persian-clip-finder" in out


def test_cli_help(capsys):
    from persian_clip_finder.cli import main

    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    captured = capsys.readouterr()
    assert "process" in captured.out or "process" in captured.err


def test_cli_process_requires_url_or_input(capsys):
    from persian_clip_finder.cli import main

    rc = main(["process"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--url" in err or "--input" in err
