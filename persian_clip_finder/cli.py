"""Command-line entry point: ``python -m persian_clip_finder …``.

Sub-commands
------------

* ``process`` – run the full Tech Immigrants pipeline on a local file or a
  YouTube URL.
* ``transcribe`` – only transcribe a file (writes transcript.json next to it).
* ``version`` – print the package version.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .download import download_youtube
from .transcribe import transcribe
from .workflow import WorkflowConfig, process_video


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="persian-clip-finder",
        description="Turn a long Persian video into ready-to-post short clips.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_version = sub.add_parser("version", help="print the version and exit")
    p_version.set_defaults(_fn=_cmd_version)

    p_tr = sub.add_parser("transcribe", help="transcribe a local video")
    p_tr.add_argument("--input", "-i", required=True, help="path to input video")
    p_tr.add_argument("--language", default="fa")
    p_tr.add_argument("--out", "-o", help="output transcript.json path")
    p_tr.set_defaults(_fn=_cmd_transcribe)

    p_pr = sub.add_parser(
        "process",
        help="run the full Tech Immigrants pipeline on a local file or YouTube URL",
    )
    p_pr.add_argument("--url", help="YouTube URL to download first")
    p_pr.add_argument(
        "--input", "-i", help="path to a local MP4 (mutually exclusive with --url)"
    )
    p_pr.add_argument("--guest", default="", help="guest name (for social posts)")
    p_pr.add_argument("--topic", default="", help="topic / title of the live")
    p_pr.add_argument("--video-title", default=None, help="display title")
    p_pr.add_argument("--clips", type=int, default=8, help="target number of clips")
    p_pr.add_argument(
        "--formats",
        default="9x16,1x1,16x9",
        help="comma-separated aspects: 9x16,1x1,16x9,short,original",
    )
    p_pr.add_argument(
        "--subtitle-mode",
        default="phrase_highlight",
        choices=["static", "word", "phrase_highlight"],
    )
    p_pr.add_argument(
        "--subtitle-position",
        default="lower",
        choices=["bottom", "lower", "middle", "top"],
    )
    p_pr.add_argument("--font", default="Geeza Pro")
    p_pr.add_argument(
        "--crop-mode",
        default="auto",
        choices=["auto", "single", "split2", "grid3", "fit"],
    )
    p_pr.add_argument(
        "--provider",
        default=None,
        choices=["claude", "gpt", "offline"],
        help="LLM provider; default: auto-detect (claude > gpt > offline)",
    )
    p_pr.add_argument("--language", default="fa")
    p_pr.add_argument(
        "--output",
        "-o",
        default="outputs/",
        help="root folder for outputs (one sub-folder per video)",
    )
    p_pr.add_argument(
        "--no-subtitles", action="store_true", help="do not burn subtitles"
    )
    p_pr.add_argument(
        "--config", default=None, help="path to a config.yaml to read defaults from"
    )
    p_pr.add_argument(
        "--no-word-timestamps", action="store_true",
        help="disable word-level timestamps (faster transcription)"
    )
    p_pr.set_defaults(_fn=_cmd_process)

    return p


def _cmd_version(_args) -> int:
    print(f"persian-clip-finder {__version__}")
    return 0


def _cmd_transcribe(args) -> int:
    src = Path(args.input).expanduser()
    transcript = transcribe(str(src), language=args.language)
    out = Path(args.out) if args.out else src.with_suffix(".transcript.json")
    transcript.save(out)
    print(f"wrote transcript with {len(transcript.segments)} segments to {out}")
    return 0


def _load_yaml_defaults(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        print(f"warning: config file {p} does not exist; ignoring", file=sys.stderr)
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        print("warning: PyYAML is not installed; ignoring config file", file=sys.stderr)
        return {}
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        print(f"warning: could not parse {p}: {e}", file=sys.stderr)
        return {}


def _cmd_process(args) -> int:
    if not args.url and not args.input:
        print("error: either --url or --input is required", file=sys.stderr)
        return 2
    if args.url and args.input:
        print("error: --url and --input are mutually exclusive", file=sys.stderr)
        return 2

    yaml_defaults = _load_yaml_defaults(args.config)

    cfg = WorkflowConfig(
        language=(args.language or yaml_defaults.get("language", "fa")),
        provider=(args.provider or yaml_defaults.get("provider")),
        top_k=int(args.clips),
        aspects=tuple(a.strip() for a in args.formats.split(",") if a.strip()),
        subtitle_mode=args.subtitle_mode,
        subtitle_position=args.subtitle_position,
        font=args.font,
        crop_mode=args.crop_mode,
        burn_subtitles=not args.no_subtitles,
        word_timestamps=not args.no_word_timestamps,
    )

    def _progress(stage: str, p: float) -> None:
        bar = int(round(p * 30))
        sys.stdout.write(f"\r[{stage:<11}] [{('#' * bar).ljust(30)}] {int(p*100):3d}%")
        sys.stdout.flush()
        if p >= 1.0:
            sys.stdout.write("\n")

    if args.url:
        print(f"downloading {args.url}…")
        info = download_youtube(args.url)
        source = info["path"]
        video_title = args.video_title or info.get("title")
    else:
        source = str(Path(args.input).expanduser())
        video_title = args.video_title

    result = process_video(
        source=source,
        out_root=args.output,
        cfg=cfg,
        guest=args.guest,
        topic=args.topic,
        video_title=video_title,
        progress=_progress,
    )
    print(f"\nwrote {len(result.exports)} clip files into {result.out_dir}")
    summary = {
        "out_dir": str(result.out_dir),
        "clips": [
            {
                "name": exp["clip_name"],
                "template": exp["template"],
                "path": exp["out_path"],
            }
            for exp in result.exports
        ],
        "social_posts": {k: str(v) for k, v in result.social_posts.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args._fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
