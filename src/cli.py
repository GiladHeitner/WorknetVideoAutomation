"""CLI entrypoint.

Simplest invocations:

    python -m src script.txt screen.mp4                    # single source
    python -m src script.txt --videos-dir examples/        # one video per [Cut to ...] section
    python -m src script.txt --videos a.mp4 b.mp4 c.mp4    # explicit per-section list

Cue overlay images are auto-discovered from ``assets/``. Use ``--dry-run``
to preview before any OpenAI calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .assets import cue_resolution_report, discover_cue_assets
from .cue_finder import CueFinderConfig
from .parser import parse_script_file
from .pipeline import run
from .preflight import PreflightError, check
from .sections import DEFAULT_MARKER, reconcile_videos, split_sections


VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".mkv", ".webm")


def _load_video_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    if Path(raw).is_file():
        return json.loads(Path(raw).read_text(encoding="utf-8"))
    return json.loads(raw)


def _discover_videos_dir(dir_path: str) -> list[str]:
    d = Path(dir_path)
    if not d.is_dir():
        raise PreflightError(f"videos-dir not a directory: {dir_path}")
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not files:
        raise PreflightError(f"no videos found in {dir_path}")
    return [str(p) for p in files]


def _resolve_videos(args: argparse.Namespace) -> tuple[str | None, list[str] | None, list[str] | None]:
    """Return (single_video, per_beat_list, section_list)."""
    if args.videos:
        return None, None, args.videos
    if args.videos_dir:
        return None, None, _discover_videos_dir(args.videos_dir)
    if args.video_list:
        return None, _load_video_list(args.video_list), None
    return args.video, None, None


def _override_assets(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    if Path(raw).is_file():
        return json.loads(Path(raw).read_text(encoding="utf-8"))
    return json.loads(raw)


def _dry_run(args: argparse.Namespace, section_videos: list[str] | None) -> int:
    beats = parse_script_file(args.script)
    discovered = discover_cue_assets(beats, args.cues_dir)
    overrides = _override_assets(args.cue_assets)
    discovered.update(overrides)

    spoken = [b for b in beats if b["type"] == "spoken_text"]
    chars = sum(len(b["content"]) for b in spoken)

    print("Script preview")
    print(f"  spoken beats : {len(spoken)}")
    print(f"  total chars  : {chars}")
    print(f"  visual cues  : {sum(1 for b in beats if b['type'] == 'visual_cue')}")
    print(f"  est. tts cost: ~${chars * 15 / 1_000_000:.4f} (tts-1 baseline)")
    print()

    if section_videos:
        try:
            sections = reconcile_videos(split_sections(beats), len(section_videos))
        except ValueError as e:
            print(f"Section check FAILED: {e}")
        else:
            cue_mode = "auto-cue ON" if args.auto_cue else "auto-cue OFF"
            print(f"Sections ({len(sections)}, matches {len(section_videos)} videos, {cue_mode})")
            for i, (sec, vid) in enumerate(zip(sections, section_videos), 1):
                cue = sec["start_cue"]
                marker = f"[{cue['content']}]" if cue else "(intro)"
                n_spoken = sum(1 for b in sec["beats"] if b["type"] == "spoken_text")
                inner_cues = [
                    b["content"] for b in sec["beats"]
                    if b.get("type") == "visual_cue"
                ]
                print(f"  {i}. {marker}")
                print(f"     video : {vid}")
                print(f"     beats : {n_spoken} spoken, {len(inner_cues)} bracket actions to time")
                for c in inner_cues:
                    print(f"       - [{c}]")
            print()

    cue_rows = cue_resolution_report(beats, args.cues_dir)
    for row in cue_rows:
        if row["cue"] in overrides:
            row["asset"] = overrides[row["cue"]]
    print("Cue overlay resolution (section-start cues are consumed, not overlaid)")
    if not cue_rows:
        print("  (no visual_cue beats)")
    for row in cue_rows:
        is_section = section_videos and row["cue"].lower().lstrip().startswith("cut to")
        status = "SECTION MARKER" if is_section else (row["asset"] or "no overlay image (drop one in assets/)")
        print(f"  [{row['cue']}] -> {status}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        prog="worknet-video",
        description="Script + video(s) → narrated mp4 with subs and cue overlays.",
    )
    p.add_argument("script", help="Path to script text file")
    p.add_argument(
        "video",
        nargs="?",
        help="Single source video (legacy mode)",
    )

    group = p.add_argument_group("multi-video modes")
    group.add_argument(
        "--videos",
        nargs="+",
        help="Ordered list of section videos, one per [Cut to ...] section",
    )
    group.add_argument(
        "--videos-dir",
        help="Directory; takes all video files sorted by filename as section videos",
    )
    group.add_argument(
        "--video-list",
        help="JSON file or inline JSON array of per-spoken-beat video paths",
    )

    p.add_argument("--out", default="out", help="Output directory (default: out)")
    p.add_argument("--voice", default="alloy", help="OpenAI TTS voice")
    p.add_argument("--tts-model", default="tts-1")
    p.add_argument(
        "--cues-dir",
        default="assets",
        help="Directory of cue overlay images (default: assets)",
    )
    p.add_argument(
        "--cue-assets",
        help="Optional JSON file/string mapping cue text → image path (overrides auto-discovery)",
    )
    p.add_argument(
        "--subs",
        choices=["vtt", "srt", "none"],
        default="vtt",
        help="Subtitle sidecar format (default: vtt)",
    )
    p.add_argument(
        "--burn-subs",
        dest="burn_subs",
        action="store_true",
        default=True,
        help="Burn subtitles into final video (default: on)",
    )
    p.add_argument(
        "--no-burn-subs",
        dest="burn_subs",
        action="store_false",
        help="Skip burning subtitles into final video",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--auto-cue",
        dest="auto_cue",
        action="store_true",
        default=True,
        help="Infer source-video timestamps for each bracketed action (default: on)",
    )
    p.add_argument(
        "--no-auto-cue",
        dest="auto_cue",
        action="store_false",
        help="Disable automatic cue-time inference; stretch each section uniformly",
    )
    p.add_argument(
        "--cue-step",
        type=float,
        default=1.0,
        help="Frame sample interval in seconds for cue inference (default: 1.0)",
    )
    p.add_argument(
        "--cue-scene-threshold",
        type=float,
        default=0.3,
        help="ffmpeg scene-detection threshold; lower = more cuts (default: 0.3)",
    )
    p.add_argument(
        "--subtitle-style",
        choices=["pill", "classic"],
        default="pill",
        dest="subtitle_style",
        help="Caption look: pill (blue rounded box, default) or classic (ASS outline)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + preview only; no API calls, no render",
    )
    p.add_argument(
        "--burn-only",
        action="store_true",
        help="Re-burn subtitles onto out/final.nosubs.mp4 using cached Whisper timings",
    )
    args = p.parse_args()

    try:
        single, per_beat, sections = _resolve_videos(args)
    except PreflightError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    if args.burn_only:
        from .alignment import align_beats
        from .subtitles import burn_into_video, write_srt
        from .tts import synthesize_beats

        out_root = Path(args.out)
        nosubs = out_root / "final.nosubs.mp4"
        if not nosubs.exists():
            print(f"no {nosubs} found — run full pipeline first", file=sys.stderr)
            sys.exit(2)
        beats = parse_script_file(args.script)
        # Uses cached mp3s + whisper JSON; no API spend if caches present.
        beats = synthesize_beats(beats, out_root / "audio", voice=args.voice, model=args.tts_model)
        beats = align_beats(beats)
        srt = write_srt(beats, out_root / "subs.srt")
        if args.subtitle_style == "pill":
            from .subtitles import burn_pill_subtitles
            burned = burn_pill_subtitles(nosubs, srt, out_root / "final.mp4")
        else:
            burned = burn_into_video(nosubs, srt, out_root / "final.mp4")
        print(f"\nBurned subtitles -> {burned}")
        sys.exit(0)

    if args.dry_run:
        try:
            check(
                args.script,
                single,
                per_beat or sections,
                need_openai=False,
                require_video=False,
            )
        except PreflightError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
        sys.exit(_dry_run(args, sections))

    try:
        check(args.script, single, per_beat or sections, need_openai=True)
    except PreflightError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    beats = parse_script_file(args.script)
    cue_assets = discover_cue_assets(beats, args.cues_dir)
    cue_assets.update(_override_assets(args.cue_assets))

    cue_config = CueFinderConfig(
        sample_interval=args.cue_step,
        scene_threshold=args.cue_scene_threshold,
    )

    manifest = run(
        script_path=args.script,
        video_path=single,
        video_paths=per_beat,
        section_videos=sections,
        out_dir=args.out,
        voice=args.voice,
        tts_model=args.tts_model,
        cue_assets=cue_assets,
        subtitles_format=None if args.subs == "none" else args.subs,
        burn_subs=args.burn_subs,
        subtitle_style=args.subtitle_style,
        fps=args.fps,
        auto_cue=args.auto_cue,
        cue_config=cue_config,
    )

    if "cue_plan" in manifest:
        _print_cue_plan(manifest["cue_plan"])

    print("\nDone.")
    for key, value in manifest.items():
        print(f"  {key:18s} {value}")


def _print_cue_plan(plan_path: str) -> None:
    """Render a compact summary of the inferred cue-plan JSON."""
    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except Exception:
        return
    print("\nInferred cue timestamps")
    for section in plan.get("sections", []):
        marker = section.get("section_marker") or "(intro)"
        print(f"  Section {section['index']:>2}: {marker}")
        if not section["assignments"]:
            print("           (no non-section cues)")
            continue
        for a in section["assignments"]:
            ts = a["source_time"]
            ts_str = f"{ts:6.2f}s" if ts is not None else "  ?    "
            print(
                f"           [{a['cue']}]  -> {ts_str}  "
                f"({a['method']}, conf {a['confidence']:.2f})"
            )


if __name__ == "__main__":
    main()
