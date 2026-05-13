"""End-to-end orchestrator: script + video → narrated mp4 (+ optional subs).

When section videos are provided, this also infers source-video timestamps
for every bracketed action via :mod:`src.cue_finder`, writes a generated
timestamped script + cue plan under ``out/``, and uses those anchors when
rendering so each action lines up with the narration that follows it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .alignment import align_beats
from .cue_finder import (
    CueFinderConfig,
    build_cue_plan,
    render_timestamped_script,
)
from .parser import parse_script_file
from .renderer import RenderConfig, render, render_sections
from .sections import DEFAULT_MARKER, reconcile_videos, split_sections
from .subtitles import burn_into_video, write_srt, write_vtt
from .tts import synthesize_beats


def run(
    script_path: str,
    video_path: Optional[str] = None,
    video_paths: Optional[list[str]] = None,
    section_videos: Optional[list[str]] = None,
    out_dir: str = "out",
    voice: str = "alloy",
    tts_model: str = "tts-1",
    cue_assets: Optional[dict[str, str]] = None,
    subtitles_format: Optional[str] = "vtt",
    burn_subs: bool = True,
    fps: int = 30,
    auto_cue: bool = True,
    cue_config: Optional[CueFinderConfig] = None,
) -> dict:
    """Run all phases. Returns a manifest dict of artifact paths."""
    out_root = Path(out_dir)
    audio_dir = out_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    beats = parse_script_file(script_path)
    beats = synthesize_beats(beats, audio_dir, voice=voice, model=tts_model)
    beats = align_beats(beats)

    cue_plan: Optional[dict] = None
    if section_videos and auto_cue:
        sections = reconcile_videos(split_sections(beats), len(section_videos))
        sections_with_videos = list(zip(sections, section_videos))
        cue_plan = build_cue_plan(
            sections_with_videos,
            config=cue_config or CueFinderConfig(),
            work_root=out_root / "cue_frames",
        )
        (out_root / "cue_plan.json").write_text(
            json.dumps(cue_plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_root / "timestamped_script.txt").write_text(
            render_timestamped_script(beats), encoding="utf-8"
        )

    (out_root / "beats.json").write_text(
        json.dumps(beats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    render_config = RenderConfig(
        output_path=str(out_root / "final.mp4"),
        fps=fps,
        cue_assets=cue_assets or {},
    )

    if section_videos:
        final_path = render_sections(beats, section_videos, render_config)
    else:
        final_path = render(
            beats,
            video_path=video_path,
            video_paths=video_paths,
            config=render_config,
        )

    manifest = {"beats": str(out_root / "beats.json"), "video": final_path}
    if cue_plan is not None:
        manifest["cue_plan"] = str(out_root / "cue_plan.json")
        manifest["timestamped_script"] = str(out_root / "timestamped_script.txt")

    if subtitles_format == "vtt":
        manifest["subtitles"] = write_vtt(beats, out_root / "subs.vtt")
    elif subtitles_format == "srt":
        manifest["subtitles"] = write_srt(beats, out_root / "subs.srt")

    if burn_subs:
        srt_path = write_srt(beats, out_root / "subs.srt")
        raw_keep = out_root / "final.nosubs.mp4"
        Path(final_path).rename(raw_keep)
        burned = burn_into_video(raw_keep, srt_path, out_root / "final.mp4")
        manifest["video"] = burned
        manifest["video_nosubs"] = str(raw_keep)

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
