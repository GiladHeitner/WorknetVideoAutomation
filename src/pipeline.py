"""End-to-end orchestrator: script + video → narrated mp4 (+ optional subs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .alignment import align_beats
from .parser import parse_script_file
from .renderer import RenderConfig, render, render_sections
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
) -> dict:
    """Run all five phases. Returns a manifest dict of artifact paths."""
    out_root = Path(out_dir)
    audio_dir = out_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    beats = parse_script_file(script_path)
    beats = synthesize_beats(beats, audio_dir, voice=voice, model=tts_model)
    beats = align_beats(beats)

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

    if subtitles_format == "vtt":
        manifest["subtitles"] = write_vtt(beats, out_root / "subs.vtt")
    elif subtitles_format == "srt":
        manifest["subtitles"] = write_srt(beats, out_root / "subs.srt")

    if burn_subs:
        # libass needs SRT; keep sidecar in the user's chosen format too.
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
