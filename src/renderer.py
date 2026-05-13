"""Phases 4 & 5 — MoviePy renderer.

Phase 4 time-remaps each segment of the raw screen recording so its length
matches the duration of the corresponding generated voiceover.

Phase 5 schedules ``visual_cue`` overlays at the end of the last spoken word
preceding the cue, mapped into the final timeline's absolute seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)

from .parser import Beat
from .sections import DEFAULT_MARKER, reconcile_videos, split_sections


@dataclass
class RenderConfig:
    """Tunables for the final render."""

    output_path: str = "out/final.mp4"
    fps: int = 30
    cue_assets: Optional[dict[str, str]] = None
    cue_duration: float = 2.0
    cue_position: tuple = ("right", "top")
    cue_margin: int = 32


def _spoken_beats(beats: list[Beat]) -> list[Beat]:
    return [b for b in beats if b.get("type") == "spoken_text"]


def _segment_source_video(
    source: VideoFileClip, n_segments: int
) -> list[VideoFileClip]:
    """Split a single source clip into n equal slices preserving order."""
    if n_segments <= 0:
        raise ValueError("need at least one spoken beat to split video against")
    total = source.duration
    boundaries = [total * i / n_segments for i in range(n_segments + 1)]
    return [source.subclip(boundaries[i], boundaries[i + 1]) for i in range(n_segments)]


def _load_segments(
    video_path: Optional[str],
    video_paths: Optional[list[str]],
    n_spoken: int,
) -> list[VideoFileClip]:
    if video_paths:
        if len(video_paths) != n_spoken:
            raise ValueError(
                f"video_paths length {len(video_paths)} != spoken beats {n_spoken}"
            )
        return [VideoFileClip(p) for p in video_paths]
    if not video_path:
        raise ValueError("either video_path or video_paths must be provided")
    source = VideoFileClip(video_path)
    return _segment_source_video(source, n_spoken)


def render(
    beats: list[Beat],
    video_path: Optional[str] = None,
    video_paths: Optional[list[str]] = None,
    config: RenderConfig = RenderConfig(),
) -> str:
    """Build the final video.

    Returns the path to the rendered mp4.
    """
    spoken = _spoken_beats(beats)
    if not spoken:
        raise ValueError("no spoken_text beats found; nothing to render")

    raw_segments = _load_segments(video_path, video_paths, len(spoken))

    fitted_clips: list = []
    audio_clips: list = []
    spoken_starts: dict[int, float] = {}
    timeline_cursor = 0.0

    for idx, beat in enumerate(spoken):
        audio_path = beat.get("audio_path")
        if not audio_path:
            raise RuntimeError(f"spoken beat missing audio_path: {beat!r}")

        audio = AudioFileClip(audio_path)
        target_dur = float(beat.get("duration") or audio.duration)

        raw = raw_segments[idx]
        # speedx with final_duration handles both stretch and squeeze.
        fitted = raw.fx(vfx.speedx, final_duration=target_dur).set_duration(target_dur)
        fitted_clips.append(fitted.without_audio())
        audio_clips.append(audio.set_duration(target_dur))

        spoken_starts[id(beat)] = timeline_cursor
        timeline_cursor += target_dur

    base = concatenate_videoclips(fitted_clips, method="compose")
    voice_track = concatenate_audioclips(audio_clips)
    base = base.set_audio(voice_track)

    overlays = _build_overlays(beats, spoken_starts, base.duration, config)
    final = CompositeVideoClip([base, *overlays]) if overlays else base

    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(
        str(out_path),
        fps=config.fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(out_path.with_suffix(".tempaudio.m4a")),
        remove_temp=True,
    )

    final.close()
    base.close()
    voice_track.close()
    for clip in fitted_clips + audio_clips + raw_segments:
        try:
            clip.close()
        except Exception:
            pass

    return str(out_path)


def render_sections(
    beats: list[Beat],
    section_videos: list[str],
    config: RenderConfig = RenderConfig(),
    marker_pattern: str = DEFAULT_MARKER,
) -> str:
    """Section-aware render: one source video per ``[Cut to ...]`` section.

    If non-section ``visual_cue`` beats carry an inferred ``source_time``,
    each section's source video is anchored at those timestamps so the
    bracketed action shows up immediately before the narration that
    follows. Source intervals between anchors are individually
    speed-fitted to match the corresponding narration interval.

    With no inferred timestamps, this falls back to the previous behavior:
    the whole section video is stretched to the section's total audio
    duration.
    """
    sections = reconcile_videos(split_sections(beats, marker_pattern), len(section_videos))

    fitted_clips: list = []
    audio_clips: list = []
    spoken_starts: dict[int, float] = {}
    timeline_cursor = 0.0
    sources: list[VideoFileClip] = [VideoFileClip(p) for p in section_videos]

    for section, source in zip(sections, sources):
        section_spoken = [b for b in section["beats"] if b.get("type") == "spoken_text"]
        if not section_spoken:
            continue

        per_dur: list[float] = []
        for b in section_spoken:
            if not b.get("audio_path"):
                raise RuntimeError(f"spoken beat missing audio_path: {b!r}")
            dur = float(b.get("duration") or 0.0)
            if dur <= 0:
                dur = float(AudioFileClip(b["audio_path"]).duration)
            per_dur.append(dur)
        section_total = sum(per_dur)

        anchors = _build_section_anchors(section, per_dur, source.duration)
        section_video = _build_anchored_section_video(source, anchors, section_total)
        fitted_clips.append(section_video.without_audio())

        for beat, dur in zip(section_spoken, per_dur):
            audio_clips.append(AudioFileClip(beat["audio_path"]).set_duration(dur))
            spoken_starts[id(beat)] = timeline_cursor
            timeline_cursor += dur

    if not fitted_clips:
        raise ValueError("no spoken beats found across sections; nothing to render")

    base = concatenate_videoclips(fitted_clips, method="compose")
    voice_track = concatenate_audioclips(audio_clips)
    base = base.set_audio(voice_track)

    section_start_ids = {id(s["start_cue"]) for s in sections if s["start_cue"]}
    overlay_beats = [
        b for b in beats
        if not (b.get("type") == "visual_cue" and id(b) in section_start_ids)
    ]
    overlays = _build_overlays(overlay_beats, spoken_starts, base.duration, config)
    final = CompositeVideoClip([base, *overlays]) if overlays else base

    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(
        str(out_path),
        fps=config.fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(out_path.with_suffix(".tempaudio.m4a")),
        remove_temp=True,
    )

    final.close()
    base.close()
    voice_track.close()
    for clip in fitted_clips + audio_clips + sources:
        try:
            clip.close()
        except Exception:
            pass

    return str(out_path)


def _build_section_anchors(
    section,
    per_dur: list[float],
    source_duration: float,
    min_gap: float = 0.05,
) -> list[tuple[float, float]]:
    """Walk this section's beats and emit ``(final_time, source_time)`` anchors.

    Each non-section ``visual_cue`` with a known ``source_time`` becomes
    an anchor at the start of the spoken text that follows it. Anchors
    are clamped to stay strictly monotonic in both axes so MoviePy can
    actually retime each interval.
    """
    anchors: list[tuple[float, float]] = [(0.0, 0.0)]
    final = 0.0
    spoken_idx = 0
    last_source = 0.0
    pending: float | None = None

    for beat in section["beats"]:
        if beat.get("type") == "visual_cue":
            t = beat.get("source_time")
            if t is None:
                continue
            t = max(float(t), last_source + min_gap)
            t = min(t, max(0.0, source_duration - min_gap))
            pending = t if pending is None else max(pending, t)
        elif beat.get("type") == "spoken_text":
            if pending is not None and final > 0.0:
                anchors.append((final, pending))
                last_source = pending
                pending = None
            final += per_dur[spoken_idx]
            spoken_idx += 1

    anchors.append((final, source_duration))
    return _dedupe_anchors(anchors, min_gap)


def _dedupe_anchors(
    anchors: list[tuple[float, float]],
    min_gap: float,
) -> list[tuple[float, float]]:
    """Drop anchors that would yield zero/near-zero intervals on either axis."""
    cleaned: list[tuple[float, float]] = []
    for f, s in anchors:
        if cleaned:
            pf, ps = cleaned[-1]
            if (f - pf) < min_gap or (s - ps) < min_gap:
                continue
        cleaned.append((f, s))
    if len(cleaned) < 2 and anchors:
        return [anchors[0], anchors[-1]]
    return cleaned


def _build_anchored_section_video(
    source: VideoFileClip,
    anchors: list[tuple[float, float]],
    section_total: float,
) -> VideoFileClip:
    """Stitch speed-fitted subclips of ``source`` across the anchor list."""
    sub_clips: list = []
    for i in range(len(anchors) - 1):
        f0, s0 = anchors[i]
        f1, s1 = anchors[i + 1]
        final_dur = f1 - f0
        if final_dur <= 0:
            continue
        s_end = min(s1, source.duration)
        if s_end <= s0:
            continue
        sub = source.subclip(s0, s_end)
        if sub.duration <= 0:
            continue
        fitted = sub.fx(vfx.speedx, final_duration=final_dur).set_duration(final_dur)
        sub_clips.append(fitted)

    if not sub_clips:
        return source.fx(vfx.speedx, final_duration=section_total).set_duration(section_total)
    if len(sub_clips) == 1:
        return sub_clips[0]
    return concatenate_videoclips(sub_clips, method="compose")


def _build_overlays(
    beats: list[Beat],
    spoken_starts: dict[int, float],
    total_duration: float,
    config: RenderConfig,
) -> list[ImageClip]:
    """For each ``visual_cue``, position an asset at the end of the last word
    spoken just before the cue, in the final concatenated timeline."""
    assets = config.cue_assets or {}
    overlays: list[ImageClip] = []
    next_cue_time: dict[int, float] = {}

    cue_anchor_times: list[tuple[int, float, Beat]] = []
    last_spoken: Beat | None = None
    for beat in beats:
        if beat["type"] == "spoken_text":
            last_spoken = beat
            continue
        if beat["type"] == "visual_cue":
            if last_spoken is None:
                anchor = 0.0
            else:
                spoken_start = spoken_starts[id(last_spoken)]
                last_word_end = (
                    last_spoken["words"][-1]["end"]
                    if last_spoken.get("words")
                    else float(last_spoken.get("duration") or 0.0)
                )
                anchor = spoken_start + float(last_word_end)
            cue_anchor_times.append((len(overlays), anchor, beat))
            overlays.append(None)  # placeholder; replaced below

    # Resolve durations: until the next cue or config.cue_duration, capped at video end.
    for i, (slot, anchor, beat) in enumerate(cue_anchor_times):
        next_anchor = (
            cue_anchor_times[i + 1][1] if i + 1 < len(cue_anchor_times) else total_duration
        )
        max_dur = max(0.0, min(config.cue_duration, next_anchor - anchor, total_duration - anchor))
        if max_dur <= 0:
            overlays[slot] = None
            continue

        asset_path = assets.get(beat["content"])
        if not asset_path or not Path(asset_path).exists():
            overlays[slot] = None
            continue

        clip = (
            ImageClip(asset_path)
            .set_start(anchor)
            .set_duration(max_dur)
            .set_position(config.cue_position)
        )
        overlays[slot] = clip
        next_cue_time[slot] = anchor

    return [o for o in overlays if o is not None]
