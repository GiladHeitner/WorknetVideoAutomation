"""Derive WebVTT/SRT from Whisper word timings and burn them into video.

Walks spoken beats in order, projects each word's per-beat ``start``/``end``
into the absolute timeline, groups into short readable lines, then emits
either WebVTT or SRT. ``burn_into_video`` post-processes the rendered mp4
with ffmpeg's ``subtitles`` filter so captions are baked into the pixels.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .parser import Beat


def _format_vtt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _format_srt_time(seconds: float) -> str:
    return _format_vtt_time(seconds).replace(".", ",")


def _absolute_word_stream(beats: list[Beat]):
    cursor = 0.0
    for beat in beats:
        if beat.get("type") != "spoken_text":
            continue
        beat_duration = float(beat.get("duration") or 0.0)
        for w in beat.get("words", []):
            yield {
                "word": w["word"],
                "start": cursor + float(w["start"]),
                "end": cursor + float(w["end"]),
            }
        cursor += beat_duration


def _wrap_two_lines(text: str, target_chars: int = 32) -> str:
    """Wrap into at most 2 visually-balanced lines using ``\\N`` separator."""
    if len(text) <= target_chars:
        return text
    words = text.split()
    best_split = len(words) // 2
    best_diff = float("inf")
    for i in range(1, len(words)):
        a = len(" ".join(words[:i]))
        b = len(" ".join(words[i:]))
        if max(a, b) > target_chars + 8:
            continue
        diff = abs(a - b)
        if diff < best_diff:
            best_diff = diff
            best_split = i
    line_a = " ".join(words[:best_split])
    line_b = " ".join(words[best_split:])
    return f"{line_a}\\N{line_b}"


def _group_into_lines(
    words: list[dict],
    max_words: int = 6,
    max_chars: int = 64,
    max_gap: float = 0.6,
) -> list[dict]:
    """Group words into short, readable cues (≤ ~6 words / ~64 chars / 0.6s gap)."""
    lines: list[dict] = []
    buf: list[dict] = []

    def buf_len() -> int:
        return sum(len(w["word"].strip()) for w in buf) + max(0, len(buf) - 1)

    def flush():
        if not buf:
            return
        text = " ".join(w["word"].strip() for w in buf).strip()
        lines.append({"start": buf[0]["start"], "end": buf[-1]["end"], "text": text})
        buf.clear()

    for w in words:
        if buf:
            gap = w["start"] - buf[-1]["end"]
            projected = buf_len() + 1 + len(w["word"].strip())
            if len(buf) >= max_words or projected > max_chars or gap > max_gap:
                flush()
        buf.append(w)
    flush()
    return lines


def write_vtt(beats: list[Beat], path: str | Path) -> str:
    lines = _group_into_lines(list(_absolute_word_stream(beats)))
    out = ["WEBVTT", ""]
    for line in lines:
        out.append(f"{_format_vtt_time(line['start'])} --> {_format_vtt_time(line['end'])}")
        out.append(line["text"])
        out.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    return str(path)


def write_srt(beats: list[Beat], path: str | Path, wrap: bool = True) -> str:
    lines = _group_into_lines(list(_absolute_word_stream(beats)))
    out: list[str] = []
    for i, line in enumerate(lines, start=1):
        out.append(str(i))
        out.append(f"{_format_srt_time(line['start'])} --> {_format_srt_time(line['end'])}")
        text = _wrap_two_lines(line["text"]) if wrap else line["text"]
        # libass parses \N inside SRT for forced line breaks; SRT spec allows literal newlines too.
        out.append(text.replace("\\N", "\n"))
        out.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out), encoding="utf-8")
    return str(path)


_DEFAULT_STYLE = (
    "FontName=Arial,FontSize=18,"
    "PrimaryColour=&H00FFFFFF,SecondaryColour=&H000000FF,"
    "OutlineColour=&H00000000,BackColour=&H80000000,"
    "Bold=1,Italic=0,Underline=0,StrikeOut=0,"
    "ScaleX=100,ScaleY=100,Spacing=0,Angle=0,"
    "BorderStyle=1,Outline=1.5,Shadow=1,"
    "Alignment=2,MarginL=15,MarginR=15,MarginV=20,Encoding=1"
)


def _ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("ffmpeg not found on PATH and imageio-ffmpeg unavailable") from e


def burn_into_video(
    video_path: str | Path,
    srt_path: str | Path,
    out_path: str | Path,
    style: str = _DEFAULT_STYLE,
) -> str:
    """Burn an SRT into ``video_path`` and write to ``out_path``."""
    ffmpeg = _ffmpeg_binary()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg's subtitles filter wants a forward-slashed path; escape ':' in
    # absolute Windows paths is not a concern here (POSIX target).
    filter_arg = f"subtitles='{srt_path}':force_style='{style}'"
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        filter_arg,
        "-c:a",
        "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return str(out_path)
