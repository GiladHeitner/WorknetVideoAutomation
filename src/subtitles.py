"""Derive WebVTT/SRT from Whisper word timings and burn them into video.

Walks spoken beats in order, projects each word's per-beat ``start``/``end``
into the absolute timeline, groups into short readable lines, then emits
either WebVTT or SRT. ``burn_into_video`` converts the SRT to a styled ASS
file and uses ffmpeg's ``ass`` filter (with ``shaping=simple``) to bake
captions into the pixels — orders of magnitude faster than the
``subtitles`` filter on long inputs.
"""

from __future__ import annotations

import re
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


_ASS_STYLE_FIELDS = [
    ("Name", "Default"),
    ("Fontname", "Arial"),
    ("Fontsize", "18"),
    ("PrimaryColour", "&H00FFFFFF"),
    ("SecondaryColour", "&H000000FF"),
    ("OutlineColour", "&H00000000"),
    ("BackColour", "&H80000000"),
    ("Bold", "1"),
    ("Italic", "0"),
    ("Underline", "0"),
    ("StrikeOut", "0"),
    ("ScaleX", "100"),
    ("ScaleY", "100"),
    ("Spacing", "0"),
    ("Angle", "0"),
    ("BorderStyle", "1"),
    ("Outline", "1.5"),
    ("Shadow", "1"),
    ("Alignment", "2"),
    ("MarginL", "15"),
    ("MarginR", "15"),
    ("MarginV", "20"),
    ("Encoding", "1"),
]


def _parse_force_style(style: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in style.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        k, v = token.split("=", 1)
        out[k.strip()] = v.strip()
    # ASS Style line uses "Fontname"/"Fontsize"; force_style uses "FontName"/"FontSize".
    if "FontName" in out:
        out.setdefault("Fontname", out["FontName"])
    if "FontSize" in out:
        out.setdefault("Fontsize", out["FontSize"])
    return out


def _build_ass_style_line(style: str) -> str:
    overrides = _parse_force_style(style)
    values = [overrides.get(k, default) for k, default in _ASS_STYLE_FIELDS]
    return "Style: " + ",".join(values)


def _srt_time_to_ass(t: str) -> str:
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    cs = int(ms) // 10
    return f"{int(h)}:{int(m):02d}:{int(s):02d}.{cs:02d}"


def _srt_text_to_ass(text: str) -> str:
    # Commas in dialogue lines collide with the comma-separated Format spec, so
    # swap to a visually-identical "single low-9 quotation mark" the way the
    # libass docs recommend; backslashes get escaped to keep ASS overrides safe.
    safe = text.replace("\\", "\\\\").replace(",", "\u201A")
    return safe.replace("\n", "\\N")


def srt_to_ass(srt_path: str | Path, ass_path: str | Path, style: str = _DEFAULT_STYLE) -> str:
    """Write a styled ``.ass`` file from a plain ``.srt``."""
    raw = Path(srt_path).read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", raw.strip())
    events: list[str] = []
    for blk in blocks:
        lines = blk.strip().splitlines()
        if len(lines) < 2:
            continue
        m = re.match(r"\s*(\S+)\s*-->\s*(\S+)", lines[1])
        if not m:
            continue
        start = _srt_time_to_ass(m.group(1))
        end = _srt_time_to_ass(m.group(2))
        body = _srt_text_to_ass("\n".join(lines[2:]))
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{body}")

    style_line = _build_ass_style_line(style)
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 640\n"
        "PlayResY: 360\n\n"
        "[V4+ Styles]\n"
        "Format: " + ", ".join(k for k, _ in _ASS_STYLE_FIELDS) + "\n"
        + style_line + "\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    out = header + "\n".join(events) + "\n"
    Path(ass_path).parent.mkdir(parents=True, exist_ok=True)
    Path(ass_path).write_text(out, encoding="utf-8")
    return str(ass_path)


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
    """Burn an SRT into ``video_path`` and write to ``out_path``.

    Internally converts the SRT to a styled ASS file and uses ffmpeg's
    ``ass`` filter with ``shaping=simple``. On long inputs this is roughly
    100x faster than the default ``subtitles`` filter (which uses HarfBuzz
    complex shaping by default and stalls out on multi-minute videos).
    """
    ffmpeg = _ffmpeg_binary()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ass_path = Path(srt_path).with_suffix(".ass")
    srt_to_ass(srt_path, ass_path, style=style)

    filter_arg = f"ass=filename='{ass_path}':shaping=simple"
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        str(video_path),
        "-vf",
        filter_arg,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        "-stats",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return str(out_path)
