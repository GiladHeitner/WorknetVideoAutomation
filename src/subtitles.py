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
import tempfile
from pathlib import Path

from .parser import Beat

# Path to the bundled font used for pill-style subtitles.
_FONT_PATH = Path(__file__).parent.parent / "assets" / "fonts" / "RobotoBold.ttf"


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


# ---------------------------------------------------------------------------
# Pill-style subtitle rendering
# ---------------------------------------------------------------------------

def _srt_seconds(t: str) -> float:
    """Convert ``HH:MM:SS,mmm`` SRT timestamp to seconds."""
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _parse_srt(srt_path: str | Path) -> list[dict]:
    """Return ``[{start, end, text}, ...]`` from a plain SRT file."""
    raw = Path(srt_path).read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", raw.strip())
    cues: list[dict] = []
    for blk in blocks:
        lines = blk.strip().splitlines()
        if len(lines) < 2:
            continue
        m = re.match(r"\s*(\S+)\s*-->\s*(\S+)", lines[1])
        if not m:
            continue
        cues.append({
            "start": _srt_seconds(m.group(1)),
            "end": _srt_seconds(m.group(2)),
            "text": "\n".join(lines[2:]).strip(),
        })
    return cues


def _wrap_pill_text(text: str, font, max_inner_px: int) -> str:
    """Word-wrap ``text`` so no line exceeds ``max_inner_px`` wide."""
    from PIL import ImageDraw, Image

    def _line_width(line: str) -> int:
        bb = font.getbbox(line)
        return bb[2] - bb[0]

    result_lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            result_lines.append("")
            continue
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if current and _line_width(candidate) > max_inner_px:
                result_lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            result_lines.append(" ".join(current))
    return "\n".join(result_lines)


def _make_pill_image(
    text: str,
    video_w: int,
    video_h: int,
    font_path: Path,
    bg_color: tuple,
    text_color: str,
    font_size: int,
    h_pad: int,
    v_pad: int,
    radius: int,
    margin_bottom: int,
    side_margin: int,
) -> "PIL.Image.Image":
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path), font_size)
    text = " ".join(text.splitlines())  # collapse any two-line SRT wrapping

    img = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bb = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    text_w = bb[2] - bb[0]
    text_h = bb[3] - bb[1]

    pill_w = text_w + h_pad * 2
    pill_h = text_h + v_pad * 2
    pill_x0 = (video_w - pill_w) // 2
    pill_y0 = video_h - margin_bottom - pill_h
    pill_x1 = pill_x0 + pill_w
    pill_y1 = pill_y0 + pill_h

    draw.rounded_rectangle(
        [pill_x0, pill_y0, pill_x1, pill_y1],
        radius=radius,
        fill=(*bg_color, 255),
    )

    text_x = pill_x0 + h_pad - bb[0]
    text_y = pill_y0 + v_pad - bb[1]
    draw.multiline_text(
        (text_x, text_y),
        text,
        font=font,
        fill=text_color,
        spacing=4,
        align="center",
    )
    return img


def _ffprobe_video_size(video_path: str | Path) -> tuple[int, int]:
    """Return (width, height) via ffprobe."""
    ffmpeg = _ffmpeg_binary()
    ffprobe = shutil.which("ffprobe") or ffmpeg.replace("ffmpeg", "ffprobe")
    result = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    w, h = result.stdout.strip().split(",")
    return int(w), int(h)


def burn_pill_subtitles(
    video_path: str | Path,
    srt_path: str | Path,
    out_path: str | Path,
    font_path: str | Path | None = None,
    bg_color: tuple = (26, 86, 219),
    text_color: str = "white",
    font_size: int = 24,
    h_pad: int = 24,
    v_pad: int = 10,
    radius: int = 20,
    margin_bottom: int = 15,
    side_margin: int = 40,
) -> str:
    """Burn blue pill-shaped subtitles onto ``video_path`` and write to ``out_path``.

    Generates a transparent PNG overlay for each SRT cue using Pillow, assembles
    them into a concat-demuxer sequence, then composites with FFmpeg's overlay
    filter — a single two-input command regardless of cue count.
    """
    resolved_font = Path(font_path) if font_path else _FONT_PATH
    if not resolved_font.exists():
        raise FileNotFoundError(f"Pill subtitle font not found: {resolved_font}")

    cues = _parse_srt(srt_path)
    if not cues:
        shutil.copy2(str(video_path), str(out_path))
        return str(out_path)

    video_w, video_h = _ffprobe_video_size(video_path)
    ffmpeg = _ffmpeg_binary()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Transparent full-frame placeholder used between cues.
        from PIL import Image
        blank = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
        blank_path = tmp / "transparent.png"
        blank.save(blank_path)

        # Render one PNG per cue.
        pill_paths: list[Path] = []
        for i, cue in enumerate(cues):
            img = _make_pill_image(
                cue["text"], video_w, video_h,
                resolved_font, bg_color, text_color,
                font_size, h_pad, v_pad, radius, margin_bottom, side_margin,
            )
            p = tmp / f"pill_{i:04d}.png"
            img.save(p)
            pill_paths.append(p)

        # Build concat manifest. Gaps between cues are filled with the blank.
        concat_lines: list[str] = []
        cursor = 0.0
        for i, cue in enumerate(cues):
            gap = cue["start"] - cursor
            if gap > 0.001:
                concat_lines += [f"file '{blank_path}'", f"duration {gap:.3f}"]
            concat_lines += [f"file '{pill_paths[i]}'", f"duration {cue['end'] - cue['start']:.3f}"]
            cursor = cue["end"]
        # No trailing entry needed — FFmpeg trims to the video length.
        concat_path = tmp / "subs_concat.txt"
        concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "info",
            "-i", str(video_path),
            "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-filter_complex", "[0:v][1:v]overlay=0:0",
            "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "copy", "-movflags", "+faststart", "-stats",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)

    return str(out_path)
