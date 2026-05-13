"""Phase 3 — Whisper alignment engine.

Sends each generated mp3 through ``whisper-1`` with verbose JSON and
word-level timestamp granularity so downstream code knows exactly when each
word is spoken inside its beat.
"""

from __future__ import annotations

import json
from pathlib import Path

from .parser import Beat
from .tts import _client


WHISPER_MODEL = "whisper-1"


def _cache_path(audio_path: str) -> Path:
    return Path(audio_path).with_suffix(".words.json")


def align_beats(beats: list[Beat], use_cache: bool = True) -> list[Beat]:
    """Annotate each spoken beat with word timings and total audio duration.

    Caches the Whisper payload as ``<audio>.words.json`` so reruns skip the API.
    """
    client = None

    for beat in beats:
        if beat.get("type") != "spoken_text":
            continue
        audio_path = beat.get("audio_path")
        if not audio_path:
            raise RuntimeError(f"spoken beat missing audio_path: {beat!r}")

        cache_file = _cache_path(audio_path)
        if use_cache and cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            if client is None:
                client = _client()
            with open(audio_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model=WHISPER_MODEL,
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )
            payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        words = payload.get("words") or []
        beat["words"] = [
            {"word": w.get("word", ""), "start": float(w["start"]), "end": float(w["end"])}
            for w in words
            if "start" in w and "end" in w
        ]
        if beat["words"]:
            beat["duration"] = beat["words"][-1]["end"]
        else:
            beat["duration"] = float(payload.get("duration", 0.0))

    return beats


def audio_duration_seconds(path: str | Path) -> float:
    """Fallback duration probe via imageio_ffmpeg if Whisper gave us nothing."""
    from moviepy.audio.io.AudioFileClip import AudioFileClip

    with AudioFileClip(str(path)) as clip:
        return float(clip.duration)


if __name__ == "__main__":
    import json
    import sys

    from .parser import parse_script_file
    from .tts import synthesize_beats

    target = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_script.txt"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "out/audio"
    beats = parse_script_file(target)
    beats = synthesize_beats(beats, out_dir)
    beats = align_beats(beats)
    print(json.dumps(beats, indent=2, ensure_ascii=False))
