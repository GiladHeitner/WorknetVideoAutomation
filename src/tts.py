"""Phase 2 — OpenAI TTS voice engine.

For each ``spoken_text`` beat, generates an ``mp3`` using ``tts-1`` and records
the local path on the beat so later phases can locate the audio.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .parser import Beat


DEFAULT_VOICE = "alloy"
DEFAULT_MODEL = "tts-1"


def _client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing; copy .env.example to .env and set it.")
    return OpenAI(api_key=api_key)


def _audio_filename(text: str, voice: str, model: str) -> str:
    digest = hashlib.sha1(f"{model}|{voice}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"vo_{digest}.mp3"


def synthesize_beats(
    beats: list[Beat],
    audio_dir: str | Path,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
) -> list[Beat]:
    """Generate one mp3 per spoken beat and annotate the beat in place.

    Skips synthesis when the deterministic cache file already exists so reruns
    are cheap.
    """
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    client = _client()

    for beat in beats:
        if beat.get("type") != "spoken_text":
            continue
        text = beat["content"]
        out_path = audio_dir / _audio_filename(text, voice, model)
        if not out_path.exists():
            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
            ) as response:
                response.stream_to_file(out_path)
        beat["audio_path"] = str(out_path)

    return beats


if __name__ == "__main__":
    import json
    import sys

    from .parser import parse_script_file

    target = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_script.txt"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "out/audio"
    beats = synthesize_beats(parse_script_file(target), out_dir)
    print(json.dumps(beats, indent=2, ensure_ascii=False))
