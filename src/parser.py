"""Phase 1 — Regex script parser.

Turns a free-form script into an ordered list of beats, where each beat is
either spoken narration or an editorial visual cue in square brackets.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, TypedDict


class Beat(TypedDict, total=False):
    type: Literal["spoken_text", "visual_cue"]
    content: str
    audio_path: str
    duration: float
    words: list[dict]
    # Populated by cue_finder for visual_cue beats inside section videos.
    source_time: float
    cue_confidence: float
    cue_method: str


# Matches anything inside [ ... ] non-greedily; the surrounding plain text is
# captured by re.split on the same pattern.
_CUE_RE = re.compile(r"\[([^\[\]]+)\]")


def parse_script(text: str) -> list[Beat]:
    """Parse raw script text into a list of beats preserving original order."""
    beats: list[Beat] = []
    cursor = 0
    for match in _CUE_RE.finditer(text):
        spoken_chunk = text[cursor : match.start()]
        _append_spoken(beats, spoken_chunk)
        beats.append({"type": "visual_cue", "content": match.group(1).strip()})
        cursor = match.end()

    _append_spoken(beats, text[cursor:])
    return beats


def _append_spoken(beats: list[Beat], chunk: str) -> None:
    cleaned = _normalize_spoken(chunk)
    if cleaned:
        beats.append({"type": "spoken_text", "content": cleaned})


def _normalize_spoken(chunk: str) -> str:
    # Collapse any internal newlines + repeated whitespace into single spaces;
    # keep paragraph intent by trimming, but a single block is one spoken beat.
    return re.sub(r"\s+", " ", chunk).strip()


def parse_script_file(path: str | Path) -> list[Beat]:
    return parse_script(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    import json
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_script.txt"
    print(json.dumps(parse_script_file(target), indent=2, ensure_ascii=False))
