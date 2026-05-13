"""Auto-discover overlay images for ``[visual_cue]`` beats.

Convention: drop image files in ``assets/`` (or a directory passed via
``--cues-dir``) named after the cue text. Both the raw text and a slugified
form are tried, in this order of preference: png, webp, jpg, jpeg, gif.
"""

from __future__ import annotations

import re
from pathlib import Path

from .parser import Beat


_EXTS = (".png", ".webp", ".jpg", ".jpeg", ".gif")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _candidates(cue: str) -> list[str]:
    names = {cue.strip(), cue.strip().lower(), _slug(cue)}
    return [f"{n}{ext}" for n in names if n for ext in _EXTS]


def discover_cue_assets(beats: list[Beat], cues_dir: str | Path) -> dict[str, str]:
    """Return a mapping of cue text -> resolved image path for cues that match."""
    cues_dir = Path(cues_dir)
    found: dict[str, str] = {}
    if not cues_dir.exists():
        return found
    for beat in beats:
        if beat.get("type") != "visual_cue":
            continue
        cue = beat["content"]
        if cue in found:
            continue
        for filename in _candidates(cue):
            candidate = cues_dir / filename
            if candidate.exists():
                found[cue] = str(candidate)
                break
    return found


def cue_resolution_report(beats: list[Beat], cues_dir: str | Path) -> list[dict]:
    """Per-cue diagnostic rows for ``--dry-run`` output."""
    resolved = discover_cue_assets(beats, cues_dir)
    seen: set[str] = set()
    rows: list[dict] = []
    for beat in beats:
        if beat.get("type") != "visual_cue":
            continue
        cue = beat["content"]
        if cue in seen:
            continue
        seen.add(cue)
        rows.append({"cue": cue, "asset": resolved.get(cue)})
    return rows
