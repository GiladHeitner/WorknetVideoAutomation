"""Optional — Scene-based / VLM-driven cue scheduling.

Phase 6+ extension point for semantic placement of script beats when speed-
matching every clip to the voiceover is no longer desirable. Two layers:

1. ``detect_scenes`` — Robust scene-cut detection via ffmpeg's ``select``
   filter; no extra Python deps. PySceneDetect can replace this later for
   richer thresholds.
2. ``plan_cues`` — Given beats + scenes, decide each spoken beat's
   ``start_sec`` against the original video timeline. Left as a stub: the
   product decision (constraints, validators, optional VLM step) belongs in
   a follow-up.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .parser import Beat


_SCENE_RE = re.compile(r"pts_time:([0-9.]+)")


def detect_scenes(video_path: str | Path, threshold: float = 0.4) -> list[float]:
    """Return scene-cut timestamps (seconds) using ffmpeg's scene filter."""
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-i",
        str(video_path),
        "-filter:v",
        f"select='gt(scene,{threshold})',showinfo",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return [float(m.group(1)) for m in _SCENE_RE.finditer(proc.stderr)]


def plan_cues(beats: list[Beat], scenes: list[float]) -> list[Beat]:
    """Assign ``start_sec`` to each spoken beat based on scenes.

    Stub: implement once product decides on constraints (min/max gap, reading
    speed, VLM-derived event timeline, etc.). The earlier generic plan
    sketches the intended LLM-planner shape.
    """
    raise NotImplementedError(
        "plan_cues is a Phase 6+ extension point; see plan for design."
    )
