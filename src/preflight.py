"""Fail-fast environment + input checks with actionable error messages."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv


class PreflightError(RuntimeError):
    """Raised when the pipeline cannot run; message is user-facing."""


def check(
    script_path: str,
    video_path: str | None,
    video_paths: list[str] | None,
    need_openai: bool = True,
    require_video: bool = True,
) -> None:
    problems: list[str] = []

    if not Path(script_path).is_file():
        problems.append(f"script not found: {script_path}")

    if video_paths:
        for v in video_paths:
            if not Path(v).is_file():
                problems.append(f"video clip not found: {v}")
    elif video_path:
        if not Path(video_path).is_file():
            problems.append(f"video not found: {video_path}")
    elif require_video:
        problems.append("no video provided — pass a path or use --video-list")

    if need_openai:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            problems.append(
                "OPENAI_API_KEY not set — `cp .env.example .env` and add your key"
            )

    if not (shutil.which("ffmpeg") or os.getenv("IMAGEIO_FFMPEG_EXE")):
        # MoviePy 1.0.3 ships imageio-ffmpeg so this is a soft warning, not fatal.
        pass

    if problems:
        raise PreflightError("\n  - ".join(["preflight failed:", *problems]))
