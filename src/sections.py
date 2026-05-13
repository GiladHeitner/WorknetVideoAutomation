"""Section splitting — group beats around ``[Cut to ...]`` markers.

Each ``[Cut to ...]`` cue starts a new section so a single source video can
play during all the spoken beats inside that section. Spoken beats before
the first marker (intro) or after the last marker (outro) are merged into
the adjacent section when the video count makes the intent unambiguous.
"""

from __future__ import annotations

import re
from typing import TypedDict

from .parser import Beat


DEFAULT_MARKER = r"(?i)^\s*cut\s+to\b"


class Section(TypedDict):
    start_cue: Beat | None
    beats: list[Beat]


def split_sections(beats: list[Beat], marker_pattern: str = DEFAULT_MARKER) -> list[Section]:
    pat = re.compile(marker_pattern)
    sections: list[Section] = [{"start_cue": None, "beats": []}]
    for beat in beats:
        if beat["type"] == "visual_cue" and pat.search(beat["content"]):
            sections.append({"start_cue": beat, "beats": []})
        else:
            sections[-1]["beats"].append(beat)
    if sections[0]["start_cue"] is None and not sections[0]["beats"]:
        sections.pop(0)
    return sections


def reconcile_videos(sections: list[Section], n_videos: int) -> list[Section]:
    """Adjust sections so their count matches ``n_videos``.

    If the script has a leading intro section without a marker, merge it
    into the first marked section so the first video covers both. Same idea
    for a trailing section that wasn't started by a marker.
    """
    if n_videos == len(sections):
        return sections

    if (
        n_videos == len(sections) - 1
        and sections[0]["start_cue"] is None
        and len(sections) > 1
    ):
        head = sections.pop(0)
        sections[0]["beats"] = head["beats"] + sections[0]["beats"]
        return sections

    raise ValueError(
        f"{n_videos} videos vs {len(sections)} sections; counts must match. "
        "Either add/remove a [Cut to ...] cue, or pass the matching number of videos."
    )
