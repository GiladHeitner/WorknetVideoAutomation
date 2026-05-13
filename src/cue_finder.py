"""Phase 6 — Automatic cue scheduling.

For each section's source video, infer when each bracketed action happens
in original-video time. The pipeline then anchors those source timestamps
to the start of the spoken text that follows, so the action is visible
just before its narration starts.

Strategy: cheap local signals first, vision model only as fallback.

1. Scene/UI-change detection via ffmpeg's ``select='gt(scene,...)`` filter
   yields candidate moments where the screen visibly changes.
2. Frame sampling at a fixed interval gives broader coverage for cues
   whose target moment isn't a hard scene cut.
3. OCR (Tesseract) reads on-screen text for each candidate frame.
4. Greedy in-order matching scores each cue's keywords against the OCR
   text plus a "scene change happened here" bonus, picking the best
   timestamp that respects the order of the cues in the script.

The result is a per-cue ``source_time``, ``confidence``, ``method``
that downstream code uses for retiming. Failures degrade gracefully:
no cue gets a timestamp, the renderer falls back to even stretching.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .parser import Beat
from .sections import DEFAULT_MARKER, Section


_SCENE_RE = re.compile(r"pts_time:([0-9.]+)")

# Words to ignore when matching cue text against OCR text.
_STOPWORDS = {
    "a", "an", "and", "appears", "are", "as", "at", "back", "be",
    "been", "but", "by", "click", "clicks", "do", "does", "down",
    "for", "from", "happens", "has", "have", "he", "her", "here",
    "him", "his", "how", "i", "in", "into", "is", "it", "its",
    "of", "off", "on", "open", "opens", "or", "out", "over", "right",
    "screen", "see", "she", "shows", "so", "than", "that", "the",
    "them", "then", "there", "these", "they", "this", "those", "to",
    "up", "upon", "user", "users", "via", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "why", "will", "with",
    "you", "your",
}


@dataclass
class CueFinderConfig:
    """Tunables for cue-time inference."""

    sample_interval: float = 0.5
    """Seconds between fixed-interval frame samples (in addition to scene cuts)."""

    scene_threshold: float = 0.2
    """ffmpeg scene-detection threshold (0-1; lower = more cuts)."""

    min_cue_gap: float = 0.25
    """Minimum spacing in source seconds between consecutive inferred cues."""

    ocr_lang: str = "eng"
    """Tesseract language."""

    common_token_threshold: float = 0.5
    """If a token appears in more than this fraction of frames, treat it as
    background chrome and de-weight it heavily during cue matching."""

    startup_grace: float = 0.5
    """Skip cue matching earlier than this many seconds at the very start of
    the section; the first frames often carry stale chrome from the
    section-start cut."""

    rare_token_min_weight: float = 0.25
    """OCR matches only count when at least one matched token weighs above
    this; otherwise UI chrome on every frame would dominate."""

    accept_threshold: float = 0.45
    """Walk frames forward; the first frame whose match score crosses this
    is accepted as the cue's source time. Mimics "watch until you see it"."""

    weak_threshold: float = 0.2
    """If no frame crosses ``accept_threshold``, fall back to the
    highest-scoring frame above this; otherwise even-distribute."""


@dataclass
class CandidateFrame:
    time: float
    is_scene_cut: bool
    ocr_text: str = ""
    ocr_tokens: set[str] = field(default_factory=set)


@dataclass
class CueAssignment:
    cue: Beat
    source_time: Optional[float]
    confidence: float
    method: str
    notes: str = ""


def _ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError("ffmpeg not found on PATH and imageio-ffmpeg unavailable") from e


def detect_scene_cuts(video_path: str | Path, threshold: float = 0.3) -> list[float]:
    """Return scene-cut timestamps (seconds) from ffmpeg's scene filter."""
    ffmpeg = _ffmpeg_binary()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i", str(video_path),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return sorted({float(m.group(1)) for m in _SCENE_RE.finditer(proc.stderr)})


def probe_duration(video_path: str | Path) -> float:
    """Return video duration in seconds via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        from moviepy.editor import VideoFileClip
        with VideoFileClip(str(video_path)) as clip:
            return float(clip.duration)
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video_path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def _extract_frame(video_path: str | Path, t: float, out_path: Path) -> bool:
    """Extract a single frame at time t to out_path. Returns True on success."""
    ffmpeg = _ffmpeg_binary()
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{t:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "3",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.returncode == 0 and out_path.exists()


def _ocr_image(image_path: Path, lang: str = "eng") -> str:
    """Run Tesseract via pytesseract; empty string if unavailable or fails."""
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return ""
    try:
        with Image.open(image_path) as im:
            return pytesseract.image_to_string(im, lang=lang) or ""
    except Exception:
        return ""


def _tokens(text: str) -> set[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
    return {w for w in raw if len(w) > 1 and w not in _STOPWORDS}


def _literal_marks(text: str) -> set[str]:
    """Pull out alphanumeric "screen-readable" tokens that ``_tokens`` skips:
    dollar amounts, percentages, version-like identifiers, channel names,
    file numbers (``$48,000``, ``40%``, ``W-2``, ``#ask-hr``, ``v4.21``).
    """
    marks: set[str] = set()
    for m in re.findall(r"[\$#]?[A-Za-z0-9][A-Za-z0-9\-\.,_/]+", text):
        if any(c.isdigit() or c in "$#" for c in m) and len(m) >= 2:
            marks.add(m.lower().strip(",."))
    return marks


@dataclass
class CueSignals:
    quoted: list[str]
    quoted_tokens: list[set[str]]
    tokens: set[str]
    literals: set[str]


def _cue_signals(cue_text: str) -> CueSignals:
    """Decompose a cue into matchable signals.

    Quoted phrases become strong "must roughly appear on screen" anchors.
    Bare words become token-based fallback signals. Numerics/symbols become
    literal substring searches.
    """
    body = re.sub(r"(?i)^\s*cut\s+to\s+", "", cue_text)
    quoted = [m.strip() for m in re.findall(r'"([^"]+)"', body) if m.strip()]
    bare = re.sub(r'"[^"]*"', " ", body)

    quoted_tokens = [_tokens(q) for q in quoted]
    bare_tokens = _tokens(bare)
    literal_pool = bare + " " + " ".join(quoted)
    literals = _literal_marks(literal_pool)

    return CueSignals(
        quoted=quoted,
        quoted_tokens=[t for t in quoted_tokens if t],
        tokens=bare_tokens | {t for ts in quoted_tokens for t in ts},
        literals=literals,
    )


def gather_candidates(
    video_path: str | Path,
    duration: float,
    config: CueFinderConfig,
    work_dir: Path,
) -> list[CandidateFrame]:
    """Build a sorted list of CandidateFrame entries for the section video."""
    work_dir.mkdir(parents=True, exist_ok=True)

    scene_times = detect_scene_cuts(video_path, threshold=config.scene_threshold)

    fixed_times: list[float] = []
    t = 0.0
    while t < duration:
        fixed_times.append(t)
        t += config.sample_interval
    if duration - 0.1 not in fixed_times:
        fixed_times.append(max(0.0, duration - 0.1))

    scene_set = {round(s, 3) for s in scene_times}
    merged: dict[float, bool] = {}
    for s in scene_times:
        merged[round(s, 3)] = True
    for f in fixed_times:
        key = round(f, 3)
        merged.setdefault(key, False)

    candidates: list[CandidateFrame] = []
    for i, (ts, _) in enumerate(sorted(merged.items())):
        frame_path = work_dir / f"frame_{i:04d}_{ts:.3f}.jpg"
        if not _extract_frame(video_path, ts, frame_path):
            continue
        text = _ocr_image(frame_path, lang=config.ocr_lang)
        candidates.append(
            CandidateFrame(
                time=ts,
                is_scene_cut=round(ts, 3) in scene_set,
                ocr_text=text,
                ocr_tokens=_tokens(text),
            )
        )
    return candidates


def schedule_section_cues(
    section: Section,
    video_path: str | Path,
    config: CueFinderConfig,
    work_dir: Path,
    marker_pattern: str = DEFAULT_MARKER,
) -> list[CueAssignment]:
    """Assign source timestamps to non-section ``visual_cue`` beats in order.

    Returns an entry per relevant cue with source_time, confidence, and method.
    """
    section_pat = re.compile(marker_pattern)
    cues = [
        b for b in section["beats"]
        if b.get("type") == "visual_cue" and not section_pat.search(b.get("content", ""))
    ]
    if not cues:
        return []

    duration = probe_duration(video_path)
    if duration <= 0:
        return [
            CueAssignment(cue=c, source_time=None, confidence=0.0,
                          method="error", notes="zero-duration video")
            for c in cues
        ]

    candidates = gather_candidates(video_path, duration, config, work_dir)
    if not candidates:
        return [
            CueAssignment(cue=c, source_time=None, confidence=0.0,
                          method="error", notes="no candidate frames extracted")
            for c in cues
        ]

    token_weights = _idf_weights(candidates, config.common_token_threshold)

    assignments: list[CueAssignment] = []
    cursor = 0.0
    n_remaining = len(cues)

    for cue_idx, cue in enumerate(cues):
        signals = _cue_signals(cue["content"])
        gap = config.min_cue_gap if assignments else 0.0
        min_t = max(cursor + gap, 0.0 if assignments else config.startup_grace)
        usable = sorted(
            [c for c in candidates if c.time >= min_t],
            key=lambda c: c.time,
        )
        if not usable:
            assignments.append(
                CueAssignment(cue=cue, source_time=None, confidence=0.0,
                              method="missing", notes="no candidate frames after cursor")
            )
            continue

        accepted, score, hits = _scan_until_match(
            usable, signals, token_weights,
            accept_threshold=config.accept_threshold,
            rare_token_min_weight=config.rare_token_min_weight,
        )
        method_label = "ocr-scan"

        if accepted is None:
            best_pair = _best_below_threshold(
                usable, signals, token_weights, config.rare_token_min_weight
            )
            if best_pair is not None and best_pair[1] >= config.weak_threshold:
                accepted, score, hits = best_pair
                method_label = "ocr-weak"

        if accepted is not None:
            assignments.append(
                CueAssignment(
                    cue=cue,
                    source_time=accepted.time,
                    confidence=min(1.0, score),
                    method=("ocr+scene" if accepted.is_scene_cut else method_label),
                    notes=_format_hits(hits),
                )
            )
            cursor = accepted.time
        else:
            even_t = _even_fallback(cursor, duration, n_remaining - cue_idx)
            assignments.append(
                CueAssignment(cue=cue, source_time=even_t, confidence=0.15,
                              method="even", notes="no scene/ocr signal; evenly distributed")
            )
            cursor = even_t

    return assignments


def _scan_until_match(
    usable: list[CandidateFrame],
    signals: CueSignals,
    token_weights: dict[str, float],
    accept_threshold: float,
    rare_token_min_weight: float,
) -> tuple[Optional[CandidateFrame], float, dict]:
    """Walk frames in time order; return the FIRST one whose match score
    crosses ``accept_threshold``. Mimics "watch the video until the action
    appears, then mark it"."""
    best_below = (None, 0.0, {})
    for cand in usable:
        score, hits = _score_frame(cand, signals, token_weights, rare_token_min_weight)
        if score >= accept_threshold:
            return cand, score, hits
        if score > best_below[1]:
            best_below = (cand, score, hits)
    return None, 0.0, {}


def _best_below_threshold(
    usable: list[CandidateFrame],
    signals: CueSignals,
    token_weights: dict[str, float],
    rare_token_min_weight: float,
) -> Optional[tuple[CandidateFrame, float, dict]]:
    best: Optional[tuple[CandidateFrame, float, dict]] = None
    for cand in usable:
        score, hits = _score_frame(cand, signals, token_weights, rare_token_min_weight)
        if score <= 0:
            continue
        if best is None or score > best[1]:
            best = (cand, score, hits)
    return best


def _score_frame(
    cand: CandidateFrame,
    signals: CueSignals,
    token_weights: dict[str, float],
    rare_token_min_weight: float,
) -> tuple[float, dict]:
    """Composite score for one frame. Returns (score, hits dict for notes)."""
    score = 0.0
    hits: dict[str, list[str]] = {"quoted": [], "tokens": [], "literals": []}
    ocr_lower = cand.ocr_text.lower()

    # Quoted phrases — strongest signal. Use word-overlap ratio so OCR errors
    # don't kill the match outright.
    for original, q_tokens in zip(signals.quoted, signals.quoted_tokens):
        overlap = q_tokens & cand.ocr_tokens
        if not overlap:
            continue
        ratio = len(overlap) / max(1, len(q_tokens))
        if ratio >= 0.5:
            score += 0.55 * ratio
            hits["quoted"].append(original)
        elif ratio >= 0.25:
            score += 0.25 * ratio
            hits["quoted"].append(f"~{original}")

    # Bare-token IDF score (gated by at least one rare token to avoid chrome).
    if signals.tokens:
        token_overlap = signals.tokens & cand.ocr_tokens
        rare = [t for t in token_overlap if token_weights.get(t, 1.0) >= rare_token_min_weight]
        if rare:
            total = sum(token_weights.get(t, 1.0) for t in signals.tokens) or 1.0
            base = sum(token_weights.get(t, 1.0) for t in token_overlap) / total
            score += 0.3 * base
            hits["tokens"] = rare

    # Literal numerics / symbols — exact substrings in OCR text.
    for lit in signals.literals:
        if lit in ocr_lower:
            score += 0.2
            hits["literals"].append(lit)

    if cand.is_scene_cut:
        score += 0.1

    return min(1.0, score), hits


def _format_hits(hits: dict) -> str:
    parts: list[str] = []
    if hits.get("quoted"):
        parts.append("quoted: " + " | ".join(hits["quoted"]))
    if hits.get("tokens"):
        parts.append("tokens: " + ", ".join(sorted(hits["tokens"])))
    if hits.get("literals"):
        parts.append("literals: " + ", ".join(sorted(hits["literals"])))
    return "; ".join(parts) if parts else "-"


def _idf_weights(
    candidates: list[CandidateFrame],
    common_threshold: float,
) -> dict[str, float]:
    """Compute IDF-style weights so frequent UI chrome ("Pay" tab visible
    on every frame) doesn't dominate match scores. Tokens appearing in more
    than ``common_threshold`` of frames get near-zero weight.
    """
    n = max(1, len(candidates))
    freq = Counter()
    for cand in candidates:
        freq.update(cand.ocr_tokens)
    weights: dict[str, float] = {}
    for token, count in freq.items():
        ratio = count / n
        if ratio >= common_threshold:
            weights[token] = 0.05
        else:
            weights[token] = math.log((1 + n) / (1 + count)) + 0.5
    return weights


def _even_fallback(cursor: float, total_duration: float, remaining: int) -> float:
    """Spread remaining cues evenly across whatever source time is left."""
    if remaining <= 0:
        return min(total_duration, cursor + 0.5)
    span = max(0.0, total_duration - cursor)
    step = span / max(1, remaining + 1)
    return min(total_duration, cursor + step)


def build_cue_plan(
    sections_with_videos: list[tuple[Section, str]],
    config: Optional[CueFinderConfig] = None,
    work_root: Optional[Path] = None,
) -> dict:
    """Run cue scheduling for every section; return a JSON-serializable plan."""
    config = config or CueFinderConfig()
    if work_root is None:
        work_root = Path(tempfile.mkdtemp(prefix="worknet_cues_"))
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    plan_sections: list[dict] = []
    for i, (section, video_path) in enumerate(sections_with_videos, start=1):
        section_dir = work_root / f"section_{i:02d}"
        assignments = schedule_section_cues(
            section, video_path, config, section_dir
        )
        for a in assignments:
            if a.source_time is not None:
                a.cue["source_time"] = float(a.source_time)
                a.cue["cue_confidence"] = float(a.confidence)
                a.cue["cue_method"] = a.method

        plan_sections.append({
            "index": i,
            "video": video_path,
            "duration": probe_duration(video_path),
            "section_marker": (section["start_cue"]["content"]
                               if section["start_cue"] else None),
            "assignments": [
                {
                    "cue": a.cue["content"],
                    "source_time": a.source_time,
                    "confidence": a.confidence,
                    "method": a.method,
                    "notes": a.notes,
                }
                for a in assignments
            ],
        })

    return {"sections": plan_sections}


def render_timestamped_script(beats: list[Beat]) -> str:
    """Render the original beats with inferred ``@ Ns`` annotations on cues."""
    lines: list[str] = []
    for b in beats:
        if b.get("type") == "visual_cue":
            ts = b.get("source_time")
            content = b.get("content", "")
            if ts is not None:
                conf = b.get("cue_confidence")
                conf_str = f", conf {conf:.2f}" if conf is not None else ""
                lines.append(f"[{content} @ {float(ts):.2f}s{conf_str}]")
            else:
                lines.append(f"[{content}]")
        elif b.get("type") == "spoken_text":
            text = b.get("content", "").strip()
            # Don't double-wrap quotes when the source already had them.
            if text.startswith('"') and text.endswith('"'):
                lines.append(text)
            else:
                lines.append(f'"{text}"')
    return "\n".join(lines) + "\n"
