# Worknet Video Automation

Script + screen recording → narrated MP4 with word-accurate subtitles and cue overlays. Built on OpenAI TTS, Whisper word timestamps, and MoviePy.

## Quickstart

```bash
./run.sh --dry-run      # one-shot: builds venv, installs deps, previews
./run.sh                # renders examples/sample_script.txt with examples/*.mp4
```

Manual mode:

```bash
make setup
echo "OPENAI_API_KEY=sk-..." > .env

python -m src examples/sample_script.txt --dry-run
python -m src examples/sample_script.txt my_screen_recording.mp4
```

## Multi-video sections

When the script uses `[Cut to ...]` cues to mark scene changes, hand one source video per section:

```bash
python -m src script.txt --videos-dir examples/        # auto-pick by sorted filename
python -m src script.txt --videos a.mp4 b.mp4 c.mp4    # explicit order
```

By default the pipeline analyzes each section's video, infers when each bracketed action happens, and uses those timestamps as anchors so the action lands immediately before its narration. If no timestamps can be inferred, the section's source clip is uniformly stretched to its total voiceover duration. If the script has an unmarked intro before the first `[Cut to ...]`, it's automatically merged into the first section so the video count matches.

Outputs land in `out/`:

- `final.mp4` — narrated, sync-stretched video with overlays
- `subs.vtt` — word-grouped captions
- `beats.json` — parsed beats with word timings
- `cue_plan.json` — inferred per-cue source timestamps, confidence, and method
- `timestamped_script.txt` — your script with `@ Ns` annotations on each bracket
- `audio/*.mp3` — cached voiceovers (and `.words.json` Whisper caches)
- `manifest.json` — index of artifact paths

## Automatic cue sync

When section videos are provided, the pipeline:

1. Parses your script into spoken beats and bracket actions.
2. Generates voiceovers via OpenAI TTS and word timings via Whisper.
3. For each section video, samples frames (every `--cue-step` seconds, default `1.0`) plus ffmpeg scene cuts and runs Tesseract OCR on each frame.
4. Greedy-matches each bracketed action against OCR text and scene-change signals to choose a source timestamp; falls back to scene-only or evenly-distributed when no signal is available.
5. Saves the inferred plan to `out/cue_plan.json` and an annotated script to `out/timestamped_script.txt` before rendering.
6. Builds the final video by speed-fitting each interval between anchors so the action shows up just before the next line of narration.

Tesseract is used locally for OCR, so install it once: `brew install tesseract` (macOS) or `apt-get install tesseract-ocr` (Debian/Ubuntu).

Disable inference and fall back to uniform stretching with `--no-auto-cue`. Tune frame sampling/scene sensitivity with `--cue-step 0.5` and `--cue-scene-threshold 0.2`.

## Cue overlays — drop files, no config

When the script contains `[nudge appears]`, the renderer looks in `assets/` for an image whose name matches the cue. These all work:

- `assets/nudge appears.png`
- `assets/nudge_appears.png`
- `.webp` / `.jpg` / `.jpeg` / `.gif` also accepted

Override the directory with `--cues-dir my_overlays/`, or pass an explicit JSON map via `--cue-assets`.

## Common flags

```bash
python -m src script.txt video.mp4 --voice onyx     # different TTS voice
python -m src script.txt video.mp4 --subs srt       # SRT instead of WebVTT
python -m src script.txt video.mp4 --subs none      # no subtitle sidecar
python -m src script.txt --video-list '["a.mp4","b.mp4","c.mp4"]'  # one clip per spoken beat
python -m src script.txt --videos-dir examples/ --no-auto-cue       # disable cue inference
python -m src script.txt --videos-dir examples/ --cue-step 0.5      # finer cue sampling
```

## How it works

1. Regex parser splits the script into spoken beats and `[visual_cue]` markers.
2. OpenAI `tts-1` generates an mp3 per spoken beat (cached by content hash).
3. Whisper (`whisper-1`, `timestamp_granularities=["word"]`) is run on each mp3 — OpenAI TTS does not return timing, so we transcribe its own output. Results are cached as `<audio>.words.json`.
4. For section videos, `cue_finder` runs ffmpeg scene detection plus Tesseract OCR on sampled frames and assigns a source timestamp to each bracketed action.
5. MoviePy `vfx.speedx(final_duration=...)` time-remaps each interval between cue anchors to match the corresponding voiceover slot, then concatenates.
6. Each `[visual_cue]` overlays its image starting at the end of the preceding beat's last spoken word.

Subtitles are derived from the same word timings, grouped into short readable lines.

## Phase-by-phase debugging

Each module is runnable on its own:

```bash
python -m src.parser    examples/sample_script.txt
python -m src.tts       examples/sample_script.txt out/audio
python -m src.alignment examples/sample_script.txt out/audio
```

## Extending

`src/cue_finder.py` is the cue-scheduling brain (scene detection + OCR). `src/scheduling.py` is the original VLM extension point if you want to wire a vision model in as a stronger fallback.
