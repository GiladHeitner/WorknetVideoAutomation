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

Each section's source clip is speed-fit to its total voiceover duration, then sliced per spoken beat so word-level overlays stay aligned. If the script has an unmarked intro before the first `[Cut to ...]`, it's automatically merged into the first section so the video count matches.

Outputs land in `out/`:

- `final.mp4` — narrated, sync-stretched video with overlays
- `subs.vtt` — word-grouped captions
- `beats.json` — parsed beats with word timings
- `audio/*.mp3` — cached voiceovers (and `.words.json` Whisper caches)
- `manifest.json` — index of artifact paths

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
```

## How it works

1. Regex parser splits the script into spoken beats and `[visual_cue]` markers.
2. OpenAI `tts-1` generates an mp3 per spoken beat (cached by content hash).
3. Whisper (`whisper-1`, `timestamp_granularities=["word"]`) is run on each mp3 — OpenAI TTS does not return timing, so we transcribe its own output. Results are cached as `<audio>.words.json`.
4. MoviePy `vfx.speedx(final_duration=...)` time-remaps each video segment to match its beat audio, then concatenates.
5. Each `[visual_cue]` overlays its image starting at the end of the preceding beat's last spoken word.

Subtitles are derived from the same word timings, grouped into short readable lines.

## Phase-by-phase debugging

Each module is runnable on its own:

```bash
python -m src.parser    examples/sample_script.txt
python -m src.tts       examples/sample_script.txt out/audio
python -m src.alignment examples/sample_script.txt out/audio
```

## Extending

`src/scheduling.py` is the Phase 6+ extension point for ffmpeg scene detection and VLM-driven cue planning when speed-matching every clip is too coarse.
