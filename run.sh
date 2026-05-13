#!/usr/bin/env bash
# One-shot runner: render examples/sample_script.txt against examples/*.mp4
# Usage:
#   ./run.sh                                  # use examples/ defaults
#   ./run.sh path/script.txt path/videos_dir  # custom inputs
#   ./run.sh --dry-run                        # preview only (uses defaults)
#   ./run.sh --voice onyx --subs srt          # any extra flags forwarded

set -euo pipefail
cd "$(dirname "$0")"

SCRIPT="examples/sample_script.txt"
VIDEOS_DIR="examples"
EXTRA_ARGS=()
POSITIONAL=()

for arg in "$@"; do
  case "$arg" in
    --*) EXTRA_ARGS+=("$arg") ;;
    *)   POSITIONAL+=("$arg") ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ge 1 ]]; then SCRIPT="${POSITIONAL[0]}"; fi
if [[ "${#POSITIONAL[@]}" -ge 2 ]]; then VIDEOS_DIR="${POSITIONAL[1]}"; fi

if [[ ! -d .venv ]]; then
  echo ">> creating virtualenv"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import openai, moviepy, dotenv" >/dev/null 2>&1; then
  echo ">> installing requirements"
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo ">> created .env from template — add your OPENAI_API_KEY before running for real"
fi

echo ">> running pipeline"
echo "   script     : $SCRIPT"
echo "   videos-dir : $VIDEOS_DIR"
[[ "${#EXTRA_ARGS[@]}" -gt 0 ]] && echo "   extra      : ${EXTRA_ARGS[*]}"

exec python -m src "$SCRIPT" --videos-dir "$VIDEOS_DIR" --out out ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
