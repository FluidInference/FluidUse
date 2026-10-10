#!/bin/zsh
# JEV-9B plays Super Mario Bros. 1-1, plus one Ghostty window: live CPU / GPU bars (ClefFlashDemo/bars.py over macmon)
# on top and the decision log below.
#   Sources/JevMarioDemo/demo.sh [bundle dir]      (or JEV_MODEL_DIR; default: download FluidInference/jev-9b-coreml)
# The emulator bridge runs through uv with gym-super-mario-bros (it bundles the NES ROM; local use only).
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUNDLE=${1:-${JEV_MODEL_DIR:-}}
LOG=${TMPDIR:-/tmp}/jev-mario-demo.log

[[ -z "$BUNDLE" || -f "$BUNDLE/config.json" ]] || { echo "no jev-9b-coreml bundle at $BUNDLE"; exit 1; }
command -v uv >/dev/null || { echo "uv is required for the emulator bridge (https://docs.astral.sh/uv/)"; exit 1; }
swift build -c release --product JevMarioDemo --package-path "$ROOT"
: > "$LOG"
if ! tmux has-session -t jev-mario 2>/dev/null; then
  tmux new-session -d -s jev-mario -x 160 -y 60 "python3 '$ROOT/Sources/ClefFlashDemo/bars.py'"
  tmux set-option -t jev-mario remain-on-exit on
  tmux split-window -v -t jev-mario "tail -n 300 -F '$LOG'"
  tmux set-hook -t jev-mario client-attached 'resize-pane -t jev-mario:0.0 -y 10'
  tmux set-hook -t jev-mario client-resized 'resize-pane -t jev-mario:0.0 -y 10'
  open -na /Applications/Ghostty.app --args -e "$(command -v tmux)" attach -t jev-mario
fi
pkill -x JevMarioDemo 2>/dev/null || true
JEV_MODEL_DIR="$BUNDLE" "$ROOT/.build/release/JevMarioDemo" > "$LOG" 2> "$LOG.stderr" &
echo "demo pid $! · log $LOG · first move after ~40 s of model loading (plus a ~13 GB download on first run)"
