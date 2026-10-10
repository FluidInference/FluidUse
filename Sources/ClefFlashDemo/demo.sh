#!/bin/zsh
# clef-flash triage demo, plus one Ghostty window with macmon (GPU / power) on top and the live decision log below.
#   Sources/ClefFlashDemo/demo.sh [bundle dir]      (or CLEF_FLASH_BUNDLE; default: download FluidInference/clef-flash-coreml)
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUNDLE=${1:-${CLEF_FLASH_BUNDLE:-}}
LOG=${TMPDIR:-/tmp}/clef-flash-demo.log

[[ -z "$BUNDLE" || -f "$BUNDLE/config.json" ]] || { echo "no clef-flash bundle at $BUNDLE"; exit 1; }
# the 9B weights need the GPU and ~7 GB of RAM to themselves; other model jobs make every ticket page
if pgrep -f "reference.py|run_coreml.py|export.py" >/dev/null; then
  echo "warning: a model-lab job is running and will slow the demo (pgrep -fl 'reference.py|run_coreml.py|export.py')"
fi
swift build -c release --product ClefFlashDemo --package-path "$ROOT"
: > "$LOG"
if ! tmux has-session -t clef-flash 2>/dev/null; then
  tmux new-session -d -s clef-flash -x 160 -y 60 macmon
  tmux set-option -t clef-flash remain-on-exit on
  tmux split-window -v -l 60% -t clef-flash "tail -n 300 -F '$LOG'"
  open -na /Applications/Ghostty.app --args -e "$(command -v tmux)" attach -t clef-flash
fi
pkill -x ClefFlashDemo 2>/dev/null || true
CLEF_FLASH_BUNDLE="$BUNDLE" "$ROOT/.build/release/ClefFlashDemo" > "$LOG" 2> "$LOG.stderr" &
echo "demo pid $! · log $LOG · first ticket after ~40 s of model loading (plus an ~11 GB download on first run)"
