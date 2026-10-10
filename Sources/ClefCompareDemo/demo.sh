#!/bin/zsh
# clef-flash 9B vs clef-text 0.6B comparison demo, plus one Ghostty window with macmon (GPU / power) on top and the live decision log below.
#   Sources/ClefCompareDemo/demo.sh     (CLEF_FLASH_BUNDLE / CLEF_TEXT_BUNDLE for local bundles; default: download both)
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUNDLE=${CLEF_FLASH_BUNDLE:-}
LOG=${TMPDIR:-/tmp}/clef-compare-demo.log

[[ -z "$BUNDLE" || -f "$BUNDLE/config.json" ]] || { echo "no clef-flash bundle at $BUNDLE"; exit 1; }
# the 9B weights need the GPU and ~7 GB of RAM to themselves; other model jobs make every ticket page
if pgrep -f "reference.py|run_coreml.py|export.py" >/dev/null; then
  echo "warning: a model-lab job is running and will slow the demo (pgrep -fl 'reference.py|run_coreml.py|export.py')"
fi
swift build -c release --product ClefCompareDemo --package-path "$ROOT"
: > "$LOG"
if ! tmux has-session -t clef-compare 2>/dev/null; then
  tmux new-session -d -s clef-compare -x 160 -y 60 macmon
  tmux set-option -t clef-compare remain-on-exit on
  tmux split-window -v -l 60% -t clef-compare "tail -n 300 -F '$LOG'"
  open -na /Applications/Ghostty.app --args -e "$(command -v tmux)" attach -t clef-compare
fi
pkill -x ClefCompareDemo 2>/dev/null || true
CLEF_FLASH_BUNDLE="$BUNDLE" CLEF_TEXT_BUNDLE="${CLEF_TEXT_BUNDLE:-}" "$ROOT/.build/release/ClefCompareDemo" > "$LOG" 2> "$LOG.stderr" &
echo "demo pid $! · log $LOG · first ticket after ~40 s of model loading (plus ~12 GB of downloads on first run)"
