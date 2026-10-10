#!/bin/zsh
# clef-flash triage demo, plus one Ghostty window: live CPU / GPU bar charts (bars.py over macmon) on top and the
# decision log below. (asitop crashes on M5 Pro; CPU / ANE power counters read 0 W on macOS 27, so bars show usage.)
#   Sources/ClefFlashDemo/demo.sh [bundle dir]      (or CLEF_FLASH_BUNDLE; default: download FluidInference/clef-flash-coreml)
#   CLEF_MODEL=text Sources/ClefFlashDemo/demo.sh   clef-text-0.6b on the Neural Engine (CLEF_TEXT_BUNDLE or download)
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
  tmux new-session -d -s clef-flash -x 160 -y 60 "python3 '$ROOT/Sources/ClefFlashDemo/bars.py'"
  tmux set-option -t clef-flash remain-on-exit on
  tmux split-window -v -t clef-flash "tail -n 300 -F '$LOG'"
  # the bars need 10 rows whatever size the terminal window attaches with
  tmux set-hook -t clef-flash client-attached 'resize-pane -t clef-flash:0.0 -y 10'
  tmux set-hook -t clef-flash client-resized 'resize-pane -t clef-flash:0.0 -y 10'
  open -na /Applications/Ghostty.app --args -e "$(command -v tmux)" attach -t clef-flash
fi
pkill -x ClefFlashDemo 2>/dev/null || true
CLEF_MODEL="${CLEF_MODEL:-}" CLEF_TEXT_BUNDLE="${CLEF_TEXT_BUNDLE:-}" CLEF_FLASH_BUNDLE="$BUNDLE" "$ROOT/.build/release/ClefFlashDemo" > "$LOG" 2> "$LOG.stderr" &
echo "demo pid $! · log $LOG · first ticket after ~40 s of model loading (plus an ~11 GB download on first run)"
