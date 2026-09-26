#!/bin/zsh
# Guess Who demo, plus a terminal (Ghostty if installed) with asitop (GPU / ANE / power) and the live model log side by side.
#   Sources/KevGuessWhoDemo/demo.sh [model dir]      (or KEV_MODEL_DIR)
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=${1:-${KEV_MODEL_DIR:-$HOME/Documents/mobius-kev-0.8b/models/computer-use/kev-0.8b/coreml/build/kev-demo-model}}
LOG=${TMPDIR:-/tmp}/kev-guess-who.log
ASITOP=$(command -v asitop || echo "$HOME/.local/bin/asitop")

swift build -c release --product KevGuessWhoDemo --package-path "$ROOT"
: > "$LOG"
# One tmux session, reused across launches (asitop keeps its sudo): asitop on top, the model log below.
if ! tmux has-session -t kev-guess-who 2>/dev/null; then
  # asitop reads powermetrics: the top pane asks for the sudo password
  tmux new-session -d -s kev-guess-who -x 160 -y 70 "sudo sh -c 'pkill -x powermetrics; exec $ASITOP'"
  tmux split-window -v -t kev-guess-who "tail -n 300 -F '$LOG'"
  if [[ -d /Applications/Ghostty.app ]]; then
    open -na Ghostty --args -e "$(command -v tmux)" attach -t kev-guess-who
  else
    osascript -e 'tell application "Terminal" to do script "tmux attach -t kev-guess-who"' \
      -e 'tell application "Terminal" to activate' >/dev/null
  fi
fi
pkill -f "release/KevGuessWhoDemo" 2>/dev/null || true
KEV_MODEL_DIR="$MODEL" "$ROOT/.build/release/KevGuessWhoDemo" > "$LOG" 2> "$LOG.stderr" &
echo "demo pid $! · log $LOG"
