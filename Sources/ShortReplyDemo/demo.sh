#!/bin/zsh
# Short Reply demo: the menu-bar app, plus a terminal (Ghostty if installed) with macmon/asitop (GPU / ANE / power) on top
# and the live model log below.
#   Sources/ShortReplyDemo/demo.sh [--x | --mock] [model dir]   (or SHORT_REPLY_MODEL_DIR; default .mobius/short-reply-lm/coreml-demo)
#   --x     also opens x.com in Chrome (real feed; draft replies, don't press Post)
#   --mock  also opens the local mock feed page (fictional posts, nothing is posted) in Chrome
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BROWSER="Google Chrome"
OPEN=""
case "${1:-}" in
  --x) OPEN="https://x.com/home"; shift ;;
  --mock) OPEN="$ROOT/Sources/ShortReplyDemo/mock-feed/index.html"; shift ;;
esac
MODEL=${1:-${SHORT_REPLY_MODEL_DIR:-$ROOT/.mobius/short-reply-lm/coreml-demo}}
LOG=${TMPDIR:-/tmp}/short-reply-demo.log
# macmon (brew install macmon) needs no sudo and works on M5; asitop 0.0.24 crashes there (KeyError E0-Cluster_active).
if command -v macmon >/dev/null; then
  MONITOR="$(command -v macmon)"
else
  MONITOR="sudo sh -c 'pkill -x powermetrics; exec $(command -v asitop || echo "$HOME/.local/bin/asitop")'"
fi

swift build -c release --product ShortReplyDemo --package-path "$ROOT"
: > "$LOG"
# One tmux session, reused across launches: the monitor on top, the model log below.
if ! tmux has-session -t short-reply 2>/dev/null; then
  # The pane stays open if the monitor exits, so the reason is visible.
  tmux new-session -d -s short-reply -x 160 -y 70 \
    "$MONITOR; echo; echo \"monitor exited (\$?) — press Enter to retry\"; read; $MONITOR"
  tmux set-option -t short-reply remain-on-exit on
  tmux split-window -v -t short-reply "tail -n 300 -F '$LOG'"
  if [[ -d /Applications/Ghostty.app ]]; then
    open -na Ghostty --args -e "$(command -v tmux)" attach -t short-reply
  else
    osascript -e 'tell application "Terminal" to do script "tmux attach -t short-reply"' \
      -e 'tell application "Terminal" to activate' >/dev/null
  fi
fi
pkill -f "release/ShortReplyDemo" 2>/dev/null || true
cd "$ROOT"
SHORT_REPLY_MODEL_DIR="$MODEL" "$ROOT/.build/release/ShortReplyDemo" > "$LOG" 2> "$LOG.stderr" &
if [[ -n "$OPEN" ]]; then open -a "$BROWSER" "$OPEN"; fi
echo "demo pid $! · log $LOG · select a post anywhere and press 9 (or ⌃⌥R)"
