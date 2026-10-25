#!/bin/zsh
# Python writer demo: the app, plus one terminal (Ghostty if installed) with macmon (GPU / ANE / power; asitop as a
# fallback) on top and the live model log below.
#   Sources/CodeWriterDemo/demo.sh [app options, e.g. --autostart]
# The model folder comes from CODE_WRITER_MODEL_DIR (default ~/Library/Application Support/FluidUse/qwen2.5-coder-0.5b-coreml).
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOG=${TMPDIR:-/tmp}/code-writer-demo.log
# macmon (brew install macmon) needs no sudo and works on M5; asitop 0.0.24 crashes there (KeyError E0-Cluster_active).
if command -v macmon >/dev/null; then
  MONITOR="$(command -v macmon)"
else
  MONITOR="sudo sh -c 'pkill -x powermetrics; exec $(command -v asitop || echo "$HOME/.local/bin/asitop")'"
fi

swift build -c release --product CodeWriterDemo --package-path "$ROOT"
: > "$LOG"
# One tmux session, reused across launches: the monitor on top, the log below.
if ! tmux has-session -t code-writer 2>/dev/null; then
  # The pane stays open if the monitor exits, so the reason is visible.
  tmux new-session -d -s code-writer -x 160 -y 70 \
    "$MONITOR; echo; echo \"monitor exited (\$?) — press Enter to retry\"; read; $MONITOR"
  tmux set-option -t code-writer remain-on-exit on
  tmux split-window -v -t code-writer "tail -n 300 -F '$LOG'"
  if [[ -d /Applications/Ghostty.app ]]; then
    open -na Ghostty --args -e "$(command -v tmux)" attach -t code-writer
  else
    osascript -e 'tell application "Terminal" to do script "tmux attach -t code-writer"' \
      -e 'tell application "Terminal" to activate' >/dev/null
  fi
fi
pkill -x CodeWriterDemo 2>/dev/null || true
cd "$ROOT"
"$ROOT/.build/release/CodeWriterDemo" "$@" > "$LOG" 2> "$LOG.stderr" &
echo "demo pid $! · log $LOG · Play (⌘↩) · Replay (⌘R)"
