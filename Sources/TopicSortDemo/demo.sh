#!/bin/zsh
# Sort by topic demo: the app, plus one terminal (Ghostty if installed) with macmon (GPU / ANE / power; asitop as a
# fallback) on top and the live Neural Engine log below.
#   Sources/TopicSortDemo/demo.sh [app options, e.g. --autostart --auto-split --rate=200]
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOG=${TMPDIR:-/tmp}/topic-sort-demo.log
# macmon (brew install macmon) needs no sudo and works on M5; asitop 0.0.24 crashes there (KeyError E0-Cluster_active).
if command -v macmon >/dev/null; then
  MONITOR="$(command -v macmon)"
else
  MONITOR="sudo sh -c 'pkill -x powermetrics; exec $(command -v asitop || echo "$HOME/.local/bin/asitop")'"
fi

swift build -c release --product TopicSortDemo --package-path "$ROOT"
: > "$LOG"
# One tmux session, reused across launches: the monitor on top, the log below.
if ! tmux has-session -t topic-sort 2>/dev/null; then
  # The pane stays open if the monitor exits, so the reason is visible.
  tmux new-session -d -s topic-sort -x 160 -y 70 \
    "$MONITOR; echo; echo \"monitor exited (\$?) — press Enter to retry\"; read; $MONITOR"
  tmux set-option -t topic-sort remain-on-exit on
  tmux split-window -v -t topic-sort "tail -n 300 -F '$LOG'"
  if [[ -d /Applications/Ghostty.app ]]; then
    open -na Ghostty --args -e "$(command -v tmux)" attach -t topic-sort
  else
    osascript -e 'tell application "Terminal" to do script "tmux attach -t topic-sort"' \
      -e 'tell application "Terminal" to activate' >/dev/null
  fi
fi
pkill -f "release/TopicSortDemo" 2>/dev/null || true
cd "$ROOT"
"$ROOT/.build/release/TopicSortDemo" "$@" > "$LOG" 2> "$LOG.stderr" &
echo "demo pid $! · log $LOG · press Start (Space) in the app"
