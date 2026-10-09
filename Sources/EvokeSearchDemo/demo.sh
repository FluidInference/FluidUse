#!/bin/zsh
# Evoke search demo: the app, plus a terminal (Ghostty if installed) with macmon/asitop (GPU / ANE / power) on top and
# the live model log below.
#   Sources/EvokeSearchDemo/demo.sh [posts.json]   (or EVOKE_POSTS; default: the built-in 40 posts)
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
POSTS=${1:-${EVOKE_POSTS:-}}
LOG=${TMPDIR:-/tmp}/evoke-search-demo.log
# macmon (brew install macmon) needs no sudo and works on M5; asitop 0.0.24 crashes there (KeyError E0-Cluster_active).
if command -v macmon >/dev/null; then
  MONITOR="$(command -v macmon)"
else
  MONITOR="sudo sh -c 'pkill -x powermetrics; exec $(command -v asitop || echo "$HOME/.local/bin/asitop")'"
fi

swift build -c release --product EvokeSearchDemo --package-path "$ROOT"
: > "$LOG"
# One tmux session, reused across launches: the monitor on top, the model log below.
if ! tmux has-session -t evoke-search 2>/dev/null; then
  # The pane stays open if the monitor exits, so the reason is visible.
  tmux new-session -d -s evoke-search -x 160 -y 70 \
    "$MONITOR; echo; echo \"monitor exited (\$?) — press Enter to retry\"; read; $MONITOR"
  tmux set-option -t evoke-search remain-on-exit on
  tmux split-window -v -t evoke-search "tail -n 300 -F '$LOG'"
  if [[ -d /Applications/Ghostty.app ]]; then
    # Ghostty on macOS runs `-e` through a launcher script reliably; a bare `-e tmux attach` opened a plain shell.
    ATTACH=${TMPDIR:-/tmp}/evoke-search-attach.sh
    printf '#!/bin/zsh\nexec %s attach -t evoke-search\n' "$(command -v tmux)" > "$ATTACH"
    chmod +x "$ATTACH"
    open -na /Applications/Ghostty.app --args -e /bin/zsh "$ATTACH"
  else
    osascript -e 'tell application "Terminal" to do script "tmux attach -t evoke-search"' \
      -e 'tell application "Terminal" to activate' >/dev/null
  fi
fi
pkill -f "release/EvokeSearchDemo" 2>/dev/null || true
cd "$ROOT"
if [[ -n "$POSTS" ]]; then
  "$ROOT/.build/release/EvokeSearchDemo" --posts "$POSTS" > "$LOG" 2> "$LOG.stderr" &
else
  "$ROOT/.build/release/EvokeSearchDemo" > "$LOG" 2> "$LOG.stderr" &
fi
echo "demo pid $! · log $LOG"
