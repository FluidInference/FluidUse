#!/bin/zsh
# Showdown demo: local server, the fine-tuned 0.8B playing in the browser, and one Ghostty window with a GPU/power
# monitor (macmon) on top and the live decision log below.
#   ./demo.sh <student coreml dir> <merged student dir> [opponent] [battles]
set -e
MODEL=$1
MERGED=$2
OPP=${3:-base}
BATTLES=${4:-1}
GO=${TMPDIR:-/tmp}/intern-decision-go
BASE_DIR=${BASE_DIR:-$HOME/Documents/intern-decision-showdown-student/base-coreml}
BASE_CK=${BASE_CK:-$HOME/.cache/huggingface/hub/models--internlm--Intern-Decision-0.8B/snapshots/85a0cc5a99d67ea8d56dfe98115689212867171d}
TEACHER_CK=${TEACHER_CK:-$HOME/.cache/huggingface/hub/models--internlm--Intern-Decision-4B/snapshots/0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd}
HERE=$(cd "$(dirname "$0")" && pwd)
SERVER=${SHOWDOWN_SERVER:-$HOME/Documents/pokemon-showdown}
LOG=${TMPDIR:-/tmp}/intern-decision-showdown.log
# macmon (no sudo, works on M5) is preferred; asitop 0.0.24 crashes parsing M5 Pro powermetrics output.
if command -v macmon >/dev/null; then MONITOR="macmon"; else MONITOR="sudo sh -c 'pkill -x powermetrics; exec $(command -v asitop)'"; fi
ENV=(--with poke-env --with torch==2.9.1 --with torchvision==0.24.1 --with transformers==5.14.1 --with Pillow --with safetensors --with coremltools==9.0 --with 'numpy<2.3')

if ! curl -s -o /dev/null http://localhost:8000/; then
  (cd "$SERVER" && nohup node pokemon-showdown start --no-security > "$SERVER/server.log" 2>&1 &)
  for i in $(seq 1 60); do curl -s -o /dev/null http://localhost:8000/ && break; sleep 1; done
fi
: > "$LOG"
if ! tmux has-session -t intern-decision-showdown 2>/dev/null; then
  tmux new-session -d -s intern-decision-showdown -x 160 -y 70 "$MONITOR"
  tmux split-window -v -t intern-decision-showdown "tail -n 300 -F '$LOG'"
  if [[ -d /Applications/Ghostty.app ]]; then
    open -na Ghostty --args -e "$(command -v tmux)" attach -t intern-decision-showdown
  else
    osascript -e 'tell application "Terminal" to do script "tmux attach -t intern-decision-showdown"' \
      -e 'tell application "Terminal" to activate' >/dev/null
  fi
fi
pkill -f "demo_battle.py" 2>/dev/null || true
cd "$HERE"
uv run --no-project --python 3.12 "${ENV[@]}" python -u demo_battle.py --model-dir "$MODEL" --checkpoint "$MERGED" \
  --opponent "$OPP" --battles "$BATTLES" --base-dir "$BASE_DIR" --base-checkpoint "$BASE_CK" \
  --teacher-checkpoint "$TEACHER_CK" --wait-for "$GO" > "$LOG" 2> "$LOG.stderr" &
echo "demo pid $! · log $LOG · when the log says ready, start the battle with ./go.sh"
