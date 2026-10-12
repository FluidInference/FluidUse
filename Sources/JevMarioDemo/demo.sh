#!/bin/zsh
# JEV-9B plays Super Mario Bros. 1-1, plus one Ghostty window: live CPU / GPU bars (ClefFlashDemo/bars.py over macmon)
# on top and the decision log below.
#   Sources/JevMarioDemo/demo.sh [bundle dir]      (or JEV_MODEL_DIR; default: download FluidInference/jev-9b-coreml)
#   JEV_MARIO_BRAIN=student MARIO_STUDENT_MODEL=<.mlpackage> Sources/JevMarioDemo/demo.sh   the small pixel student
#   MONITOR=ane ...   bars also show Neural Engine power via `sudo powermetrics` (type your password in the top pane;
#                     asitop 0.0.24 crashes on M5 Pro: no E0-Cluster in powermetrics)
# The emulator bridge runs through uv with gym-super-mario-bros (it bundles the NES ROM; local use only).
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BUNDLE=${1:-${JEV_MODEL_DIR:-}}
LOG=${TMPDIR:-/tmp}/jev-mario-demo.log

[[ -z "$BUNDLE" || -f "$BUNDLE/config.json" ]] || { echo "no jev-9b-coreml bundle at $BUNDLE"; exit 1; }
[[ "${JEV_MARIO_BRAIN:-}" != student || -d "${MARIO_STUDENT_MODEL:-}" ]] || { echo "set MARIO_STUDENT_MODEL to the student .mlpackage"; exit 1; }
command -v uv >/dev/null || { echo "uv is required for the emulator bridge (https://docs.astral.sh/uv/)"; exit 1; }
swift build -c release --product JevMarioDemo --package-path "$ROOT"
: > "$LOG"
if ! tmux has-session -t jev-mario 2>/dev/null; then
  TOP="python3 '$ROOT/Sources/ClefFlashDemo/bars.py' --log '$LOG'"
  [[ "${MONITOR:-}" == ane ]] && TOP="$TOP --ane" || true
  tmux new-session -d -s jev-mario -x 160 -y 60 "$TOP"
  tmux set-option -t jev-mario remain-on-exit on
  tmux set-option -t jev-mario window-size latest  # follow the Ghostty window, not the detached 160-column default
  tmux split-window -h -t jev-mario "tail -n 300 -F '$LOG'"  # bars left, decision log right
  open -na /Applications/Ghostty.app --args -e "$(command -v tmux)" attach -t jev-mario
  sleep 1
  GHOSTTY_PID=$(pgrep -n -x ghostty || true)
fi
pkill -x JevMarioDemo 2>/dev/null || true
MARIO_PLAYLIST="${MARIO_PLAYLIST:-}" MARIO_TURBO="${MARIO_TURBO:-}" MARIO_LOOKAHEAD="${MARIO_LOOKAHEAD:-}" MARIO_LEVEL="${MARIO_LEVEL:-1-1}" JEV_MODEL_DIR="$BUNDLE" JEV_MARIO_BRAIN="${JEV_MARIO_BRAIN:-}" MARIO_STUDENT_MODEL="${MARIO_STUDENT_MODEL:-}" \
  "$ROOT/.build/release/JevMarioDemo" > "$LOG" 2> "$LOG.stderr" &
APP_PID=$!
# T layout: the game window centred across the top, the terminal full width underneath (bars | log)
(
  for _ in {1..40}; do
    osascript -e "tell application \"System Events\" to count windows of (first process whose unix id is $APP_PID)" 2>/dev/null \
      | grep -q '^[1-9]' && break
    sleep 0.5
  done
  read -r _ _ SCREEN_W SCREEN_H < <(osascript -e 'tell application "Finder" to get bounds of window of desktop' | tr -d ',')
  MENU=33
  GAME_BOTTOM=$(osascript -e "tell application \"System Events\" to tell front window of (first process whose unix id is $APP_PID)" \
    -e "set {w, h} to size" -e "set position to {($SCREEN_W - w) div 2, $MENU}" -e "return $MENU + h" -e "end tell" 2>/dev/null)
  [[ -n "${GHOSTTY_PID:-}" && -n "$GAME_BOTTOM" ]] || exit 0
  for _ in {1..20}; do  # Ghostty opens its window asynchronously and may re-place it once; re-apply until it sticks
    WIDTH=$(osascript -e "tell application \"System Events\" to tell front window of (first process whose unix id is $GHOSTTY_PID)" \
      -e "set {position, size} to {{0, $GAME_BOTTOM}, {$SCREEN_W, $SCREEN_H - $GAME_BOTTOM}}" -e "return item 1 of (get size)" \
      -e "end tell" 2>/dev/null)
    [[ "$WIDTH" == "$SCREEN_W" ]] && break
    sleep 0.5
  done
) &
echo "demo pid $APP_PID · log $LOG · first move after ~40 s of model loading (plus a ~13 GB download on first run)"
