#!/bin/zsh
# Unlimited-OCR reader demo, plus one Ghostty window: live CPU / GPU bars (ClefFlashDemo/bars.py over macmon, no sudo;
# asitop crashes on M5 Pro / macOS 27) on top and the demo's block-by-block log below.
#   Sources/UnlimitedOcrDemo/demo.sh [doc.pdf]      (UNLIMITED_OCR_BUNDLE for a local bundle; default: Hub download)
#   Turbo decoder by default; UNLIMITED_OCR_TURBO=0 for the 8-bit one.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PDF=${1:-${UNLIMITED_OCR_PDF:-}}
LOG=${TMPDIR:-/tmp}/unlimited-ocr-demo.log
XCODE=/Applications/Xcode-27.0.0.app/Contents/Developer
[[ -d "$XCODE" ]] && export DEVELOPER_DIR=$XCODE  # MLX kernels need Xcode's Metal toolchain

swift build -c release --product UnlimitedOcrDemo --package-path "$ROOT"
: > "$LOG"
if ! tmux has-session -t unlimited-ocr 2>/dev/null; then
  tmux new-session -d -s unlimited-ocr -x 160 -y 60 "python3 '$ROOT/Sources/ClefFlashDemo/bars.py'"
  tmux set-option -t unlimited-ocr remain-on-exit on
  tmux split-window -v -t unlimited-ocr "tail -n 300 -F '$LOG'"
  # the bars need 10 rows whatever size the terminal window attaches with
  tmux set-hook -t unlimited-ocr client-attached 'resize-pane -t unlimited-ocr:0.0 -y 10'
  tmux set-hook -t unlimited-ocr client-resized 'resize-pane -t unlimited-ocr:0.0 -y 10'
fi
# open the terminal window unless one is already attached (e.g. it was closed since the last run)
if [[ -z "$(tmux list-clients -t unlimited-ocr 2>/dev/null)" ]]; then
  open -na /Applications/Ghostty.app --args -e "$(command -v tmux)" attach -t unlimited-ocr
fi
pkill -x UnlimitedOcrDemo 2>/dev/null || true
UNLIMITED_OCR_PDF="$PDF" UNLIMITED_OCR_BUNDLE="${UNLIMITED_OCR_BUNDLE:-}" UNLIMITED_OCR_TURBO="${UNLIMITED_OCR_TURBO:-1}" \
  "$ROOT/.build/release/UnlimitedOcrDemo" > "$LOG" 2> "$LOG.stderr" &
echo "demo pid $! · log $LOG · press Start (Return) in the app"
