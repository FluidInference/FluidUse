#!/bin/zsh
# One Ghostty window, two rows: macmon (chip power/usage) on top, the colored per-comment log below.
LOG=${MODERATION_DEMO_LOG:-/tmp/moderation-demo.log}
touch "$LOG"
tmux kill-session -t moderation 2>/dev/null
exec tmux new-session -s moderation 'macmon' \; set remain-on-exit on \; \
    split-window -v "tail -n 0 -F $LOG" \; select-pane -t 0
