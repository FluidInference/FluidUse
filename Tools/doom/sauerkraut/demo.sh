#!/bin/zsh
# One-command SauerkrautLM-Doom demo: game window + asitop + decision log terminals.
#   Tools/doom/sauerkraut/demo.sh              # live, seeds from 10016 (25 kills / full 60 s)
#   Tools/doom/sauerkraut/demo.sh --record doom.mp4 --episodes 1
set -e
cd "$(dirname "$0")/../../.."
if [ ! -x .venv/bin/python ]; then
    uv venv -q -p 3.12 .venv
    uv pip install -q -p .venv vizdoom==1.3.0 numpy coremltools pygame imageio imageio-ffmpeg huggingface_hub
fi
if [ ! -d Tools/doom/sauerkraut/models/SauerkrautDoom_L1026_fp16.mlpackage ]; then
    echo "Missing Tools/doom/sauerkraut/models/SauerkrautDoom_L1026_fp16.mlpackage (run convert_ane.py, see README)"
    exit 1
fi
exec .venv/bin/python Tools/doom/sauerkraut/play.py --seed 10016 "$@"
