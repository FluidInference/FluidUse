#!/bin/zsh
# Merged student checkpoint -> Core ML buckets -> Showdown battles with the Core ML student.
#   ./export_student.sh <merged checkpoint dir> <build dir> [battles]
set -euo pipefail
MERGED=$1
BUILD=$2
BATTLES=${3:-10}
HERE=$(cd "$(dirname "$0")" && pwd)
COREML=$HERE/coreml
REF=(--with torch==2.9.1 --with torchvision==0.24.1 --with transformers==5.14.1 --with Pillow --with safetensors)
CVT=(--with torch==2.7.0 --with coremltools==9.0 --with safetensors --with 'numpy<2.3')
HARNESS=(--with poke-env "${REF[@]}" --with coremltools==9.0 --with 'numpy<2.3')

cd "$COREML"
uv run --no-project --python 3.12 "${REF[@]}" python extract_reference.py --checkpoint "$MERGED" --out "$BUILD/merged"
for spec in 512:8 1024:16; do
    uv run --no-project --python 3.12 "${CVT[@]}" python convert-coreml.py --merged "$BUILD/merged" \
        --length ${spec%%:*} --max-fields ${spec##*:} --build "$BUILD"
done
cp "$MERGED/tokenizer.json" "$BUILD/"
cd "$HERE"
uv run --no-project --python 3.12 "${HARNESS[@]}" python run_battles.py --model-dir "$BUILD" --checkpoint "$MERGED" \
    --battles "$BATTLES" --opponents random max_power heuristics --tag=-student --out "$BUILD/runs"
