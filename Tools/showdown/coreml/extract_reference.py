"""Save the Intern-Decision language backbone for export: fp32 text weights with Qwen3_5TextModel-relative names,
the tied embedding table, and meta.json (text config, marker id, answer-symbol ids, calibration temperature).

    uv run --no-project --python 3.12 --with torch==2.9.1 --with transformers==5.14.1 --with safetensors \
        python extract_reference.py --checkpoint <hf snapshot dir> --out build/merged
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from decision_export import load_inference_module

PREFIX = "model.language_model."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("build/merged"))
    args = ap.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    index_path = args.checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
        shards = sorted({v for k, v in index.items() if k.startswith(PREFIX)})
    else:  # single-file checkpoint (e.g. a merged LoRA student saved by save_pretrained)
        shards = ["model.safetensors"]
    state = {}
    for shard in shards:
        for key, value in load_file(str(args.checkpoint / shard)).items():
            if key.startswith(PREFIX):
                state[key[len(PREFIX):]] = value.float().contiguous()
    dropped = [k for k in state if k.startswith("mtp")]
    for k in dropped:
        del state[k]
    tok = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True)
    inference = load_inference_module(args.checkpoint)
    symbol_ids = []
    for s in inference.ANSWER_SYMBOLS:
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, (s, ids)
        symbol_ids.append(ids[0])
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(state, str(args.out / "text.safetensors"))
    digest = hashlib.sha256()
    for shard in shards:
        digest.update((args.checkpoint / shard).read_bytes())
    lm_hash = digest.hexdigest()
    meta = {"checkpoint": args.checkpoint.name, "model_name": inference.MODEL_NAME, "language_shard_sha256": lm_hash,
            "language_shards": shards,
            "temperature": inference.DEFAULT_TEMPERATURE, "marker_id": tok.convert_tokens_to_ids(inference.DECISION_TOKEN),
            "pad_id": tok.pad_token_id, "symbols": inference.ANSWER_SYMBOLS, "symbol_ids": symbol_ids,
            "system_prompt": inference.SYSTEM_PROMPT, "text_config": config["text_config"], "dropped_keys": dropped}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("model_name", "temperature", "marker_id", "pad_id", "language_shard_sha256")}, indent=2))
    print("keys:", len(state), "dropped:", dropped, "embed:", tuple(state["embed_tokens.weight"].shape))


if __name__ == "__main__":
    main()
