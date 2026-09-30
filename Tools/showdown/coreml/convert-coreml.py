"""Convert `DecisionRow` to one fixed-length Core ML package.

Outputs (build/L<length>_F<fields>/):
  DecisionRow_fp16.mlpackage   hidden [1,L,D], cos/sin [L,R], field_onehot [F,L] -> logits [F,62]
  embeddings.f16               token embedding table [vocab, D] fp16, row-major (host gather)
  config.json                  shapes, marker/pad/symbol ids, temperature, checkpoint hash

    uv run --no-project --python 3.12 --with torch==2.7.0 --with coremltools==9.0 --with safetensors --with "numpy<2.3" \
        python convert-coreml.py --length 512 --max-fields 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from decision_export import load_decision_row, row_inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, default=Path("build/merged"))
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--max-fields", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--build", type=Path, default=Path("build"))
    args = ap.parse_args()
    L, F = args.length, args.max_fields
    row, cfg, meta, embed = load_decision_row(args.merged, L, F, args.chunk_size)
    out = args.build / f"L{L}_F{F}"
    out.mkdir(parents=True, exist_ok=True)
    emb_path = args.build / "embeddings.f16"
    if not emb_path.exists():
        embed.to(torch.float16).numpy().tofile(emb_path)
    ids = [meta["pad_id"]] * 12 + [meta["marker_id"]] * 0
    example = row_inputs(cfg, embed, ids, [5, 9], L, F, meta["pad_id"])
    start = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(row, example, check_trace=False)
    shapes = [(1, L, cfg.hidden_size), (L, cfg.rotary_dim), (L, cfg.rotary_dim), (F, L)]
    names = ["hidden", "cos", "sin", "field_onehot"]
    model = ct.convert(
        traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[ct.TensorType(name=n, shape=s, dtype=np.float32) for n, s in zip(names, shapes)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
    )
    model.short_description = "Intern-Decision-0.8B: Qwen3.5 backbone, <decision> marker readout, one prefill pass"
    model.author = "Shanghai AI Laboratory (Intern-Decision, Apache-2.0); Fluid Inference (Core ML conversion)"
    model.license = "Apache-2.0"
    model.user_defined_metadata.update({"checkpoint": meta["checkpoint"], "length": str(L), "max_fields": str(F),
                                        "temperature": str(meta["temperature"])})
    package = out / f"DecisionRow_{args.precision}.mlpackage"
    model.save(str(package))
    config = {"model_name": meta["model_name"], "checkpoint": meta["checkpoint"],
              "language_shard_sha256": meta["language_shard_sha256"], "length": L, "max_fields": F,
              "hidden_size": cfg.hidden_size, "rotary_dim": cfg.rotary_dim, "rope_theta": cfg.rope_theta,
              "vocab_size": int(embed.shape[0]), "pad_id": meta["pad_id"], "marker_id": meta["marker_id"],
              "symbols": meta["symbols"], "symbol_ids": meta["symbol_ids"], "temperature": meta["temperature"],
              "system_prompt": meta["system_prompt"], "precision": args.precision}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"saved {package} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
