"""Weight-only compression of a converted `DecisionRow` package (int8 per-channel or int4 per-block).

    uv run --no-project --python 3.12 --with torch==2.7.0 --with coremltools==9.0 --with "numpy<2.3" \
        python quantize.py --package build/L512_F8/DecisionRow_fp16.mlpackage --mode w8
"""
import argparse
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--mode", choices=["w8", "w4"], default="w8")
    ap.add_argument("--block-size", type=int, default=32)
    args = ap.parse_args()
    model = ct.models.MLModel(str(args.package), compute_units=ct.ComputeUnit.CPU_ONLY)
    if args.mode == "w8":
        op = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
    else:
        op = cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_block",
                                         block_size=args.block_size)
    start = time.time()
    compressed = cto.linear_quantize_weights(model, cto.OptimizationConfig(global_config=op))
    out = args.package.with_name(args.package.name.replace("_fp16", f"_{args.mode}"))
    compressed.save(str(out))
    print(f"saved {out} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
