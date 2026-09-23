"""Persistent JSON-lines adapter to the conversion authors' published Core ML runtimes.

This process only loads local Core ML packages; no upstream PyTorch checkpoint is loaded.
The caller supplies an environment containing the published runtime's dependencies.

Protocol: after loading, one `ready` line. Then, for each request line, exactly one reply line:
`ok <answer JSON>` or `error <message JSON string>`. Replies use a private copy of the original
standard output; file descriptor 1 is redirected to standard error so that native or library
prints cannot corrupt the protocol.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def strict_context(tokenizer, predict, length=128):
    """Reject requests the L128 package could only take by shortening the state (FLUIDUSE_STRICT_CONTEXT=1)."""
    if os.environ.get("FLUIDUSE_STRICT_CONTEXT") != "1":
        return predict
    from kev.api import SystemOneRequest, to_record
    from kev.model import encode

    def checked(request):
        record, _ = to_record(SystemOneRequest.model_validate(request))
        full = encode(tokenizer, record, option_isolation=False, max_state=8192, max_branch=16384)
        if len(full["ids"]) > length:
            raise ValueError(f"request needs {len(full['ids'])} tokens; exceeds L{length} without truncation")
        return predict(request)

    return checked


def runtime(model: str, root: Path, precision: str):
    if model == "kev-0-5b":
        sys.path.insert(0, str(root))
        module = importlib.import_module("runtime")
        package = root / f"kev_0_5b_{precision}_L128_options32.mlpackage"
        session = module.KevCoreML(root, package=package)
        return strict_context(session.tokenizer, session.predict)

    if model == "kev-0.6b":
        sys.path.insert(0, str(root / "source"))
        module = importlib.import_module("runtime")
        if precision not in module.PACKAGES:
            raise ValueError("Kev 0.6B precision must be fp16 or w8")
        package = root / module.PACKAGES[precision]
        training = json.loads((root / "config" / "training_config.json").read_text())
        if training["args"]["option_isolation"] not in (0, False):
            raise ValueError("Kev 0.6B requires option_isolation=False")
        tokenizer = module.AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
        shape = module.Shape()
        coreml = module.ct.models.MLModel(str(package), compute_units=module.COMPUTE_UNITS["all"])

        # Mirrors runtime.predict, which reloads the tokenizer and package on every call.
        def kev06(request):
            arrays, encoded, metadata, parsed = module.prepare_runtime_inputs(tokenizer, request, shape)
            probabilities = module.np.asarray(coreml.predict(arrays)["probabilities"], dtype=module.np.float64)
            count = len(metadata[0]["keys"])
            if probabilities.shape != (1, shape.max_options) or not module.np.isfinite(probabilities).all():
                raise ValueError("Invalid Kev 0.6B Core ML probabilities")
            selected = probabilities[0, :count].tolist()
            answers = module.to_answers([selected], metadata)
            # Upstream answers round to two decimals; the unrounded values match the Kev 0.5B runtime's extra fields.
            return {"model": parsed.model, "answers": answers,
                    "usage": {"input_tokens": len(encoded["ids"]),
                              "output_tokens": module.output_tokens(tokenizer, answers)},
                    "option_keys": metadata[0]["keys"], "probabilities": selected}

        return strict_context(tokenizer, kev06)

    if model in {"decision-1.0-kai", "decision-1.0-lex"}:
        sys.path.insert(0, str(root / "conversion"))
        module = importlib.import_module("run_coreml")
        compressed = ()
        if precision == "w8":
            compressed = ("choice", "noul", "score") if model == "decision-1.0-kai" else ("noul", "score")
        session = module.CoreMLSystemOne(root, compressed)
        return session.evaluate

    if model == "lfm2-5-350m-rlcd":
        sys.path.insert(0, str(root))
        module = importlib.import_module("runtime")
        package = root / "lfm350_rlcd_fp16_L256_B8_V16.mlpackage"
        session = module.RLCDCoreML(package, root)
        return lambda request: session.constrained(request["context"], request["schema"])

    if model == "jeff":
        sys.path.insert(0, str(root))
        module = importlib.import_module("runtime")
        package_name = "JeffDecision-L128-W8.mlpackage" if precision == "w8" else "JeffDecision-L128-FP16.mlpackage"
        session = module.JeffCoreML(root, root / package_name)

        def classify(request):
            labels = request["labels"]
            probabilities = session.score(
                request["text"], labels,
                request.get("name", ""), request.get("description", ""),
            )
            selected = max(range(len(labels)), key=lambda index: probabilities[index])
            return {"labels": labels, "probabilities": probabilities,
                    "selected_index": selected, "selected_label": labels[selected]}

        return classify

    if model == "nanojev":
        import coremltools as ct
        import numpy as np
        from transformers import AutoTokenizer

        sys.path.insert(0, str(root))
        assets = importlib.import_module("assets")
        preprocessing = importlib.import_module("preprocessing")
        source = assets.snapshot(with_weights=False)
        tokenizer = AutoTokenizer.from_pretrained(source / "tokenizer", local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoder = ct.models.MLModel(str(root / "build/nanojev_encoder_fp16_L128_K4.mlpackage"),
                                     compute_units=ct.ComputeUnit.CPU_AND_NE)
        head = ct.models.MLModel(str(root / "build/nanojev_heads_fp16_K4.mlpackage"),
                                  compute_units=ct.ComputeUnit.CPU_AND_NE)

        def nanojev(request):
            inputs, candidate_mask, example = preprocessing.prepare_request(
                source, tokenizer, request, 128, 4)
            kind = example["type"]
            embeddings = encoder.predict(inputs)["embeddings"]
            output = head.predict({
                "embeddings": np.asarray(embeddings, dtype=np.float32),
                "candidate_mask": candidate_mask,
                "use_set_head": np.array([[kind == "choice"]], dtype=np.float32),
                "is_boolean": np.array([[kind == "boolean"]], dtype=np.float32),
            })
            candidates = example["candidate_ids"]
            probabilities = np.asarray(output["probabilities"])[0, :len(candidates)].tolist()
            chosen = int(np.argmax(probabilities))
            return {"type": kind, "candidate_ids": candidates,
                    "probabilities": probabilities, "selected_id": candidates[chosen]}

        return nanojev

    raise ValueError(f"Unsupported published Core ML model: {model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--precision", default="fp16")
    args = parser.parse_args()

    replies = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    sys.stdin.reconfigure(encoding="utf-8")

    predict = runtime(args.model, args.root.resolve(strict=True), args.precision)
    replies.write("ready\n")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            answer = json.dumps(predict(json.loads(line)), ensure_ascii=False, allow_nan=False)
        except Exception as exc:
            replies.write("error " + json.dumps(f"{type(exc).__name__}: {exc}") + "\n")
        else:
            replies.write("ok " + answer + "\n")


if __name__ == "__main__":
    main()
