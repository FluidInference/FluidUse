"""Distil teacher decision logs into Intern-Decision-0.8B with LoRA (PyTorch, Apple GPU).

Each logged decision is re-rendered with the student's own prompt compiler (the checkpoint's inference.py), with the
option order shuffled every time it is seen (the answer symbols move with the options, so the student cannot learn
a symbol prior). The loss is KL(teacher || student) between the teacher's logged probabilities and the student's
temperature-scaled restricted softmax at the position before the `<decision>` marker, the same readout the runtime
uses. The vision tower and embeddings are frozen; LoRA goes on every linear layer of the language model.

    uv run --no-project --python 3.12 --with torch==2.9.1 --with torchvision==0.24.1 --with transformers==5.14.1 \
        --with peft --with Pillow --with safetensors python train_student.py --checkpoint <0.8B snapshot> \
        --logs dataset/teacher-*.jsonl --out build/student-r1 --epochs 2
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch


def load_inference(checkpoint: Path):
    spec = importlib.util.spec_from_file_location("student_inference", checkpoint / "inference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_decisions(patterns: list[str]) -> list[dict]:
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for line in open(path):
                row = json.loads(line)
                criteria = row["request"]["questions"]["action"]["criteria"]
                probabilities = row["probabilities"]
                if len(criteria) < 2 or set(criteria) != set(probabilities):
                    continue
                rows.append({"battle": row["battle"], "state": row["request"]["state"],
                             "instructions": row["request"]["questions"]["action"]["instructions"],
                             "options": list(criteria.items()), "teacher": probabilities})
    return rows


class Renderer:
    def __init__(self, checkpoint: Path):
        from transformers import AutoTokenizer

        self.inference = load_inference(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        self.marker = self.tokenizer.convert_tokens_to_ids(self.inference.DECISION_TOKEN)
        self.symbol_ids = [self.tokenizer.encode(s, add_special_tokens=False)[0] for s in self.inference.ANSWER_SYMBOLS]

    def encode(self, row: dict, order: list[int]):
        options = [row["options"][i] for i in order]
        request = {"state": row["state"], "questions": {"action": {"type": "choice", "instructions": row["instructions"],
                                                                   "criteria": dict(options)}}}
        compiled = self.inference.compile_row(request)
        text = self.tokenizer.apply_chat_template(compiled.messages, tokenize=False, add_generation_prompt=False,
                                                  enable_thinking=False, add_vision_id=True)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        positions = [i - 1 for i, t in enumerate(ids) if t == self.marker]
        assert len(positions) == 1 and positions[0] >= 0
        teacher = torch.tensor([row["teacher"][label] for label, _ in options], dtype=torch.float32)
        return ids, positions[0], teacher / teacher.sum()


def split_by_battle(rows: list[dict], holdout: float, seed: int = 0):
    battles = sorted({r["battle"] for r in rows})
    random.Random(seed).shuffle(battles)
    held = set(battles[: max(1, int(len(battles) * holdout))])
    return [r for r in rows if r["battle"] not in held], [r for r in rows if r["battle"] in held]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--accumulate", type=int, default=8)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=200, help="optimizer steps")
    ap.add_argument("--eval-samples", type=int, default=300)
    ap.add_argument("--resume", type=Path, help="adapter directory to continue from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration

    renderer = Renderer(args.checkpoint)
    rows = load_decisions(args.logs)
    train, held = split_by_battle(rows, args.holdout, args.seed)
    print(f"decisions {len(rows)}: train {len(train)}, held-out {len(held)} ({len({r['battle'] for r in held})} battles)",
          flush=True)
    device = torch.device("mps")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(args.checkpoint), dtype=torch.bfloat16,
                                                            local_files_only=True, attn_implementation="sdpa")
    if args.resume:
        model = PeftModel.from_pretrained(model, str(args.resume), is_trainable=True)
    else:
        config = LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|"
                           r"in_proj_qkv|in_proj_z|out_proj)$")
        model = get_peft_model(model, config)
    model.to(device)
    model.print_trainable_parameters()
    temperature = renderer.inference.DEFAULT_TEMPERATURE
    symbol_ids = torch.tensor(renderer.symbol_ids, device=device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    total_steps = math.ceil(len(train) * args.epochs / args.accumulate)
    warmup = max(10, total_steps // 20)

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    def student_logprobs(row, order):
        ids, position, teacher = renderer.encode(row, order)
        if len(ids) > args.max_tokens:
            return None, None
        input_ids = torch.tensor([ids], device=device)
        logits = model(input_ids=input_ids, use_cache=False, logits_to_keep=torch.tensor([position], device=device)).logits[0, 0]
        restricted = logits[symbol_ids[: len(order)]].float() / temperature
        return torch.log_softmax(restricted, dim=-1), teacher.to(device)

    @torch.no_grad()
    def evaluate(sample):
        model.eval()
        agree, kl, n = 0, 0.0, 0
        for row in sample:
            order = list(range(len(row["options"])))
            logp, teacher = student_logprobs(row, order)
            if logp is None:
                continue
            kl += float((teacher * (torch.log(teacher.clamp_min(1e-9)) - logp)).sum())
            agree += int(logp.argmax().item() == teacher.argmax().item())
            n += 1
        model.train()
        return {"agreement": agree / max(1, n), "kl": kl / max(1, n), "n": n}

    eval_sample = held[: args.eval_samples]
    print("before:", json.dumps(evaluate(eval_sample)), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    history = []
    step, seen, running, start = 0, 0, 0.0, time.time()
    model.train()
    epoch_rows = []
    while step < total_steps:
        if not epoch_rows:
            epoch_rows = list(train)
            rng.shuffle(epoch_rows)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.accumulate):
            if not epoch_rows:
                break
            row = epoch_rows.pop()
            order = list(range(len(row["options"])))
            rng.shuffle(order)
            logp, teacher = student_logprobs(row, order)
            if logp is None:
                continue
            loss = (teacher * (torch.log(teacher.clamp_min(1e-9)) - logp)).sum() / args.accumulate
            loss.backward()
            running += loss.item() * args.accumulate
            seen += 1
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.step()
        step += 1
        if step % 20 == 0:
            elapsed = time.time() - start
            print(f"step {step}/{total_steps} seen {seen} loss {running / max(1, 20 * args.accumulate):.4f} "
                  f"lr {lr_at(step):.2e} {elapsed / 60:.1f} min, {seen / elapsed:.2f} samples/s", flush=True)
            running = 0.0
        if step % args.eval_every == 0 or step == total_steps:
            metrics = {"step": step, "seen": seen, **evaluate(eval_sample), "minutes": (time.time() - start) / 60}
            history.append(metrics)
            print("eval:", json.dumps(metrics), flush=True)
            model.save_pretrained(str(args.out / "adapter"))
            (args.out / "history.json").write_text(json.dumps(history, indent=2))
    # merged checkpoint in the Hub layout so the Core ML pipeline and the harness can load it unchanged
    merged_dir = args.out / "merged"
    merged = model.merge_and_unload()
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
                 "special_tokens_map.json", "chat_template.jinja", "preprocessor_config.json",
                 "video_preprocessor_config.json", "inference.py", "LICENSE", "LICENSE-QWEN"):
        source = args.checkpoint / name
        if source.exists():
            shutil.copy(source, merged_dir / name)
    (args.out / "train_args.json").write_text(json.dumps({**vars(args), "checkpoint": str(args.checkpoint),
                                                          "out": str(args.out), "resume": str(args.resume),
                                                          "train_decisions": len(train), "held_out": len(held)},
                                                         indent=2))
    print("saved", merged_dir, flush=True)


if __name__ == "__main__":
    main()
