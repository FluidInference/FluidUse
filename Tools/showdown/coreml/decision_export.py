"""Intern-Decision (Qwen3.5 backbone, `<decision>` marker readout) as one fixed-length, prefill-only request.

Intern-Decision renders the state and every typed question as one chat prompt whose assistant turn is a JSON skeleton
with one `<decision>` token per field. The logits at the position immediately before each marker, restricted to the
single-token answer symbols (A-Z, a-z, 0-9), are that field's answer. Nothing is generated. `DecisionRow` runs the
Qwen3.5 decoder from `qwen35_export.py`, selects the pre-marker hidden states with one one-hot map per field, applies
the final norm and multiplies by the 62 tied-embedding rows of the answer symbols.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from qwen35_export import DecoderChunk, RMSNorm, TextConfig, rope_cos_sin


class DecisionRow(nn.Module):
    def __init__(self, cfg: TextConfig, seq_len: int, max_fields: int, n_symbols: int = 62, chunk_size: int = 64):
        super().__init__()
        self.decoder = DecoderChunk(cfg, 0, cfg.num_layers, seq_len, chunk_size=chunk_size, with_head=False)
        self.norm = RMSNorm(cfg.hidden_size, cfg.eps)
        self.symbol_head = nn.Linear(cfg.hidden_size, n_symbols, bias=False)
        self.max_fields = max_fields

    def forward(self, hidden, cos, sin, field_onehot):
        """hidden [1, L, D] (host embedding gather), cos/sin [L, R], field_onehot [F, L] (row i selects the token
        before field i's marker; unused rows are zero) -> symbol logits [F, 62]."""
        states = self.decoder(hidden, cos, sin)[0]  # [L, D]
        selected = self.norm(torch.matmul(field_onehot, states))  # [F, D]
        return self.symbol_head(selected)


def load_decision_row(merged: Path, seq_len: int, max_fields: int, chunk_size: int = 64):
    meta = json.loads((merged / "meta.json").read_text())
    cfg = TextConfig(meta["text_config"])
    row = DecisionRow(cfg, seq_len, max_fields, len(meta["symbol_ids"]), chunk_size=chunk_size)
    state = load_file(str(merged / "text.safetensors"))
    row.decoder.load_merged(state)
    row.norm.load_state_dict({"weight": state["norm.weight"]})
    embed = state["embed_tokens.weight"]
    row.symbol_head.load_state_dict({"weight": embed[torch.tensor(meta["symbol_ids"])].clone()})
    return row.eval(), cfg, meta, embed


def row_inputs(cfg: TextConfig, embed: torch.Tensor, ids, positions, seq_len: int, max_fields: int, pad_id: int):
    """Right-pad one request to `seq_len` and build the export inputs. Padding follows every real token, and the model
    is causal with a forward-scan recurrence, so it cannot change any real token's state."""
    n = len(ids)
    if n > seq_len or len(positions) > max_fields:
        raise ValueError(f"request needs {n} tokens / {len(positions)} fields; bucket holds {seq_len} / {max_fields}")
    token_ids = torch.full((seq_len,), pad_id, dtype=torch.long)
    token_ids[:n] = torch.tensor(ids)
    pos = torch.arange(seq_len, dtype=torch.long)
    cos, sin = rope_cos_sin(cfg, pos.unsqueeze(0).expand(3, -1))
    field_onehot = torch.zeros(max_fields, seq_len)
    for i, index in enumerate(positions):
        field_onehot[i, index] = 1
    return embed[token_ids].unsqueeze(0), cos, sin, field_onehot


def load_inference_module(checkpoint: Path):
    """Import the checkpoint's own `inference.py` (prompt compiler, symbols, temperature scaling)."""
    spec = importlib.util.spec_from_file_location("intern_decision_inference", checkpoint / "inference.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Compiler:
    """Request dict -> token ids, pre-marker positions, per-field (labels, n_options) using the checkpoint's compiler
    and chat template, exactly as `inference.HFBackend.encode` does for text-only requests."""

    def __init__(self, checkpoint: Path):
        from transformers import AutoTokenizer

        self.inference = load_inference_module(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
        self.marker_id = self.tokenizer.convert_tokens_to_ids(self.inference.DECISION_TOKEN)

    def encode(self, request: dict):
        row = self.inference.validate_request(request)
        if row.get("images"):
            raise ValueError("text-only export")
        compiled = self.inference.compile_row(row)
        text = self.tokenizer.apply_chat_template(compiled.messages, tokenize=False, add_generation_prompt=False,
                                                  enable_thinking=False, add_vision_id=True)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        positions = [i - 1 for i, t in enumerate(ids) if t == self.marker_id]
        if len(positions) != len(compiled.fields) or min(positions) < 0:
            raise ValueError("decision marker count or position mismatch")
        fields = []
        for field in compiled.fields:
            options = self.inference._options(row["questions"][field])
            fields.append((field, [value for value, _ in options]))
        return ids, positions, fields, row


def field_probabilities(logits: torch.Tensor, fields, temperature: float):
    """Restricted softmax per field over its first n option symbols, then temperature scaling as
    `inference.scale_probabilities` does (softmax(log p / T) == softmax(logits / T))."""
    out = []
    for i, (_, labels) in enumerate(fields):
        raw = torch.softmax(logits[i, : len(labels)].float(), dim=-1)
        scaled = torch.softmax(logits[i, : len(labels)].float() / temperature, dim=-1)
        out.append((raw, scaled))
    return out
