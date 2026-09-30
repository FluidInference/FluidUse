"""Trace-friendly, prefill-only Qwen3.5 text decoder (copied from models/computer-use/cua-s1-4b/coreml, mobius PR #104).

Cua-S1-4B reads one forward pass: the logits of the answer letters A..Z at the
last prompt position. Nothing is generated, so this graph has no KV cache, no
recurrent-state I/O and no full vocabulary head:

- inputs are `inputs_embeds` (host-side embedding gather, so a multimodal
  caller can splice vision features in) plus host-computed M-RoPE `cos`/`sin`;
- prompts are right-padded to a fixed bucket; the model is causal and the gated
  delta rule is a forward scan, so padding after the last real token cannot
  change anything before it;
- the last chunk gathers the hidden state at `last_index` (one-hot matmul),
  applies the final norm and multiplies by the 26 tied-embedding rows of the
  letter tokens.

The gated delta rule uses the chunked (WY/UT) form. The unit-lower-triangular
solve is a recursive 2x2 block inverse (log2(chunk) batched matmul levels). The
nilpotent power series (I + N)(I + N^2)... is exact algebraically but its terms
grow combinatorially before cancelling and overflow on real Qwen3.5 layers.
Within-chunk decay differences are computed as masked sums of the per-step
decays instead of differences of cumulative sums, which keeps them accurate in
fp16 when the cumulative decay is large.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class TextConfig:
    def __init__(self, cfg: dict):
        self.hidden_size = cfg["hidden_size"]
        self.intermediate_size = cfg["intermediate_size"]
        self.num_layers = cfg["num_hidden_layers"]
        self.layer_types = cfg["layer_types"]
        self.eps = cfg["rms_norm_eps"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.lin_k_heads = cfg["linear_num_key_heads"]
        self.lin_v_heads = cfg["linear_num_value_heads"]
        self.lin_k_dim = cfg["linear_key_head_dim"]
        self.lin_v_dim = cfg["linear_value_head_dim"]
        self.conv_kernel = cfg["linear_conv_kernel_dim"]
        rope = cfg["rope_parameters"]
        self.rope_theta = rope["rope_theta"]
        self.rotary_dim = int(self.head_dim * rope.get("partial_rotary_factor", 1.0))
        self.mrope_section = rope.get("mrope_section", [11, 11, 10])


def rope_cos_sin(cfg: TextConfig, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved M-RoPE tables. `position_ids` is [3, L] (t, h, w); returns [L, rotary_dim] fp32 each.

    Mirrors `Qwen3_5TextRotaryEmbedding` (computed in float64 then cast, host side)."""
    dim = cfg.rotary_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    freqs = position_ids.to(torch.float64)[:, :, None] * inv_freq[None, None, :]  # [3, L, dim/2]
    thw = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        length = cfg.mrope_section[axis] * 3
        thw[:, offset:length:3] = freqs[axis][:, offset:length:3]
    emb = torch.cat([thw, thw], dim=-1)
    return emb.cos().float(), emb.sin().float()


class RMSNorm(nn.Module):
    """Qwen3.5 zero-centred RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + w)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * (1.0 + self.weight)


class MLP(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class FullAttention(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        h, kv, d = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, h * d * 2, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv * d, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv * d, bias=False)
        self.o_proj = nn.Linear(h * d, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(d, cfg.eps)
        self.k_norm = RMSNorm(d, cfg.eps)

    def forward(self, x, cos, sin, mask):
        cfg = self.cfg
        L = x.shape[1]
        h, kv, d, r = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim
        qg = self.q_proj(x).view(1, L, h, 2 * d)
        q, gate = qg[..., :d], qg[..., d:]
        gate = gate.reshape(1, L, h * d)
        q = self.q_norm(q).transpose(1, 2)  # [1, h, L, d]
        k = self.k_norm(self.k_proj(x).view(1, L, kv, d)).transpose(1, 2)
        v = self.v_proj(x).view(1, L, kv, d).transpose(1, 2)

        c, s = cos[None, None], sin[None, None]  # [1, 1, L, r]
        q = torch.cat([q[..., :r] * c + rotate_half(q[..., :r]) * s, q[..., r:]], dim=-1)
        k = torch.cat([k[..., :r] * c + rotate_half(k[..., :r]) * s, k[..., r:]], dim=-1)

        rep = h // kv
        k = k[:, :, None].expand(1, kv, rep, L, d).reshape(1, h, L, d)
        v = v[:, :, None].expand(1, kv, rep, L, d).reshape(1, h, L, d)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (d**-0.5) + mask
        out = torch.matmul(torch.softmax(scores, dim=-1), v)  # [1, h, L, d]
        out = out.transpose(1, 2).reshape(1, L, h * d)
        return self.o_proj(out * torch.sigmoid(gate))


class GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, gate):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * F.silu(gate)


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: TextConfig, seq_len: int, chunk_size: int):
        super().__init__()
        self.cfg = cfg
        self.chunk = chunk_size
        assert seq_len % chunk_size == 0, "bucket length must be a multiple of the delta-rule chunk size"
        self.num_chunks = seq_len // chunk_size
        kd = cfg.lin_k_heads * cfg.lin_k_dim
        vd = cfg.lin_v_heads * cfg.lin_v_dim
        self.key_dim, self.value_dim = kd, vd
        self.conv_dim = 2 * kd + vd
        self.in_proj_qkv = nn.Linear(cfg.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(cfg.hidden_size, vd, bias=False)
        self.in_proj_b = nn.Linear(cfg.hidden_size, cfg.lin_v_heads, bias=False)
        self.in_proj_a = nn.Linear(cfg.hidden_size, cfg.lin_v_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, cfg.conv_kernel, groups=self.conv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(cfg.lin_v_heads))
        self.A_log = nn.Parameter(torch.zeros(cfg.lin_v_heads))
        self.norm = GatedRMSNorm(cfg.lin_v_dim, cfg.eps)
        self.out_proj = nn.Linear(vd, cfg.hidden_size, bias=False)

        C = chunk_size
        idx = torch.arange(C)
        # incl[i, j, k] = 1 if j < k <= i : sum_k g_k = cum_i - cum_j (lower triangle incl. diagonal is 0 on i == j)
        incl = ((idx[None, None, :] > idx[None, :, None]) & (idx[None, None, :] <= idx[:, None, None])).float()
        self.register_buffer("pair_sum_t", incl.reshape(C * C, C), persistent=False)  # [C*C, C]
        self.register_buffer("prefix_sum_t", (idx[None, :] <= idx[:, None]).float(), persistent=False)  # [C, C]
        self.register_buffer("suffix_sum_t", (idx[None, :] > idx[:, None]).float(), persistent=False)
        self.register_buffer("lower_incl", (idx[None, :] <= idx[:, None]).float(), persistent=False)
        self.register_buffer("strict_lower", (idx[None, :] < idx[:, None]).float(), persistent=False)
        self.register_buffer("eye", torch.eye(C), persistent=False)
        assert C & (C - 1) == 0, "delta-rule chunk size must be a power of two"
        for n in (C // (2 * s) for s in (2**i for i in range(int(math.log2(C))))):
            self.register_buffer(f"block_eye_{n}", torch.eye(n), persistent=False)

    def unit_lower_inverse(self, t: torch.Tensor) -> torch.Tensor:
        """Inverse of unit-lower-triangular [..., C, C] by recursive 2x2 blocking:
        [[A, 0], [X, B]]^-1 = [[A^-1, 0], [-B^-1 X A^-1, B^-1]].

        Stable like forward substitution (it only forms the true inverse's sub-blocks),
        but log2(C) batched levels instead of C sequential row updates."""
        C = self.chunk
        lead = t.shape[:-2]
        t = t.reshape(-1, C, C)  # Core ML tensors are rank <= 5
        b = t.shape[0]
        inv = torch.ones(b, C, 1, 1, dtype=t.dtype, device=t.device)  # 1x1 diagonal blocks of a unit triangle
        s = 1
        while s < C:
            n2 = C // (2 * s)
            # diagonal 2s x 2s blocks of t, then their lower-left s x s part
            blocks = (t.reshape(b, n2, 2 * s, n2, 2 * s) * getattr(self, f"block_eye_{n2}")[:, None, :, None]).sum(-2)
            x = blocks[..., s:, :s]  # [b, n2, s, s]
            pair = inv.reshape(b, n2, 2, s, s)
            a_inv, b_inv = pair[..., 0, :, :], pair[..., 1, :, :]
            lower = -torch.matmul(b_inv, torch.matmul(x, a_inv))
            top = torch.cat([a_inv, torch.zeros_like(a_inv)], dim=-1)
            bottom = torch.cat([lower, b_inv], dim=-1)
            inv = torch.cat([top, bottom], dim=-2)  # [b, n2, 2s, 2s]
            s *= 2
        return inv.reshape(*lead, C, C)

    def forward(self, x):
        cfg = self.cfg
        L = x.shape[1]
        C, N = self.chunk, self.num_chunks
        H, Dk, Dv = cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim
        rep = cfg.lin_v_heads // cfg.lin_k_heads

        qkv = self.in_proj_qkv(x).transpose(1, 2)  # [1, conv_dim, L]
        qkv = F.silu(self.conv1d(F.pad(qkv, (cfg.conv_kernel - 1, 0)))).transpose(1, 2)
        q = qkv[..., : self.key_dim].reshape(1, L, cfg.lin_k_heads, Dk)
        k = qkv[..., self.key_dim : 2 * self.key_dim].reshape(1, L, cfg.lin_k_heads, Dk)
        v = qkv[..., 2 * self.key_dim :].reshape(1, L, H, Dv)
        z = self.in_proj_z(x).reshape(1, L, H, Dv)
        beta = torch.sigmoid(self.in_proj_b(x))  # [1, L, H]
        g = -torch.exp(self.A_log) * F.softplus(self.in_proj_a(x) + self.dt_bias)  # [1, L, H], <= 0

        # l2norm (FLA convention) then repeat k heads up to v heads
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q[:, :, :, None].expand(1, L, cfg.lin_k_heads, rep, Dk).reshape(1, L, H, Dk)
        k = k[:, :, :, None].expand(1, L, cfg.lin_k_heads, rep, Dk).reshape(1, L, H, Dk)
        q = q * (Dk**-0.5)

        # -> [H, N, C, D]
        q = q[0].permute(1, 0, 2).reshape(H, N, C, Dk)
        k = k[0].permute(1, 0, 2).reshape(H, N, C, Dk)
        v = v[0].permute(1, 0, 2).reshape(H, N, C, Dv)
        beta = beta[0].T.reshape(H, N, C, 1)
        g = g[0].T.reshape(H, N, C)

        # constant-first matmuls: `x @ const` would lower to Core ML `linear` and be picked up by weight compression
        g_col = g.unsqueeze(-1)  # [H, N, C, 1]
        cum = torch.matmul(self.prefix_sum_t, g_col).squeeze(-1)  # cum_i = sum_{k<=i} g_k
        to_end = torch.matmul(self.suffix_sum_t, g_col).squeeze(-1)  # sum_{k>i} g_k = cum_last - cum_i
        pair = torch.matmul(self.pair_sum_t, g_col).reshape(H, N, C, C)  # cum_i - cum_j (0 above diagonal)
        pair_decay = torch.exp(pair) * self.lower_incl

        k_beta = k * beta
        v_beta = v * beta
        kkt = torch.matmul(k_beta, k.transpose(-1, -2)) * pair_decay
        attn_intra = torch.matmul(q, k.transpose(-1, -2)) * pair_decay

        inv = self.unit_lower_inverse(self.eye + kkt * self.strict_lower)
        new_v = torch.matmul(inv, v_beta)  # [H, N, C, Dv]
        k_cumdecay = torch.matmul(inv, k_beta * torch.exp(cum)[..., None])  # [H, N, C, Dk]

        q_dec = q * torch.exp(cum)[..., None]
        k_dec = k * torch.exp(to_end)[..., None]
        chunk_decay = torch.exp(cum[..., -1:])[..., None]  # [H, N, 1, 1]

        state = torch.zeros(H, Dk, Dv, dtype=x.dtype, device=x.device)
        outs = []
        for i in range(N):
            v_new = new_v[:, i] - torch.matmul(k_cumdecay[:, i], state)
            outs.append(torch.matmul(q_dec[:, i], state) + torch.matmul(attn_intra[:, i], v_new))
            if i + 1 < N:
                state = state * chunk_decay[:, i] + torch.matmul(k_dec[:, i].transpose(-1, -2), v_new)
        core = torch.stack(outs, dim=1).reshape(H, L, Dv).permute(1, 0, 2)  # [L, H, Dv]

        core = self.norm(core, z[0])
        return self.out_proj(core.reshape(1, L, H * Dv))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: TextConfig, layer_idx: int, seq_len: int, chunk_size: int):
        super().__init__()
        self.is_linear = cfg.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(cfg, seq_len, chunk_size)
        else:
            self.self_attn = FullAttention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.eps)

    def forward(self, x, cos, sin, mask):
        h = self.input_layernorm(x)
        h = self.linear_attn(h) if self.is_linear else self.self_attn(h, cos, sin, mask)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x))


class DecoderChunk(nn.Module):
    """Layers [start, end). The last chunk (with_head) returns letter logits [1, n_letters]."""

    def __init__(
        self,
        cfg: TextConfig,
        start: int,
        end: int,
        seq_len: int,
        chunk_size: int = 64,
        with_head: bool = False,
        n_letters: int = 26,
    ):
        super().__init__()
        self.start, self.end, self.with_head = start, end, with_head
        self.layers = nn.ModuleList([DecoderLayer(cfg, i, seq_len, chunk_size) for i in range(start, end)])
        mask = torch.full((seq_len, seq_len), -1e4).triu(1)
        self.register_buffer("mask", mask[None, None], persistent=False)
        if with_head:
            self.norm = RMSNorm(cfg.hidden_size, cfg.eps)
            self.letter_head = nn.Linear(cfg.hidden_size, n_letters, bias=False)

    def forward(self, hidden, cos, sin, last_onehot=None):
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, self.mask)
        if not self.with_head:
            return hidden
        last = torch.matmul(last_onehot, hidden[0])  # [1, hidden]
        return self.letter_head(self.norm(last))

    def load_merged(self, state: dict[str, torch.Tensor], letter_rows: torch.Tensor | None = None):
        """`state` uses Qwen3_5TextModel names relative to the language model (`layers.N...`, `norm.weight`)."""
        own = {}
        for local, global_idx in enumerate(range(self.start, self.end)):
            prefix = f"layers.{global_idx}."
            for key, value in state.items():
                if key.startswith(prefix):
                    own[f"layers.{local}.{key[len(prefix):]}"] = value
        if self.with_head:
            own["norm.weight"] = state["norm.weight"]
            own["letter_head.weight"] = letter_rows
        missing, unexpected = self.load_state_dict({k: v.float() for k, v in own.items()}, strict=False)
        missing = [m for m in missing if not m.endswith(("mask",))]
        if missing or unexpected:
            raise RuntimeError(f"weight mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
