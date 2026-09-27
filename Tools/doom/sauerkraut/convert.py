"""Core ML export of SauerkrautLM-Doom-MultiVec-1.3M: static-mask re-implementation of the traced forward."""
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, coremltools as ct
from huggingface_hub import snapshot_download
sys.path.insert(0, os.path.dirname(__file__))
from sauer_eval import load_torch, make_game, Encoder, MODEL_ID, MAX_TOKENS, FRAME_SKIP

NEG = -1e4

def rope(theta, head_dim, n):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    f = torch.outer(torch.arange(n, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos()[None, None], e.sin()[None, None]

def rot_half(x, h):
    return torch.cat((-x[..., h:], x[..., :h]), -1)

class Static(nn.Module):
    def __init__(self, clf, n=MAX_TOKENS):
        super().__init__()
        enc = clf.encoder; c = enc.config
        self.enc, self.clf = enc, clf
        self.h, self.nh = c.hidden_size, c.num_attention_heads
        self.hd = self.h // self.nh
        self.local = [i % c.global_attn_every_n_layers != 0 for i in range(c.num_hidden_layers)]
        gc, gs = rope(c.global_rope_theta, self.hd, n); lc, ls = rope(c.local_rope_theta, self.hd, n)
        self.register_buffer("gcos", gc); self.register_buffer("gsin", gs)
        self.register_buffer("lcos", lc); self.register_buffer("lsin", ls)
        idx = torch.arange(n)
        win = ((idx[None] - idx[:, None]).abs() > c.local_attention // 2).float() * NEG
        self.register_buffer("win", win[None, None])

    def forward(self, ids, mask, depth):
        e = self.enc
        x = e.embeddings.tok_embeddings(ids.long()) + e.depth_embedding(depth.long())
        x = e.embeddings.norm(x)
        pad = ((1.0 - mask.float()) * NEG)[:, None, None, :]
        local_bias = pad + self.win
        for i, layer in enumerate(e.layers):
            a = layer.attn
            qkv = a.Wqkv(layer.attn_norm(x)).view(1, -1, 3, self.nh, self.hd)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            cos, sin = (self.lcos, self.lsin) if self.local[i] else (self.gcos, self.gsin)
            q = q * cos + rot_half(q, self.hd // 2) * sin; k = k * cos + rot_half(k, self.hd // 2) * sin
            w = torch.matmul(q, k.transpose(2, 3)) * (self.hd ** -0.5) + (local_bias if self.local[i] else pad)
            o = torch.matmul(torch.softmax(w, -1), v).transpose(1, 2).reshape(1, -1, self.h)
            x = x + a.Wo(o)
            x = x + layer.mlp(layer.mlp_norm(x))
        x = e.final_norm(x)
        s = self.clf.attn_weight(x).squeeze(-1) + (1.0 - mask.float()) * NEG
        pooled = (x * torch.softmax(s, 1).unsqueeze(-1)).sum(1)
        return self.clf.classifier(pooled)

if __name__ == "__main__":
    model_dir = snapshot_download(MODEL_ID)
    clf = load_torch(model_dir)
    static = Static(clf).eval()
    g = make_game(); enc = Encoder(model_dir); frames = []
    rng = np.random.default_rng(0)
    for seed in (1, 2):
        g.set_seed(seed); g.new_episode()
        while not g.is_episode_finished():
            s = g.get_state(); frames.append(enc(s.screen_buffer, s.depth_buffer))
            g.make_action([int(x) for x in rng.integers(0, 2, 4)], FRAME_SKIP)
    g.close()
    frames = frames[::3][:80]
    T = lambda f: [torch.from_numpy(x) for x in f]
    with torch.no_grad():
        ref = torch.stack([clf(*[t.long() for t in T(f)[:2]], depth_ids=T(f)[2].long())["logits"][0] for f in frames]).numpy()
        mine = torch.stack([static(*T(f))[0] for f in frames]).numpy()
    print(f"static vs HF torch: max|dlogit| {np.abs(mine-ref).max():.2e}, lengths {sorted(set(int(f[1].sum()) for f in frames))}")
    with torch.no_grad():
        traced = torch.jit.trace(static, tuple(T(frames[0])))
    inputs = [ct.TensorType(name=n, shape=(1, MAX_TOKENS), dtype=np.int32) for n in ("input_ids", "attention_mask", "depth_ids")]
    for prec, name in ((ct.precision.FLOAT32, "fp32"), (ct.precision.FLOAT16, "fp16")):
        ml = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name="logits")], compute_precision=prec,
                        minimum_deployment_target=ct.target.macOS14, convert_to="mlprogram")
        path = f"SauerkrautDoom_{name}.mlpackage"; ml.save(path)
        for units in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"):
            mm = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, units))
            out = np.stack([mm.predict({"input_ids": f[0], "attention_mask": f[1], "depth_ids": f[2]})["logits"][0] for f in frames])
            print(f"{name} {units}: max|dlogit| {np.abs(out-ref).max():.2e}, argmax agree {(out.argmax(1)==ref.argmax(1)).mean():.1%} / {len(frames)}")
