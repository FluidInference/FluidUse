"""ANE variant: fixed 1026 tokens (no padding exists: 40x25 ASCII + 24 newlines + CLS/SEP), one-hot matmul embeddings."""
import numpy as np, torch, coremltools as ct, time
from convert import *
N = 1026

class Ane(Static):
    def __init__(self, clf):
        super().__init__(clf, n=N)
        e = self.enc
        self.register_buffer("tok_table", e.embeddings.tok_embeddings.hash_embeddings.weight.detach().clone())
        self.register_buffer("depth_table", e.depth_embedding.depth_emb.weight.detach().clone())
    def forward(self, ids, depth):
        e = self.enc
        oh = (ids.float().unsqueeze(-1) == torch.arange(self.tok_table.shape[0]).float()).float()
        od = (depth.float().unsqueeze(-1) == torch.arange(self.depth_table.shape[0]).float()).float()
        he = e.embeddings.tok_embeddings
        x = he.norm(he.projection(oh @ self.tok_table)) + od @ self.depth_table
        x = e.embeddings.norm(x)
        for i, layer in enumerate(e.layers):
            a = layer.attn
            qkv = a.Wqkv(layer.attn_norm(x)).view(1, -1, 3, self.nh, self.hd)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            cos, sin = (self.lcos, self.lsin) if self.local[i] else (self.gcos, self.gsin)
            q = q * cos + rot_half(q, self.hd // 2) * sin; k = k * cos + rot_half(k, self.hd // 2) * sin
            w = torch.matmul(q, k.transpose(2, 3)) * (self.hd ** -0.5)
            if self.local[i]: w = w + self.win
            o = torch.matmul(torch.softmax(w, -1), v).transpose(1, 2).reshape(1, -1, self.h)
            x = x + a.Wo(o)
            x = x + layer.mlp(layer.mlp_norm(x))
        x = e.final_norm(x)
        pooled = (x * torch.softmax(self.clf.attn_weight(x).squeeze(-1), 1).unsqueeze(-1)).sum(1)
        return self.clf.classifier(pooled)

clf = load_torch(snapshot_download(MODEL_ID)); m = Ane(clf).eval()
d = np.load("frames.npz"); ids, mask, dep = d["ids"], d["mask"], d["depth"]
assert (mask.sum(1) == N).all()
with torch.no_grad():
    ref = np.stack([clf(torch.from_numpy(ids[i:i+1]).long(), torch.from_numpy(mask[i:i+1]).long(), depth_ids=torch.from_numpy(dep[i:i+1]).long())["logits"][0].numpy() for i in range(len(ids))])
    mine = np.stack([m(torch.from_numpy(ids[i:i+1, :N]), torch.from_numpy(dep[i:i+1, :N]))[0].numpy() for i in range(len(ids))])
    print(f"ane-torch vs HF: {np.abs(mine-ref).max():.2e}")
    tr = torch.jit.trace(m, (torch.from_numpy(ids[:1, :N]), torch.from_numpy(dep[:1, :N])))
ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=(1, N), dtype=np.int32) for n in ("input_ids", "depth_ids")],
                outputs=[ct.TensorType(name="logits")], compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.macOS14, convert_to="mlprogram")
ml.save("SauerkrautDoom_L1026_fp16.mlpackage")
for u in ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"):
    mm = ct.models.MLModel("SauerkrautDoom_L1026_fp16.mlpackage", compute_units=getattr(ct.ComputeUnit, u))
    out = np.stack([mm.predict({"input_ids": ids[i:i+1, :N], "depth_ids": dep[i:i+1, :N]})["logits"][0] for i in range(len(ids))])
    f = {"input_ids": ids[:1, :N], "depth_ids": dep[:1, :N]}
    for _ in range(20): mm.predict(f)
    t = []
    for _ in range(200):
        s = time.perf_counter(); mm.predict(f); t.append((time.perf_counter() - s) * 1000)
    print(f"L1026 fp16 {u:12s} max|d| {np.abs(out-ref).max():.2e} agree {(out.argmax(1)==ref.argmax(1)).mean():.1%}  median {np.median(t):.2f} ms p95 {np.percentile(t,95):.2f}")
