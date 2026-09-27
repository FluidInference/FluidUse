"""Evaluate SauerkrautLM-Doom-MultiVec (PyTorch or Core ML) on defend_the_center, Space-identical setup."""
import argparse, json, os, statistics, time
import numpy as np, torch, vizdoom
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from doom_multivec.ascii.converter import AsciiConverter
from doom_multivec.model.classifier import DoomMultiVecClassifier

MODEL_ID = "VAGOsolutions/SauerkrautLM-Doom-MultiVec-1.3M"
DEPTH_BINS, NO_DEPTH, MAX_TOKENS, FRAME_SKIP, TIMEOUT = 16, 16, 1100, 4, 2100
NAMES = ["shoot", "move_forward", "turn_left", "turn_right"]
BUTTONS = {"shoot": [1,0,0,0], "move_forward": [0,1,0,0], "turn_left": [0,0,1,0], "turn_right": [0,0,0,1]}

def load_torch(model_dir):
    w = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    m = DoomMultiVecClassifier(model_dir, pool_mode="attention", num_actions=4)
    m.load_state_dict(w); m.eval(); return m

def make_game():
    g = vizdoom.DoomGame()
    g.load_config(os.path.join(vizdoom.scenarios_path, "defend_the_center.cfg"))
    g.set_window_visible(False)
    g.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    g.set_screen_format(vizdoom.ScreenFormat.RGB24)
    g.set_render_hud(True); g.set_depth_buffer_enabled(True)
    g.clear_available_buttons()
    for b in ("ATTACK", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"):
        g.add_available_button(getattr(vizdoom.Button, b))
    for v in ("HEALTH", "AMMO2", "KILLCOUNT"):
        g.add_available_game_variable(getattr(vizdoom.GameVariable, v))
    g.set_episode_timeout(TIMEOUT); g.set_mode(vizdoom.Mode.PLAYER); g.init(); return g

class Encoder:
    def __init__(self, model_dir):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.conv = AsciiConverter(width=40, height=25)
    def __call__(self, screen, depth):
        text, bins = self.conv.convert_with_depth(screen, depth, num_bins=DEPTH_BINS)
        enc = self.tok(text, return_tensors="np", max_length=MAX_TOKENS, padding="max_length", truncation=True)
        ids = enc["input_ids"].astype(np.int32); mask = enc["attention_mask"].astype(np.int32)
        d = np.full((1, MAX_TOKENS), NO_DEPTH, dtype=np.int32)
        n = min(len(bins), MAX_TOKENS - 2)
        d[0, 1:1+n] = np.asarray(bins[:n])
        return ids, mask, d

def composite(probs):
    ranked = np.argsort(probs)[::-1]
    best, second = NAMES[ranked[0]], NAMES[ranked[1]]
    buttons = list(BUTTONS[best])
    press = lambda n: [max(a, b) for a, b in zip(buttons, BUTTONS[n])]
    if best != "shoot" and probs[0] > probs[ranked[0]] * 0.75: buttons = press("shoot")
    kind = {"move_forward": "move", "turn_left": "turn", "turn_right": "turn"}
    if probs[ranked[1]] > 0.15 and second != "shoot" and kind.get(best) and kind.get(second) and kind[best] != kind[second]:
        buttons = press(second)
    return buttons

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="torch")  # torch | coreml
    ap.add_argument("--mlpackage", default=None)
    ap.add_argument("--units", default="CPU_AND_NE")
    ap.add_argument("--seeds", default="10000:10100")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model_dir = snapshot_download(MODEL_ID)
    enc = Encoder(model_dir)
    if a.backend == "torch":
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
        m = load_torch(model_dir)
        def logits(ids, mask, d):
            with torch.inference_mode():
                return m(torch.from_numpy(ids).long(), torch.from_numpy(mask).long(), depth_ids=torch.from_numpy(d).long())["logits"][0].numpy()
    else:
        import coremltools as ct
        ml = ct.models.MLModel(a.mlpackage, compute_units=getattr(ct.ComputeUnit, a.units))
        names = {i.name for i in ml.get_spec().description.input}
        def logits(ids, mask, d):
            if "attention_mask" not in names:  # L1026 variant: every frame is exactly 1026 tokens
                assert mask.sum() == 1026
                return ml.predict({"input_ids": ids[:, :1026], "depth_ids": d[:, :1026]})["logits"][0]
            return ml.predict({"input_ids": ids, "attention_mask": mask, "depth_ids": d})["logits"][0]
    lo, hi = map(int, a.seeds.split(":"))
    g = make_game(); kills_all, ms_all, tics_all, died_all = [], [], [], []
    t0 = time.time()
    for seed in range(lo, hi):
        g.set_seed(seed); g.new_episode()
        while not g.is_episode_finished():
            s = g.get_state()
            ids, mask, d = enc(s.screen_buffer, s.depth_buffer)
            t = time.perf_counter(); lg = np.asarray(logits(ids, mask, d), dtype=np.float64); ms_all.append((time.perf_counter()-t)*1000)
            p = np.exp(lg - lg.max()); p /= p.sum()
            g.make_action(composite(p), FRAME_SKIP)
        kills_all.append(int(g.get_game_variable(vizdoom.GameVariable.KILLCOUNT))); tics_all.append(g.get_episode_time()); died_all.append(g.is_player_dead())
    g.close()
    print(f"{a.backend}{'/'+a.units if a.backend=='coreml' else ''}: {len(kills_all)} eps, mean kills {statistics.mean(kills_all):.2f} "
          f"(sd {statistics.pstdev(kills_all):.2f}), model ms median {statistics.median(ms_all):.2f} p95 {np.percentile(ms_all,95):.2f}, wall {time.time()-t0:.0f}s, survived {statistics.mean(tics_all)/35:.1f}s avg, {sum(not d for d in died_all)} full episodes")
    if a.out: json.dump({"tics": tics_all, "died": died_all, "kills": kills_all, "ms_median": statistics.median(ms_all)}, open(a.out, "w"))

if __name__ == "__main__": main()
