"""SauerkrautLM-Doom-MultiVec 1.3M on Core ML, playing ViZDoom defend_the_center.

Split-screen demo: the game on the left; on the right the 40x25 depth grid the model reads, its four
action probabilities, and live ms per decision. No PyTorch at runtime.

The model's input is 1026 tokens: a character per cell plus a learned embedding of that cell's depth
(16 bins). Upstream's ASCII path overflows uint8 and emits "@" for every cell, in training and here,
so the depth bins carry all the information; the panel draws them.

    .venv/bin/python Tools/doom/sauerkraut/play.py                        # window, real-time
    .venv/bin/python Tools/doom/sauerkraut/play.py --record doom.mp4 --episodes 3
    .venv/bin/python Tools/doom/sauerkraut/play.py --check 10              # headless, no rendering

Window mode also opens two Terminal windows: `sudo asitop` (GPU/ANE/CPU power) and `tail -f` of the
decision log (DOOM_DEMO_LOG, default /tmp/doom-demo.log). Pass --no-terminals to skip them.

The window renders every game tic (35 fps). ViZDoom's rendering consumes game randomness, so a
seed does not replay the headless benchmark path exactly (`--check` does); the play is equally
strong (seeds 10000-10029: 19.70 kills / 50.5 s rendered vs 20.37 / 50.7 s headless).

Keys: space pause, n next episode, q quit.
"""

import argparse
import json
import os
import statistics
import subprocess
import time

import numpy as np
import vizdoom

MODEL_ID = "VAGOsolutions/SauerkrautLM-Doom-MultiVec-1.3M"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "SauerkrautDoom_L1026_fp16.mlpackage")

ROWS, COLS, DEPTH_BINS = 25, 40, 16
NO_DEPTH = DEPTH_BINS
SEQ = 1 + ROWS * COLS + (ROWS - 1) + 1  # [CLS] + chars + newlines + [SEP]
CHARS = " .:-=+*#%@"
FRAME_SKIP, TIMEOUT = 4, 2100
NAMES = ["shoot", "move_forward", "turn_left", "turn_right"]
BUTTONS = {"shoot": [1, 0, 0, 0], "move_forward": [0, 1, 0, 0], "turn_left": [0, 0, 1, 0], "turn_right": [0, 0, 0, 1]}
LOG_PATH = os.environ.get("DOOM_DEMO_LOG", "/tmp/doom-demo.log")
PYTORCH_CPU_MS = 57.7  # one-thread PyTorch on the same M5 Pro, measured on the same frame


# ---- observation: the upstream AsciiConverter.convert_with_depth + character tokenizer ----
def downscale(img, h, w):
    bh, bw = img.shape[0] // h, img.shape[1] // w
    return img[: bh * h, : bw * w].reshape(h, bh, w, bw).mean(axis=(1, 3))


def observe(depth, vocab):
    coarse = downscale(depth.astype(np.uint8), ROWS, COLS).astype(np.uint8)
    peak = coarse.max()
    bright = 255 - (coarse * 255 / peak).astype(np.uint8) if peak > 0 else np.zeros_like(coarse)
    levels = np.minimum((bright.astype(np.int64) * len(CHARS)) // 256, len(CHARS) - 1)
    rows = ["".join(CHARS[v] for v in row) for row in levels]  # upstream uint8 overflow: always "@"

    d = downscale(depth.astype(np.float32), ROWS, COLS).astype(np.uint8).astype(np.float32)  # upstream truncates
    lo, hi = d.min(), d.max()
    norm = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    bins = np.clip((norm * DEPTH_BINS).astype(int), 0, DEPTH_BINS - 1)

    ids, depth_ids = [vocab["[CLS]"]], [NO_DEPTH]
    for y in range(ROWS):
        for x in range(COLS):
            ids.append(vocab[rows[y][x]])
            depth_ids.append(int(bins[y, x]))
        if y < ROWS - 1:
            ids.append(vocab["\n"])
            depth_ids.append(NO_DEPTH)
    ids.append(vocab["[SEP]"])
    depth_ids.append(NO_DEPTH)
    return bins, np.array([ids], dtype=np.int32), np.array([depth_ids], dtype=np.int32)


def choose(probs):
    """Upstream composite rule: add a shot when nearly as likely, add a turn/move runner-up."""
    ranked = np.argsort(probs)[::-1]
    best, second = NAMES[ranked[0]], NAMES[ranked[1]]
    buttons, label = list(BUTTONS[best]), [best]

    def press(name):
        return [max(a, b) for a, b in zip(buttons, BUTTONS[name])]

    if best != "shoot" and probs[0] > probs[ranked[0]] * 0.75:
        buttons = press("shoot")
        label.append("shoot")
    kind = {"move_forward": "move", "turn_left": "turn", "turn_right": "turn"}
    if probs[ranked[1]] > 0.15 and second != "shoot" and kind.get(best) and kind.get(second):
        if kind[best] != kind[second]:
            buttons = press(second)
            label.append(second)
    return buttons, " + ".join(label)


class DecisionLog:
    """ANSI-colored decision lines for a `tail -f` terminal next to asitop."""

    def __init__(self, path, units):
        self.file = open(path, "w", buffering=1)
        self.device = {"CPU_AND_GPU": "GPU", "CPU_AND_NE": "Neural Engine", "CPU_ONLY": "CPU"}.get(units, units)

    def episode(self, seed):
        self.file.write(f"\033[1;36m▶ episode · seed {seed}\033[0m\n")

    def decision(self, elapsed, probs, action, ms):
        scores = "  ".join(f"{n.split('_')[-1]} {p:.2f}" for n, p in zip(NAMES, probs))
        self.file.write(
            f"{elapsed:5.1f}s  \033[33m→ {action:<24}\033[0m {scores}  "
            f"\033[1;31mmodel call {ms:.1f} ms on {self.device}\033[0m\n"
        )

    def result(self, kills, outcome, survived):
        self.file.write(f"\033[32m■ {kills} kills · {outcome} {survived:.1f} s\033[0m\n")


def open_terminals(log_path):
    """asitop needs sudo, so it runs in a real Terminal window where the password can be typed."""
    asitop = next(
        (p for p in (os.path.expanduser("~/.local/bin/asitop"), "/opt/homebrew/bin/asitop", "/usr/local/bin/asitop")
         if os.access(p, os.X_OK)),
        None,
    )
    commands = [f"clear; tail -n 0 -f {log_path}"]
    if asitop:
        commands.insert(0, f"sudo {asitop}")
    else:
        print("asitop not found: `uv tool install asitop` to show power usage")
    for command in commands:
        script = f'tell application "Terminal"\nactivate\ndo script "{command}"\nend tell'
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True)


def make_game():
    game = vizdoom.DoomGame()
    game.load_config(os.path.join(vizdoom.scenarios_path, "defend_the_center.cfg"))
    game.set_window_visible(False)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.set_screen_format(vizdoom.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_depth_buffer_enabled(True)
    game.clear_available_buttons()
    for button in ("ATTACK", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"):
        game.add_available_button(getattr(vizdoom.Button, button))
    for variable in ("HEALTH", "AMMO2", "KILLCOUNT"):
        game.add_available_game_variable(getattr(vizdoom.GameVariable, variable))
    game.set_episode_timeout(TIMEOUT)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.init()
    return game


class Policy:
    def __init__(self, path, units):
        import coremltools as ct
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(MODEL_ID, "tokenizer.json")) as f:
            self.vocab = json.load(f)["model"]["vocab"]
        started = time.perf_counter()
        self.model = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, units))
        self.load_s = time.perf_counter() - started
        self.units = units

    def __call__(self, depth):
        bins, ids, depth_ids = observe(depth, self.vocab)
        assert ids.shape[1] == SEQ
        started = time.perf_counter()
        logits = self.model.predict({"input_ids": ids, "depth_ids": depth_ids})["logits"][0].astype(np.float64)
        ms = (time.perf_counter() - started) * 1000
        probs = np.exp(logits - logits.max())
        return bins, probs / probs.sum(), ms


def check(policy, first_seed, episodes):
    game, kills, tics, ms = make_game(), [], [], []
    for seed in range(first_seed, first_seed + episodes):
        game.set_seed(seed)
        game.new_episode()
        while not game.is_episode_finished():
            _, probs, elapsed = policy(game.get_state().depth_buffer)
            ms.append(elapsed)
            game.make_action(choose(probs)[0], FRAME_SKIP)
        kills.append(int(game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)))
        tics.append(game.get_episode_time())
        print(f"seed {seed}: {kills[-1]} kills, {tics[-1] / 35:.1f} s")
    game.close()
    print(
        f"{episodes} episodes: {statistics.mean(kills):.2f} kills, {statistics.mean(tics) / 35:.1f} s, "
        f"{statistics.median(ms):.2f} ms median per decision"
    )


class Viewer:
    W, H = 1280, 720
    GAME = (40, 90, 800, 600)  # x, y, w, h

    def __init__(self, record, headless):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame

        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("SauerkrautLM-Doom on Core ML")
        mono = pygame.font.match_font("menlo,monaco,couriernew")
        sans = pygame.font.match_font("helveticaneue,helvetica,arial")
        self.ascii_font = pygame.font.Font(mono, 11)
        self.mono = pygame.font.Font(mono, 16)
        self.big = pygame.font.Font(sans, 30)
        self.label = pygame.font.Font(sans, 18)
        self.small = pygame.font.Font(sans, 15)
        self.writer = None
        if record:
            import imageio

            self.writer = imageio.get_writer(record, fps=35, codec="libx264", quality=7, macro_block_size=8)

    def text(self, font, s, pos, color=(230, 230, 230)):
        self.screen.blit(font.render(s, True, color), pos)

    def draw(self, frame, hud):
        pg, s = self.pg, self.screen
        s.fill((16, 17, 20))
        self.text(self.big, "A 1.3M-parameter model plays Doom from a 40×25 depth grid", (40, 22))
        self.text(
            self.small,
            "SauerkrautLM-Doom-MultiVec · Core ML on the Mac GPU · ViZDoom defend_the_center",
            (42, 60),
            (150, 155, 165),
        )
        x, y, w, h = self.GAME
        surface = pg.surfarray.make_surface(frame.swapaxes(0, 1))
        s.blit(pg.transform.smoothscale(surface, (w, h)), (x, y))

        px = 870
        self.text(self.label, "What the model reads: depth, near = bright", (px, 90), (150, 155, 165))
        cell_w, cell_h = 9, 12
        for gy, row in enumerate(hud["bins"]):
            for gx, b in enumerate(row):
                shade = int(235 - b * 13)
                pg.draw.rect(s, (shade // 3, shade, shade // 2), (px + gx * cell_w, 118 + gy * cell_h, cell_w - 1, cell_h - 1))

        by = 430
        self.text(self.label, "Action probabilities", (px, by), (150, 155, 165))
        for i, name in enumerate(NAMES):
            yy = by + 30 + i * 30
            p = hud["probs"][i]
            self.text(self.small, name.replace("_", " "), (px, yy))
            pg.draw.rect(s, (45, 48, 55), (px + 110, yy + 2, 250, 16), border_radius=3)
            pg.draw.rect(s, (90, 170, 250), (px + 110, yy + 2, int(250 * p), 16), border_radius=3)
            self.text(self.small, f"{p:.2f}", (px + 370, yy))
        self.text(self.small, f"pressing: {hud['action']}", (px, by + 152), (250, 200, 90))

        sy = 615
        self.text(self.mono, f"{hud['ms']:.1f} ms", (px, sy), (90, 170, 250))
        self.text(self.small, "per decision (Core ML, GPU)", (px + 90, sy + 2), (150, 155, 165))
        self.text(self.mono, f"{PYTORCH_CPU_MS:.0f} ms", (px, sy + 26), (150, 155, 165))
        self.text(self.small, "same model, PyTorch CPU (1 thread)", (px + 90, sy + 28), (150, 155, 165))
        self.text(
            self.label,
            f"kills {hud['kills']}   health {hud['health']}   ammo {hud['ammo']}   "
            f"{hud['elapsed']:4.1f} s / 60 s   seed {hud['seed']}",
            (40, 695),
        )
        if hud.get("banner"):
            b = self.big.render(hud["banner"], True, (255, 255, 255))
            pg.draw.rect(s, (0, 0, 0), (x, y + h // 2 - 30, w, 60))
            s.blit(b, (x + (w - b.get_width()) // 2, y + h // 2 - b.get_height() // 2))
        pg.display.flip()
        if self.writer:
            self.writer.append_data(pg.surfarray.array3d(s).swapaxes(0, 1))

    def events(self):
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                return "quit"
            if event.type == self.pg.KEYDOWN:
                return {self.pg.K_q: "quit", self.pg.K_SPACE: "pause", self.pg.K_n: "next"}.get(event.key)
        return None

    def close(self):
        if self.writer:
            self.writer.close()
        self.pg.quit()


def play(policy, args):
    log = DecisionLog(LOG_PATH, policy.units)
    if not args.headless and not args.no_terminals:
        open_terminals(LOG_PATH)
        time.sleep(1.0)
    viewer = Viewer(args.record, args.headless)
    game = make_game()
    realtime = not (args.record and args.headless)
    ms_window = []
    seed = args.seed
    try:
        for _ in range(args.episodes):
            game.set_seed(seed)
            game.new_episode()
            log.episode(seed)
            state = game.get_state()
            paused, quit_requested = False, False
            while not game.is_episode_finished():
                command = viewer.events()
                if command == "quit":
                    quit_requested = True
                    break
                if command == "next":
                    break
                if command == "pause":
                    paused = not paused
                if paused:
                    time.sleep(0.03)
                    continue
                bins, probs, ms = policy(state.depth_buffer)
                ms_window = (ms_window + [ms])[-30:]
                buttons, action = choose(probs)
                log.decision(game.get_episode_time() / 35, probs, action, ms)
                for _tic in range(FRAME_SKIP):  # render every tic so video runs at 35 fps
                    tic_started = time.perf_counter()
                    game.make_action(buttons, 1)
                    if game.is_episode_finished():
                        break
                    state = game.get_state()
                    health, ammo, kills = (
                        int(game.get_game_variable(getattr(vizdoom.GameVariable, v)))
                        for v in ("HEALTH", "AMMO2", "KILLCOUNT")
                    )
                    viewer.draw(
                        state.screen_buffer,
                        {
                            "bins": bins, "probs": probs, "action": action, "ms": statistics.median(ms_window),
                            "kills": kills, "health": health, "ammo": ammo,
                            "elapsed": game.get_episode_time() / 35, "seed": seed,
                        },
                    )
                    if realtime:
                        time.sleep(max(0.0, 1 / 35 - (time.perf_counter() - tic_started)))
            if quit_requested:
                break
            kills = int(game.get_game_variable(vizdoom.GameVariable.KILLCOUNT))
            survived = game.get_episode_time() / 35
            outcome = "survived" if not game.is_player_dead() else "died at"
            last = game.get_state() or state
            for _ in range(70):  # 2 s end card
                viewer.draw(
                    last.screen_buffer if last is not None else np.zeros((480, 640, 3), np.uint8),
                    {
                        "bins": bins, "probs": probs, "action": "—", "ms": statistics.median(ms_window),
                        "kills": kills, "health": 0 if game.is_player_dead() else "—", "ammo": "—",
                        "elapsed": survived, "seed": seed,
                        "banner": f"{kills} kills · {outcome} {survived:.1f} s",
                    },
                )
                if realtime:
                    time.sleep(1 / 35)
            log.result(kills, outcome, survived)
            print(f"seed {seed}: {kills} kills, {outcome} {survived:.1f} s")
            seed += 1
    finally:
        game.close()
        viewer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--units", default="CPU_AND_GPU", help="CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, ALL")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--record", default=None, help="write an mp4 of the window")
    parser.add_argument("--headless", action="store_true", help="no window; with --record, renders as fast as possible")
    parser.add_argument("--no-terminals", action="store_true", help="don't open the asitop and log Terminal windows")
    parser.add_argument("--check", type=int, default=0, help="headless score check over N seeds, no rendering")
    args = parser.parse_args()
    policy = Policy(args.model, args.units)
    print(f"Loaded {os.path.basename(args.model)} ({args.units}) in {policy.load_s:.1f} s")
    if args.check:
        check(policy, args.seed, args.check)
    else:
        play(policy, args)


if __name__ == "__main__":
    main()
