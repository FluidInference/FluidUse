"""Super Mario Bros. emulator bridge on stable-retro (unlimited save states, any level start), for JevMarioDemo.

Requests, one JSON line each:
  {"cmd": "reset", "level": "1-3", "idle": 0}             start a level (optional idle frames)
  {"cmd": "step", "buttons": 130}                          one frame
  {"cmd": "run", "buttons": [130, 131, ...], "hold": true} a whole move in one call; with hold, the last input is
                                                           held while the game has control (pipe entry, ...)
Reply: one JSON line {"x", "screen_x", "screen_y", "on_ground", "done", "flag", "time", "frames", "images"} then
`images` raw 240x256 RGB frames: one for reset / step, two for run (the last frame and the one 4 frames before it,
which the pixel student needs). Buttons are NES bitmasks (0x80 right, 0x40 left, 0x20 down, 0x10 up, 0x02 B, 0x01 A).

Needs stable-retro and gym-super-mario-bros 7.4.0 (whose bundled ROM is copied in as retro's rom.nes; local use
only). demo.sh runs it through `uv run --with ...`.
"""
import collections
import json
import os
import shutil
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import stable_retro as retro  # noqa: E402

GAME = "SuperMarioBros-Nes-v0"
ORDER = ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
BITS = {"RIGHT": 0x80, "LEFT": 0x40, "DOWN": 0x20, "UP": 0x10, "START": 0x08, "SELECT": 0x04, "B": 0x02, "A": 0x01}


def install_rom():
    target = os.path.join(os.path.dirname(retro.data.__file__), "stable", GAME, "rom.nes")
    if not os.path.exists(target):
        import gym_super_mario_bros

        source = os.path.join(os.path.dirname(gym_super_mario_bros.__file__), "_roms", "super-mario-bros.nes")
        shutil.copy(source, target)


def action(mask: int) -> np.ndarray:
    return np.array([1 if name and mask & BITS[name] else 0 for name in ORDER], np.int8)


def area_for(world, stage):
    return stage + 1 if world in (1, 2, 4, 7) and stage >= 2 else stage


class Game:
    def __init__(self):
        install_rom()
        self.env = retro.make(GAME, state="Level1-1", use_restricted_actions=retro.Actions.ALL, render_mode=None)
        self.env.reset()
        self.u, self.em = self.env.unwrapped, self.env.unwrapped.em

    @property
    def ram(self):
        return self.u.get_ram()

    def poke(self, address, value):
        self.u.data.memory.assign(address, "|u1", int(value))

    def frame(self) -> bytes:
        out = np.zeros((240, 256, 3), np.uint8)
        s = self.em.get_screen()
        out[8:8 + s.shape[0], 8:8 + s.shape[1]] = s
        return out.tobytes()

    def dying(self):
        r = self.ram
        return int(r[0x0E]) in (0x06, 0x0B) or int(r[0xB5]) > 1

    def flag(self):
        r = self.ram
        return int(r[0x1D]) == 3 or int(r[0x770]) == 2

    def reset(self, level, idle):
        world, stage = (int(v) for v in level.split("-"))
        self.env.reset()
        for _ in range(30):
            self.env.step(action(0))
        for address, value in ((0x75F, world - 1), (0x75C, stage - 1), (0x760, area_for(world, stage) - 1),
                               (0x75B, 0), (0x75A, 9), (0xB5, 2)):
            self.poke(address, value)  # target level, then a death reload starts it from the beginning
        for _ in range(900):
            self.env.step(action(0))
            r = self.ram
            if int(r[0x770]) == 1 and int(r[0x0E]) == 8 and int(r[0x7A0]) == 0 and int(r[0xB5]) == 1:
                break
        for _ in range(idle):
            self.env.step(action(0))

    def step(self, mask):
        self.env.step(action(mask))

    def meta(self, frames, images):
        r = self.ram
        done = self.flag() or self.dying()
        return {"x": int(r[0x6D]) * 256 + int(r[0x86]), "screen_x": (int(r[0x86]) - int(r[0x71C])) % 256,
                "screen_y": int(r[0xCE]), "on_ground": int(r[0x1D]) == 0, "done": done, "flag": self.flag(),
                "time": int(r[0x7F8]) * 100 + int(r[0x7F9]) * 10 + int(r[0x7FA]), "frames": frames, "images": images,
                "control": int(r[0x0E]) == 8}


def reply(meta, frames):
    sys.stdout.buffer.write((json.dumps(meta) + "\n").encode())
    for f in frames:
        sys.stdout.buffer.write(f)
    sys.stdout.buffer.flush()


def main():
    game = Game()
    game.reset(os.environ.get("MARIO_LEVEL", "1-1"), 0)
    for line in sys.stdin:
        request = json.loads(line)
        cmd = request["cmd"]
        if cmd == "reset":
            game.reset(request.get("level", "1-1"), int(request.get("idle", 0)))
            reply(game.meta(0, 1), [game.frame()])
        elif cmd == "step":
            game.step(int(request["buttons"]))
            reply(game.meta(1, 1), [game.frame()])
        elif cmd == "run":
            masks, recent, n = request["buttons"], collections.deque(maxlen=5), 0
            for mask in masks:
                game.step(int(mask))
                n += 1
                recent.append(game.frame())
                if game.flag() or game.dying():
                    break
            if request.get("hold") and not (game.flag() or game.dying()):
                for _ in range(300):  # the game has control (pipe entry, area change): hold the last input
                    if int(game.ram[0x0E]) == 8:
                        break
                    game.step(int(masks[-1]))
                    n += 1
                    recent.append(game.frame())
                    if game.flag() or game.dying():
                        break
            last = recent[-1] if recent else game.frame()
            reply(game.meta(n, 2), [last, recent[0] if len(recent) == 5 else last])


if __name__ == "__main__":
    main()
