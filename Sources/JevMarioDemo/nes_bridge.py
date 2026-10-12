"""Super Mario Bros. emulator bridge for JevMarioDemo: frames in, button presses out, over stdin/stdout.

Requests (one JSON line each): {"cmd": "reset", "level": "1-3", "idle": 17} (both optional: the current level, 0 idle
frames before play) or {"cmd": "step", "buttons": <NES button bitmask>}.
Reply: one JSON line {"x", "screen_x", "screen_y", "on_ground", "done", "flag", "time"} followed by the frame as
240*256*3 raw RGB bytes. The decisions are made by the Swift app (JEV-9B on Core ML); this only emulates.

MARIO_LEVEL picks the level (default 1-1). Needs gym-super-mario-bros 7.4.0 (it bundles the NES ROM; install it yourself, local use only), nes-py, gym 0.26.2,
numpy<2 -- `demo.sh` runs it through `uv run --with ...`.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
from gym_super_mario_bros import SuperMarioBrosEnv  # noqa: E402

FLOAT_STATE, SCREEN_Y = 0x001D, 0x03B8


def reply(env, obs, done, info):
    meta = {"x": int(info.get("x_pos", 40)), "screen_x": int(env._left_x_position), "screen_y": int(env.ram[SCREEN_Y]),
            "on_ground": bool(env.ram[FLOAT_STATE] == 0), "done": bool(done), "flag": bool(info.get("flag_get", False)),
            "time": int(info.get("time", 400))}
    sys.stdout.buffer.write((json.dumps(meta) + "\n").encode())
    sys.stdout.buffer.write(obs.tobytes())
    sys.stdout.buffer.flush()


def make(level: str):
    world, stage = (int(v) for v in level.split("-"))
    return SuperMarioBrosEnv(rom_mode="vanilla", lost_levels=False, target=(world, stage))


def main():
    level = os.environ.get("MARIO_LEVEL", "1-1")
    env = make(level)
    obs, done, info = env.reset(), False, {}
    for line in sys.stdin:
        request = json.loads(line)
        if request["cmd"] == "reset":
            if request.get("level", level) != level:
                env.close()
                level = request["level"]
                env = make(level)
            obs, done, info = env.reset(), False, {}
            for _ in range(int(request.get("idle", 0))):  # a different start: enemies in other phases
                obs, _, done, info = env.step(0)
        elif request["cmd"] == "step":
            obs, _, done, info = env.step(int(request["buttons"]))
        reply(env, obs, done, info)


if __name__ == "__main__":
    main()
