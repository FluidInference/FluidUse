#!/usr/bin/env python3
"""Live bar charts for the demo's top pane: CPU / GPU utilisation and GPU power from `macmon pipe` (no sudo).

On M5 Pro / macOS 27, macmon's CPU and ANE power read 0 W and powermetrics' GPU power reads milliwatts while the GPU
is busy, so this shows the counters that do track load: P-/E-cluster usage, GPU usage and GPU watts.
"""
import json
import shutil
import subprocess
import sys

BAR = "█"
COLORS = {"P-CPU": "\033[38;5;75m", "E-CPU": "\033[38;5;111m", "GPU": "\033[38;5;214m", "GPU W": "\033[38;5;203m"}
RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"


def bar(label: str, fraction: float, text: str, width: int) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return (f"  {BOLD}{label:<6}{RESET} {COLORS[label]}{BAR * filled}{RESET}{DIM}{'·' * (width - filled)}{RESET} "
            f"{text}")


def main() -> None:
    process = subprocess.Popen(["macmon", "pipe", "-i", "500"], stdout=subprocess.PIPE, text=True)
    gpu_watts_max = 30.0
    sys.stdout.write("\033[?25l")  # hide cursor
    try:
        for line in process.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            width = max(10, shutil.get_terminal_size().columns - 30)
            p_mhz, p_use = d["pcpu_usage"]
            e_mhz, e_use = d["ecpu_usage"]
            g_mhz, g_use = d["gpu_usage"]
            g_watts = d.get("gpu_power", 0.0)
            gpu_watts_max = max(gpu_watts_max, g_watts)
            ram = d.get("memory", {})
            rows = [
                f"  {BOLD}Apple silicon load{RESET}  {DIM}macmon · 0.5 s{RESET}",
                "",
                bar("P-CPU", p_use, f"{p_use * 100:5.1f}%  {p_mhz:4d} MHz", width),
                bar("E-CPU", e_use, f"{e_use * 100:5.1f}%  {e_mhz:4d} MHz", width),
                bar("GPU", g_use, f"{g_use * 100:5.1f}%  {g_mhz:4d} MHz", width),
                bar("GPU W", g_watts / gpu_watts_max, f"{g_watts:5.1f} W", width),
            ]
            if ram:
                rows += ["", f"  {DIM}RAM {ram.get('ram_usage', 0) / 2**30:.1f} / {ram.get('ram_total', 0) / 2**30:.0f} GB"
                             f"   swap {ram.get('swap_usage', 0) / 2**30:.1f} GB{RESET}"]
            sys.stdout.write("\033[H\033[2J" + "\n".join(rows) + "\n")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h")
        process.terminate()


if __name__ == "__main__":
    main()
