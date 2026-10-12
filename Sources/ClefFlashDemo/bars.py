#!/usr/bin/env python3
"""Live bar charts for the demo's top pane: CPU / GPU utilisation and GPU power from `macmon pipe` (no sudo); with
--ane also Neural Engine power from `sudo powermetrics` (asks for the password once).

On M5 Pro / macOS 27, macmon's CPU and ANE power read 0 W and powermetrics' GPU power reads milliwatts while the GPU
is busy, so this shows the counters that do track load: P-/E-cluster usage, GPU usage and GPU watts.
"""
from __future__ import annotations

import json
import plistlib
import shutil
import subprocess
import sys
import threading
import time

BAR = "█"
COLORS = {"P-CPU": "\033[38;5;75m", "E-CPU": "\033[38;5;111m", "GPU": "\033[38;5;214m", "GPU W": "\033[38;5;203m",
          "ANE W": "\033[38;5;120m", "ANE": "\033[38;5;84m"}
RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"


def bar(label: str, fraction: float, text: str, width: int) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return (f"  {BOLD}{label:<6}{RESET} {COLORS[label]}{BAR * filled}{RESET}{DIM}{'·' * (width - filled)}{RESET} "
            f"{text}")


def find_ane_watts(node) -> float | None:
    """ANE power from a powermetrics plist sample (key names vary by macOS / chip: ane_power in mW, ane_energy ...)."""
    if isinstance(node, dict):
        for key, value in node.items():
            k = key.lower()
            if "ane" in k and "power" in k and isinstance(value, (int, float)):
                return float(value) / 1000  # mW
        for value in node.values():
            found = find_ane_watts(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_ane_watts(value)
            if found is not None:
                return found
    return None


class AneMeter:
    """`sudo powermetrics --samplers ane_power` (macmon's ANE power reads 0 W on macOS 27). Latest watts + peak."""

    def __init__(self):
        self.watts, self.peak, self.error = 0.0, 0.0, None
        self.process = subprocess.Popen(
            ["sudo", "powermetrics", "--samplers", "ane_power", "-i", "500", "-f", "plist"], stdout=subprocess.PIPE)
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        buffer = b""
        for chunk in iter(lambda: self.process.stdout.read(4096), b""):
            buffer += chunk
            while b"\0" in buffer:
                sample, buffer = buffer.split(b"\0", 1)
                try:
                    watts = find_ane_watts(plistlib.loads(sample.strip()))
                except Exception as e:  # noqa: BLE001
                    self.error = str(e)[:60]
                    continue
                if watts is None:
                    self.error = "no ANE power key in powermetrics output"
                    continue
                self.watts, self.peak = watts, max(self.peak, watts)


class AneBusy:
    """Neural Engine busy time from the demo's own decision log (`... 0.55 ms` per line): the model's ops all run on
    the ANE (MLComputePlan), so model milliseconds over the last second = ANE busy %. No sudo, no hardware counter."""

    def __init__(self, path: str, window: float = 1.0):
        self.window, self.events, self.placement = window, [], ""
        try:  # the placement line is printed once at load, possibly before this pane started
            self.placement = next((l.split(",", 1)[-1].strip() for l in open(path) if "Neural Engine" in l), "")
        except OSError:
            pass
        self.process = subprocess.Popen(["tail", "-n", "0", "-F", path], stdout=subprocess.PIPE, text=True,
                                        stderr=subprocess.DEVNULL)
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        for line in self.process.stdout:
            parts = line.split()
            if "Neural Engine" in line and "ops" in line:
                self.placement = line.split(",", 1)[-1].strip()
            if len(parts) >= 2 and parts[-1] == "ms":
                try:
                    self.events.append((time.time(), float(parts[-2])))
                except ValueError:
                    pass

    def stats(self) -> tuple[float, float, float]:
        """(busy fraction, median ms per decision, decisions per second) over the window."""
        now = time.time()
        self.events = [e for e in self.events if now - e[0] <= max(self.window, 5.0)]
        recent = [ms for t, ms in self.events if now - t <= self.window]
        busy = sum(recent) / (self.window * 1000)
        ms = sorted(recent)[len(recent) // 2] if recent else 0.0
        return busy, ms, len(recent) / self.window


def main() -> None:
    busy = AneBusy(sys.argv[sys.argv.index("--log") + 1]) if "--log" in sys.argv else None
    ane = None
    if "--ane" in sys.argv:
        print("ANE power needs `sudo powermetrics`:", flush=True)
        subprocess.run(["sudo", "-v"])  # password prompt here, before the screen starts redrawing
        ane = AneMeter()
    process = subprocess.Popen(["macmon", "pipe", "-i", "500"], stdout=subprocess.PIPE, text=True)
    gpu_watts_max = 30.0
    sys.stdout.write("\033[?25l")  # hide cursor
    try:
        for line in process.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # label (9) + space + bar + space + the widest right-hand text, so nothing clips at any window width
            width = max(10, shutil.get_terminal_size().columns - 9 - 2 - (24 if busy else 15) - 2)
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
            if busy is not None:
                fraction, ms, rate = busy.stats()
                # the bar is drawn at least one cell while the model runs: busy time is a fraction of a percent
                cells = max(1 if rate else 0, int(round(fraction * width)))
                rows.append(f"  {BOLD}{'ANE':<6}{RESET} {COLORS['ANE']}{BAR * cells}{RESET}{DIM}{'·' * (width - cells)}{RESET} "
                            f"{fraction * 100:5.2f}%  {ms:4.2f} ms × {rate:.0f}/s")
                if busy.placement:
                    ops = busy.placement.split("·")[-1].strip()
                    rows.append(f"  {DIM}ANE % = model time per second · {ops} on the Neural Engine{RESET}")
            if ane is not None:
                # no utilisation counter exists for the ANE; power vs. a 2 W scale (or the session peak) stands in
                scale = max(2.0, ane.peak)
                rows.append(bar("ANE W", ane.watts / scale, f"{ane.watts * 1000:5.0f} mW", width))
                if ane.error:
                    rows.append(f"  {DIM}ANE: {ane.error}{RESET}")
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
