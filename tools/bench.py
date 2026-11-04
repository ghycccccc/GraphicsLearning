"""Sweep InstancingLab configurations over blade counts and plot the results.

    python tools/bench.py run            # runs the sweep, appends to bench/results.csv
    python tools/bench.py plot           # writes bench/m2_submission.png from the CSV

Each run launches the app in benchmark mode (--bench-frames), which measures one
configuration, appends a CSV row and exits.
"""

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXE = ROOT / "build" / "Release" / "InstancingLab.exe"
DEFAULT_CSV = ROOT / "bench" / "results.csv"
DEFAULT_PNG = ROOT / "bench" / "m2_submission.png"
BUDGET_MS = 1000.0 / 60.0

COUNTS = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]

# (label, app arguments, largest blade count worth running)
CONFIGS = [
    ("Per-blade draws", ["--path", "per-blade"], 1_000_000),
    ("Per-blade draws, redundant state", ["--path", "per-blade", "--redundant", "1"], 300_000),
    ("Static batch", ["--path", "static-batch"], 3_000_000),
    ("Instanced (attributes)", ["--path", "instanced-attrib"], 10_000_000),
    ("Instanced (SSBO)", ["--path", "instanced-ssbo"], 10_000_000),
    ("Dynamic batch", ["--path", "dynamic-batch", "--upload", "persistent"], 300_000),
    ("Instanced, CPU wind, glBufferSubData",
     ["--path", "instanced-ssbo", "--animation", "cpu", "--upload", "subdata"], 10_000_000),
    ("Instanced, CPU wind, orphaning",
     ["--path", "instanced-ssbo", "--animation", "cpu", "--upload", "orphan"], 10_000_000),
    ("Instanced, CPU wind, persistent ring",
     ["--path", "instanced-ssbo", "--animation", "cpu", "--upload", "persistent"], 10_000_000),
]

# Which configs appear on which panel of the plot.
STATIC_PANEL = ["per-blade/0/none", "per-blade-redundant", "static-batch", "instanced-attrib", "instanced-ssbo"]
STREAM_PANEL = ["dynamic-batch", "instanced-ssbo/cpu/subdata", "instanced-ssbo/cpu/orphan",
                "instanced-ssbo/cpu/persistent", "instanced-ssbo"]

LABELS = {
    "per-blade/0/none": "Per-blade draws",
    "per-blade-redundant": "Per-blade draws, redundant state",
    "static-batch": "Static batch",
    "instanced-attrib": "Instanced (attributes)",
    "instanced-ssbo": "Instanced (SSBO), wind in VS",
    "dynamic-batch": "Dynamic batch (CPU rebuilds vertices)",
    "instanced-ssbo/cpu/subdata": "Instanced, CPU wind, glBufferSubData",
    "instanced-ssbo/cpu/orphan": "Instanced, CPU wind, orphaning",
    "instanced-ssbo/cpu/persistent": "Instanced, CPU wind, persistent ring",
}


def run(args):
    exe = Path(args.exe)
    if not exe.exists():
        sys.exit(f"{exe} not found; build the Release target first")
    csv_path = Path(args.csv)
    counts = COUNTS[:5] if args.quick else COUNTS
    for label, extra, max_count in CONFIGS:
        for count in counts:
            if count > max_count:
                continue
            cmd = [str(exe), "--instances", str(count), "--cap", str(count),
                   "--bench-frames", str(args.frames), "--bench-seconds", str(args.seconds),
                   "--csv", str(csv_path), *extra]
            print(f"{label:45s} {count:>10,d} ... ", end="", flush=True)
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            line = next((l for l in result.stdout.splitlines() if l.startswith("bench:")), None)
            print(line.split(":", 1)[1].strip() if line else f"failed (exit {result.returncode})")


def series_key(row):
    path, anim, upload = row["path"], row["animation"], row["upload"]
    if path == "per-blade":
        return "per-blade-redundant" if row["redundant"] == "1" else "per-blade/0/none"
    if path == "instanced-ssbo" and anim == "cpu":
        return f"instanced-ssbo/cpu/{upload}"
    if path in ("instanced-attrib",) and anim == "cpu":
        return None  # not part of the default sweep
    return path


def load(csv_path):
    series = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["drawn"]) < int(row["instances"]):
                continue  # capped run: did not draw everything it was asked to
            key = series_key(row)
            if key is None:
                continue
            # Later rows for the same point replace earlier ones (re-runs).
            series.setdefault(key, {})[int(row["instances"])] = row
    return {k: sorted(v.items()) for k, v in series.items()}


def budget_capacity(points):
    """Blade count at which the frame time crosses the budget (log-log interpolation)."""
    prev = None
    for n, row in points:
        ms = float(row["frame_avg"])
        if ms > BUDGET_MS:
            if prev is None:
                return f"< {n:,}"
            (n0, ms0) = prev
            t = (math.log(BUDGET_MS) - math.log(ms0)) / (math.log(ms) - math.log(ms0))
            return f"~{int(math.exp(math.log(n0) + t * (math.log(n) - math.log(n0)))):,}"
        prev = (n, ms)
    return f">= {points[-1][0]:,}" if points else "-"


def plot(args):
    import matplotlib.pyplot as plt

    series = load(args.csv)
    if not series:
        sys.exit("no usable rows in the CSV")
    gpu = next(iter(series.values()))[0][1]["gpu"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    panels = [
        (axes[0], STATIC_PANEL, "Static per-blade data: submission cost"),
        (axes[1], STREAM_PANEL, "Wind computed on the CPU every frame: update + upload cost"),
    ]
    for ax, keys, title in panels:
        for key in keys:
            points = series.get(key)
            if not points:
                continue
            xs = [n for n, _ in points]
            ys = [float(r["frame_avg"]) for _, r in points]
            style = "--" if key == "instanced-ssbo" and keys is STREAM_PANEL else "-"
            ax.plot(xs, ys, style, marker="o", markersize=4,
                    label=f"{LABELS[key]}  [60 fps up to {budget_capacity(points)}]")
        ax.axhline(BUDGET_MS, color="gray", linewidth=1, linestyle=":")
        ax.text(1.05e4, BUDGET_MS * 1.08, "60 fps budget", color="gray", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("blades (7 triangles each)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("average frame time (ms)")
    fig.suptitle(f"Milestone 2: submission and upload paths ({gpu})", fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    # Text summary for quick reading.
    for key, points in series.items():
        print(f"{LABELS.get(key, key):45s} 60 fps capacity {budget_capacity(points)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run the sweep")
    p_run.add_argument("--exe", default=str(DEFAULT_EXE))
    p_run.add_argument("--csv", default=str(DEFAULT_CSV))
    p_run.add_argument("--frames", type=int, default=120)
    p_run.add_argument("--seconds", type=float, default=15.0)
    p_run.add_argument("--quick", action="store_true", help="stop at 1M blades")
    p_run.set_defaults(func=run)
    p_plot = sub.add_parser("plot", help="plot the CSV")
    p_plot.add_argument("--csv", default=str(DEFAULT_CSV))
    p_plot.add_argument("--out", default=str(DEFAULT_PNG))
    p_plot.set_defaults(func=plot)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
