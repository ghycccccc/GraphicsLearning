"""Sweep InstancingLab configurations over blade counts and plot the results.

    python tools/bench.py run  --suite m3     # runs the sweep, appends to bench/m3_results.csv
    python tools/bench.py plot --suite m3     # writes bench/m3_residency.png from that CSV

Suites:
    m2  submission and upload paths (per-blade draws, batching, instancing)
    m3  data residency (VS wind vs CPU spring sim vs compute), upload scope, layout, readback

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
BUDGET_MS = 1000.0 / 60.0
K, M = 1_000, 1_000_000

# --- Suite definitions -----------------------------------------------------------
# Each config: (label, app arguments, blade counts)

M2_COUNTS = [10 * K, 30 * K, 100 * K, 300 * K, 1 * M, 3 * M, 10 * M]
M2_CONFIGS = [
    ("Per-blade draws", ["--path", "per-blade"], [n for n in M2_COUNTS if n <= 1 * M]),
    ("Per-blade draws, redundant state", ["--path", "per-blade", "--redundant", "1"],
     [n for n in M2_COUNTS if n <= 300 * K]),
    ("Static batch", ["--path", "static-batch"], [n for n in M2_COUNTS if n <= 3 * M]),
    ("Instanced (attributes)", ["--path", "instanced-attrib"], M2_COUNTS),
    ("Instanced (SSBO)", ["--path", "instanced-ssbo"], M2_COUNTS),
    ("Dynamic batch", ["--path", "dynamic-batch", "--upload", "persistent"], [n for n in M2_COUNTS if n <= 300 * K]),
] + [
    (f"Instanced, CPU wind, {upload}",
     ["--path", "instanced-ssbo", "--animation", "cpu", "--upload", upload, "--upload-scope", "full"], M2_COUNTS)
    for upload in ("subdata", "orphan", "persistent")
]

M3_COUNTS = [10 * K, 30 * K, 100 * K, 300 * K, 1 * M, 3 * M, 10 * M, 16 * M]
M3_SSBO = ["--path", "instanced-ssbo", "--upload", "persistent"]
M3_CONFIGS = [
    # A: where the simulation runs
    ("VS wind", M3_SSBO + ["--animation", "vs"], M3_COUNTS),
    ("Spring, CPU 1 thread", M3_SSBO + ["--animation", "spring-cpu"], [n for n in M3_COUNTS if n <= 3 * M]),
    ("Spring, CPU all cores", M3_SSBO + ["--animation", "spring-mt"], [n for n in M3_COUNTS if n <= 10 * M]),
    ("Spring, compute", M3_SSBO + ["--animation", "spring-gpu"], M3_COUNTS),
    # B: what the CPU path uploads
    ("Spring, CPU all cores, full upload", M3_SSBO + ["--animation", "spring-mt", "--upload-scope", "full"],
     [n for n in M3_COUNTS if n <= 10 * M]),
] + [
    # C: instance layout in the GPU-bound regime
    (f"Layout {layout}, {name}", ["--path", path, "--animation", anim, "--layout", layout], [3 * M, 10 * M, 16 * M])
    for name, path, anim in [("SSBO VS wind", "instanced-ssbo", "vs"),
                             ("SSBO compute", "instanced-ssbo", "spring-gpu"),
                             ("attributes VS wind", "instanced-attrib", "vs")]
    for layout in ("float", "packed")
] + [
    # D: reading simulation results back on the CPU, with and without other CPU work in the frame
    (f"Readback {mode}, game logic {logic} ms",
     M3_SSBO + ["--animation", "spring-gpu", "--readback", mode, "--game-logic-ms", str(logic)], [1 * M, 10 * M])
    for logic in (0, 10)
    for mode in ("none", "sync", "async")
]

SUITES = {
    "m2": {"configs": M2_CONFIGS, "csv": ROOT / "bench" / "m2_results.csv", "png": ROOT / "bench" / "m2_submission.png"},
    "m3": {"configs": M3_CONFIGS, "csv": ROOT / "bench" / "m3_results.csv", "png": ROOT / "bench" / "m3_residency.png"},
}


def run(args):
    exe = Path(args.exe)
    if not exe.exists():
        sys.exit(f"{exe} not found; build the Release target first")
    suite = SUITES[args.suite]
    csv_path = Path(args.csv or suite["csv"])
    for label, extra, counts in suite["configs"]:
        for count in counts:
            if args.max_count and count > args.max_count:
                continue
            cmd = [str(exe), "--instances", str(count), "--cap", str(count),
                   "--bench-frames", str(args.frames), "--bench-seconds", str(args.seconds),
                   "--csv", str(csv_path), *extra]
            print(f"{label:45s} {count:>11,d} ... ", end="", flush=True)
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            line = next((l for l in result.stdout.splitlines() if l.startswith("bench:")), None)
            print(line.split(":", 1)[1].strip() if line else f"failed (exit {result.returncode})")


# --- Loading and helpers ---------------------------------------------------------------

def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["drawn"]) < int(row["instances"]):
                continue  # capped run: did not draw everything it was asked to
            rows.append(row)
    return rows


def select(rows, **match):
    """Rows whose fields equal `match`, keyed by blade count (later rows win)."""
    points = {}
    for row in rows:
        if all(row.get(k, "none") == v for k, v in match.items()):
            points[int(row["instances"])] = row
    return sorted(points.items())


def budget_capacity(points):
    """Blade count at which the frame time crosses the budget (log-log interpolation)."""
    prev = None
    for n, row in points:
        ms = float(row["frame_avg"])
        if ms > BUDGET_MS:
            if prev is None:
                return f"< {n:,}"
            n0, ms0 = prev
            t = (math.log(BUDGET_MS) - math.log(ms0)) / (math.log(ms) - math.log(ms0))
            return f"~{int(math.exp(math.log(n0) + t * (math.log(n) - math.log(n0)))):,}"
        prev = (n, ms)
    return f">= {points[-1][0]:,}" if points else "-"


def line_panel(ax, series, title, ylabel="average frame time (ms)", field="frame_avg", budget=True):
    for label, points, style in series:
        if not points:
            continue
        xs = [n for n, _ in points]
        ys = [max(float(r[field]), 1e-3) for _, r in points]
        suffix = f"  [60 fps up to {budget_capacity(points)}]" if budget else ""
        ax.plot(xs, ys, style, marker="o", markersize=4, label=label + suffix)
    if budget:
        ax.axhline(BUDGET_MS, color="gray", linewidth=1, linestyle=":")
        ax.text(1.05e4, BUDGET_MS * 1.08, "60 fps budget", color="gray", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("blades (7 triangles each)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")


def save(fig, out, title, gpu):
    fig.suptitle(f"{title} ({gpu})", fontsize=12)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


# --- Plots ------------------------------------------------------------------------------

def plot_m2(rows, out):
    import matplotlib.pyplot as plt

    def series(path, **extra):
        return select(rows, path=path, **extra)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    static = [
        ("Per-blade draws", series("per-blade", redundant="0"), "-"),
        ("Per-blade draws, redundant state", series("per-blade", redundant="1"), "-"),
        ("Static batch", series("static-batch"), "-"),
        ("Instanced (attributes)", series("instanced-attrib", animation="vs"), "-"),
        ("Instanced (SSBO), wind in VS", series("instanced-ssbo", animation="vs"), "-"),
    ]
    stream = [("Dynamic batch (CPU rebuilds vertices)", series("dynamic-batch"), "-")] + [
        (f"Instanced, CPU wind, {name}", series("instanced-ssbo", animation="cpu", upload=key), "-")
        for key, name in [("subdata", "glBufferSubData"), ("orphan", "orphaning"), ("persistent", "persistent ring")]
    ] + [("Instanced (SSBO), wind in VS", series("instanced-ssbo", animation="vs"), "--")]
    line_panel(axes[0], static, "Static per-blade data: submission cost")
    line_panel(axes[1], stream, "Wind computed on the CPU every frame: update + upload cost")
    save(fig, out, "Milestone 2: submission and upload paths", rows[0]["gpu"])


def plot_m3(rows, out):
    import matplotlib.pyplot as plt
    import numpy as np

    ssbo = dict(path="instanced-ssbo", layout="float", game_logic_ms="0")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # A: simulation residency
    line_panel(axes[0][0], [
        ("VS wind (stateless, no sim)", select(rows, animation="vs", readback="none", **ssbo), "--"),
        ("Spring sim, CPU 1 thread", select(rows, animation="spring-cpu", upload_scope="changed", **ssbo), "-"),
        ("Spring sim, CPU all cores", select(rows, animation="spring-mt", upload_scope="changed", **ssbo), "-"),
        ("Spring sim, compute shader", select(rows, animation="spring-gpu", readback="none", **ssbo), "-"),
    ], "A. Where the simulation runs")

    # B: bytes uploaded per frame
    changed = select(rows, animation="spring-mt", upload_scope="changed", **ssbo)
    full = select(rows, animation="spring-mt", upload_scope="full", **ssbo)
    line_panel(axes[0][1], [
        ("CPU sim, full instance data (40 B/blade)", full, "-"),
        ("CPU sim, changed bends only (8 B/blade)", changed, "-"),
    ], "B. Upload per frame (compute sim uploads 0 B)", ylabel="MB uploaded per frame", field="upload_mb",
        budget=False)
    ax_b = axes[0][1]
    for label, points in [("full", full), ("changed", changed)]:
        if points:
            n, row = points[-1]
            ax_b.annotate(f"{float(row['frame_avg']):.0f} ms/frame", (n, float(row["upload_mb"])),
                          textcoords="offset points", xytext=(-60, 8), fontsize=8)

    # C: layout, GPU time only
    ax = axes[1][0]
    groups = [("SSBO, VS wind", "instanced-ssbo", "vs"), ("SSBO, compute sim", "instanced-ssbo", "spring-gpu"),
              ("Attributes, VS wind", "instanced-attrib", "vs")]
    counts = [3 * M, 10 * M, 16 * M]
    width = 0.13
    x = np.arange(len(counts))
    for gi, (name, path, anim) in enumerate(groups):
        for li, layout in enumerate(("float", "packed")):
            points = dict(select(rows, path=path, animation=anim, layout=layout, readback="none", game_logic_ms="0"))
            ys = [float(points[n]["gpu_avg"]) if n in points else 0 for n in counts]
            offset = (gi * 2 + li - 2.5) * width
            bars = ax.bar(x + offset, ys, width, label=f"{name}, {'32 B' if layout == 'float' else '16 B'}",
                          hatch="" if layout == "float" else "//", color=f"C{gi}", alpha=0.9 if li == 0 else 0.55)
            ax.bar_label(bars, fmt="%.1f", fontsize=6)
    ax.set_xticks(x, [f"{n // M}M blades" for n in counts])
    ax.set_ylabel("GPU work per frame (ms)")
    ax.set_title("C. Instance layout: 32 B floats vs 16 B packed (GPU-bound)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    # D: readback, grouped by blade count and amount of other CPU work per frame
    ax = axes[1][1]
    modes = [("none", "No readback"), ("sync", "Sync glGetBufferSubData"), ("async", "Async, 2 frames late")]
    groups = [(n, logic) for n in (1 * M, 10 * M) for logic in ("0", "10")]
    x = np.arange(len(groups))
    width = 0.25
    base = dict(path="instanced-ssbo", layout="float", animation="spring-gpu")
    for mi, (mode, name) in enumerate(modes):
        rows_by_group = [dict(select(rows, readback=mode, game_logic_ms=logic, **base)).get(n) for n, logic in groups]
        ys = [float(r["frame_avg"]) if r else 0 for r in rows_by_group]
        bars = ax.bar(x + (mi - 1) * width, ys, width, label=name)
        labels = [f"{float(r['frame_avg']):.1f}" + (f"\n(stall {float(r['readback_ms']):.1f})" if mode == "sync" else "")
                  if r else "" for r in rows_by_group]
        ax.bar_label(bars, labels=labels, fontsize=7)
    ax.set_xticks(x, [f"{n // M}M blades\n+{logic} ms game logic" for n, logic in groups])
    ax.set_ylabel("average frame time (ms)")
    ax.set_title("D. Reading GPU simulation results on the CPU", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    save(fig, out, "Milestone 3: data residency", rows[0]["gpu"])

    print("60 fps capacity:")
    for name, anim in [("VS wind", "vs"), ("CPU 1 thread", "spring-cpu"), ("CPU all cores", "spring-mt"),
                       ("Compute", "spring-gpu")]:
        extra = {"upload_scope": "changed"} if anim in ("spring-cpu", "spring-mt") else {"readback": "none"}
        print(f"  {name:15s} {budget_capacity(select(rows, animation=anim, **extra, **ssbo))}")


def plot(args):
    suite = SUITES[args.suite]
    rows = load_rows(args.csv or suite["csv"])
    if not rows:
        sys.exit("no usable rows in the CSV")
    (plot_m2 if args.suite == "m2" else plot_m3)(rows, args.out or suite["png"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run a sweep")
    p_run.add_argument("--suite", choices=SUITES, default="m3")
    p_run.add_argument("--exe", default=str(DEFAULT_EXE))
    p_run.add_argument("--csv")
    p_run.add_argument("--frames", type=int, default=120)
    p_run.add_argument("--seconds", type=float, default=15.0)
    p_run.add_argument("--max-count", type=int, help="skip blade counts above this")
    p_run.set_defaults(func=run)
    p_plot = sub.add_parser("plot", help="plot a suite's CSV")
    p_plot.add_argument("--suite", choices=SUITES, default="m3")
    p_plot.add_argument("--csv")
    p_plot.add_argument("--out")
    p_plot.set_defaults(func=plot)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
