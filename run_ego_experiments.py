"""
run_ego_experiments.py

Runs GLASS with Ego-graph sampling across a specified dataset and multiple
num_seeds values. Logs results to logs/EgoNet/ with the same format as
logs/Sampling/ (NeighborSampler) for direct comparison.

Algorithm (per forward pass):
    1. Sample num_seeds seed (ego-center) nodes uniformly at random.
    2. Expand 1 hop: collect all direct neighbors of each seed.
    3. Keep ALL edges in the induced subgraph of
       {seeds} ∪ {all neighbors of seeds}.

Difference from GraphSAINT random walk:
    GraphSAINT follows random walk paths — can drift far from roots.
    EgoNet always expands exactly 1 hop — dense star-shaped neighborhoods,
    preserves local clustering (triangles) within each ego graph.

Usage:
    python run_ego_experiments.py --dataset density
    python run_ego_experiments.py --dataset coreness
    python run_ego_experiments.py --dataset cut_ratio
    python run_ego_experiments.py --dataset component

Output:
    logs/EgoNet/
        {dataset}_baseline.log
        {dataset}_s50.log       ← num_seeds=50
        {dataset}_s100.log
        {dataset}_s200.log
        {dataset}_s500.log
        {dataset}_s1000.log
        {dataset}_summary.txt
"""

import subprocess
import os
import sys
import time
import argparse
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNTHETIC_DATASETS = ["density", "cut_ratio", "coreness", "component"]
REALWORLD_DATASETS = ["ppi_bp", "hpo_metab", "hpo_neuro", "em_user"]

# num_seeds values — None always runs first as baseline (no sampling)
DEFAULT_SEEDS = [50, 100, 200, 500, 1000]

REPEAT = 3
DEVICE = -1

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Run GLASS EgoNet ego-graph sampling experiments."
)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--repeat",  type=int, default=REPEAT)
parser.add_argument("--device",  type=int, default=DEVICE)
parser.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="num_seeds values to test. Baseline always runs first.")
args = parser.parse_args()

DATASET  = args.dataset
REPEAT   = args.repeat
DEVICE   = args.device
S_LIST   = args.seeds if args.seeds is not None else DEFAULT_SEEDS
S_VALUES = [None] + S_LIST   # None = baseline

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

ALL_DATASETS = SYNTHETIC_DATASETS + REALWORLD_DATASETS
if DATASET not in ALL_DATASETS:
    print(f"ERROR: Unknown dataset '{DATASET}'. Valid: {ALL_DATASETS}")
    sys.exit(1)

FEATURE_FLAG = "--use_one" if DATASET in SYNTHETIC_DATASETS else "--use_nodeid"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOG_DIR = "logs/EgoNet"
os.makedirs(LOG_DIR, exist_ok=True)


def s_label(s):
    return "baseline" if s is None else f"s{s}"


def log_path(s):
    return f"{LOG_DIR}/{DATASET}_{s_label(s)}.log"


def summary_path():
    return f"{LOG_DIR}/{DATASET}_summary.txt"

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

def build_command(s):
    cmd = [
        "python", "GLASSTest.py",
        FEATURE_FLAG,
        "--use_seed",
        "--use_maxzeroone",
        "--repeat", str(REPEAT),
        "--device", str(DEVICE),
        "--dataset", DATASET,
    ]
    if s is not None:
        cmd += ["--sampler", "ego", "--num_seeds", str(s)]
    return cmd

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment(s):
    label    = s_label(s)
    cmd      = build_command(s)
    log_file = log_path(s)

    print(f"\n{'='*65}")
    print(f"  Dataset: {DATASET}   Sampler: EgoNet   Seeds: {label}   Repeat: {REPEAT}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log:     {log_file}")
    print(f"{'='*65}")

    start = time.time()
    with open(log_file, "w") as f:
        f.write(f"{'='*65}\n")
        f.write(f"GLASS EgoNet Ego-Graph Experiment\n")
        f.write(f"Dataset:    {DATASET}\n")
        f.write(f"Sampler:    {label}\n")
        f.write(f"num_seeds:  {s if s is not None else 'N/A (full adj, no sampling)'}\n")
        f.write(f"Repeat:     {REPEAT}\n")
        f.write(f"Device:     {DEVICE}\n")
        f.write(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command:    {' '.join(cmd)}\n")
        f.write(f"{'='*65}\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1
        )
        output_lines = []
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
            output_lines.append(line)

        process.wait()
        elapsed = time.time() - start

        f.write(f"\n{'='*65}\n")
        f.write(f"Exit code: {process.returncode}\n")
        f.write(f"Wall time: {elapsed:.1f}s\n")
        f.write(f"{'='*65}\n")

    return parse_results(output_lines, elapsed, process.returncode)

# ---------------------------------------------------------------------------
# Parse results  (identical regex to run_neighbor_sampling_experiments.py)
# ---------------------------------------------------------------------------

def parse_results(lines, wall_time, returncode):
    result = {
        "avg_f1": None, "std_err": None, "train_time": None,
        "wall_time": wall_time, "returncode": returncode,
        "edges_kept": None, "edges_total": None,
        "edge_pct": None, "edge_reduction": None,
    }
    train_time_sum, train_time_count = 0.0, 0

    for line in lines:
        m = re.search(r"average\s+([\d.]+)\s+error\s+([\d.]+)", line)
        if m:
            result["avg_f1"]  = float(m.group(1))
            result["std_err"] = float(m.group(2))

        m = re.search(r"end: epoch\s+\d+,\s+train time\s+([\d.]+)\s+s", line)
        if m:
            train_time_sum   += float(m.group(1))
            train_time_count += 1

        m = re.search(
            r"\[EdgeCoverage\] kept=(\d+)/(\d+) \(([\d.]+)% of edges\) reduction=([\d.]+)",
            line
        )
        if m:
            result["edges_kept"]     = int(m.group(1))
            result["edges_total"]    = int(m.group(2))
            result["edge_pct"]       = float(m.group(3))
            result["edge_reduction"] = float(m.group(4))

    if train_time_count > 0:
        result["train_time"] = train_time_sum / train_time_count
    return result


def parse_existing_log(path):
    with open(path) as f:
        lines = f.readlines()
    wall_time, returncode = 0.0, 0
    for line in lines:
        m = re.search(r"Wall time:\s*([\d.]+)s", line)
        if m: wall_time = float(m.group(1))
        m = re.search(r"Exit code:\s*(-?\d+)", line)
        if m: returncode = int(m.group(1))
    return parse_results(lines, wall_time, returncode)

# ---------------------------------------------------------------------------
# Write summary  (same table format as logs/Sampling/)
# ---------------------------------------------------------------------------

def write_summary(all_results):
    path = summary_path()
    with open(path, "w") as f:
        f.write(f"GLASS EgoNet Ego-Graph — Results Summary\n")
        f.write(f"{'='*65}\n")
        f.write(f"Dataset:  {DATASET}\n")
        f.write(f"Feature:  {FEATURE_FLAG}\n")
        f.write(f"Repeat:   {REPEAT}\n")
        f.write(f"Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*65}\n\n")

        f.write("[ Accuracy & Timing ]\n\n")
        h = f"{'Method':<23} {'Avg F1':>8} {'±Err':>7} {'Time/epoch':>12} {'Wall Time':>11}"
        f.write(h + "\n")
        f.write("-" * len(h) + "\n")
        for s, res in all_results.items():
            label = s_label(s)
            avg_f1     = f"{res['avg_f1']:.4f}"      if res['avg_f1']     is not None else "FAILED"
            std_err    = f"{res['std_err']:.4f}"     if res['std_err']    is not None else "N/A"
            train_time = f"{res['train_time']:.1f}s" if res['train_time'] is not None else "N/A"
            wall_time  = f"{res['wall_time']:.1f}s"
            f.write(f"{'GLASS-Ego ('+label+')':<23} {avg_f1:>8} {std_err:>7} "
                    f"{train_time:>12} {wall_time:>11}\n")

        f.write("\n\n[ Edge Coverage — Memory Impact of Sampling ]\n\n")
        h2 = f"{'Method':<23} {'Edges Kept':>12} {'Total Edges':>13} {'% Kept':>8} {'% Reduced':>11}"
        f.write(h2 + "\n")
        f.write("-" * len(h2) + "\n")
        for s, res in all_results.items():
            label = s_label(s)
            if s is None:
                te = next((v["edges_total"] for v in all_results.values()
                           if v["edges_total"] is not None), None)
                if te is not None:
                    edges_kept = edges_total = f"{te:,}"
                    pct_kept = "100.0%"; pct_reduced = "0.0%"
                else:
                    edges_kept = edges_total = pct_kept = pct_reduced = "N/A"
            else:
                if res["edges_kept"] is not None:
                    edges_kept  = f"{res['edges_kept']:,}"
                    edges_total = f"{res['edges_total']:,}"
                    pct_kept    = f"{res['edge_pct']:.1f}%"
                    pct_reduced = f"{res['edge_reduction']*100:.1f}%"
                else:
                    edges_kept = edges_total = pct_kept = pct_reduced = "N/A"
            f.write(f"{'GLASS-Ego ('+label+')':<23} {edges_kept:>12} {edges_total:>13} "
                    f"{pct_kept:>8} {pct_reduced:>11}\n")

        f.write("\n\n[ Notes ]\n\n")
        f.write("Sampler:    EgoNet ego-graph sampler\n")
        f.write("num_seeds:  ego-center nodes sampled per forward pass\n")
        f.write("Kept edges: induced subgraph of {seeds} ∪ {1-hop neighbors}\n")
        f.write("            (includes inter-neighbor edges / triangles)\n")
        f.write("\n")
        f.write("Key difference from GraphSAINT-RW (logs/GraphSAINT/):\n")
        f.write("  GraphSAINT-RW: node set from random walk paths (can wander far).\n")
        f.write("  EgoNet:        node set from 1-hop expansion of seeds (stays local).\n")
        f.write("  Both keep full induced subgraph; NeighborSampler does not.\n")
        f.write("\nIndividual logs: logs/EgoNet/{dataset}_{method}.log\n")

    print(f"\nSummary written → {path}")
    return path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\nGLASS EgoNet Ego-Graph Experiments")
    print(f"Dataset:  {DATASET}  |  Feature: {FEATURE_FLAG}")
    print(f"Seeds:    {[s_label(s) for s in S_VALUES]}")
    print(f"Repeat:   {REPEAT}  |  Device: {DEVICE}")

    all_results = {}
    total_start = time.time()

    for s in S_VALUES:
        label    = s_label(s)
        existing = log_path(s)
        if os.path.exists(existing) and os.path.getsize(existing) > 100:
            print(f"\n⊙ {label}: skipping — log exists at {existing}")
            all_results[s] = parse_existing_log(existing)
            continue

        result = run_experiment(s)
        all_results[s] = result

        if result["avg_f1"] is not None:
            edge_info = (f"  edges_kept={result['edge_pct']:.1f}%"
                         if result["edge_pct"] is not None else "")
            print(f"\n✓ {label}: F1={result['avg_f1']:.4f} ± {result['std_err']:.4f}"
                  f"  time={result['train_time']:.1f}s/epoch{edge_info}")
        else:
            print(f"\n✗ {label}: FAILED — check {log_path(s)}")

        write_summary(all_results)

    total = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"Done. Total time: {total/60:.1f} min")
    print(f"Summary: {summary_path()}")
    print(f"{'='*65}\n")
    print(open(summary_path()).read())


if __name__ == "__main__":
    main()
