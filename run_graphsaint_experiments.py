"""
run_graphsaint_experiments.py

Runs GLASS with GraphSAINT random walk sampling across a specified dataset
and multiple num_roots values. Logs results to logs/GraphSAINT/ with the
same format as logs/Sampling/ (NeighborSampler) for direct comparison.

Algorithm (per forward pass):
    1. Sample num_roots root nodes uniformly at random.
    2. From each root, take a random walk of walk_len steps.
    3. Keep ALL edges in the induced subgraph of visited nodes.

Difference from NeighborSampler:
    NeighborSampler keeps exactly k outgoing edges per source node.
    GraphSAINT keeps all edges within a random-walk-reachable node set
    (including inter-neighbor edges / triangles).

Usage:
    python run_graphsaint_experiments.py --dataset density
    python run_graphsaint_experiments.py --dataset coreness
    python run_graphsaint_experiments.py --dataset cut_ratio
    python run_graphsaint_experiments.py --dataset component

Output:
    logs/GraphSAINT/
        {dataset}_baseline.log
        {dataset}_r100.log      ← num_roots=100, walk_len=2
        {dataset}_r200.log
        {dataset}_r500.log
        {dataset}_r1000.log
        {dataset}_r2000.log
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

# num_roots values — None always runs first as baseline (no sampling)
DEFAULT_ROOTS = [100, 200, 500, 1000, 2000]
WALK_LEN      = 2     # fixed walk length; change with --walk_len

REPEAT = 3
DEVICE = -1

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Run GLASS GraphSAINT random walk sampling experiments."
)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--repeat",  type=int, default=REPEAT)
parser.add_argument("--device",  type=int, default=DEVICE)
parser.add_argument("--walk_len", type=int, default=WALK_LEN,
                    help=f"Random walk length (default: {WALK_LEN})")
parser.add_argument("--roots", type=int, nargs="+", default=None,
                    help="num_roots values to test. Baseline always runs first.")
args = parser.parse_args()

DATASET   = args.dataset
REPEAT    = args.repeat
DEVICE    = args.device
WALK_LEN  = args.walk_len
R_LIST    = args.roots if args.roots is not None else DEFAULT_ROOTS
R_VALUES  = [None] + R_LIST   # None = baseline

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

LOG_DIR = "logs/GraphSAINT"
os.makedirs(LOG_DIR, exist_ok=True)


def r_label(r):
    return "baseline" if r is None else f"r{r}"


def log_path(r):
    return f"{LOG_DIR}/{DATASET}_{r_label(r)}.log"


def summary_path():
    return f"{LOG_DIR}/{DATASET}_summary.txt"

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

def build_command(r):
    cmd = [
        "python", "GLASSTest.py",
        FEATURE_FLAG,
        "--use_seed",
        "--use_maxzeroone",
        "--repeat", str(REPEAT),
        "--device", str(DEVICE),
        "--dataset", DATASET,
    ]
    if r is not None:
        cmd += ["--sampler", "graphsaint",
                "--num_roots", str(r),
                "--walk_len",  str(WALK_LEN)]
    return cmd

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment(r):
    label    = r_label(r)
    cmd      = build_command(r)
    log_file = log_path(r)

    print(f"\n{'='*65}")
    print(f"  Dataset: {DATASET}   Sampler: GraphSAINT-RW   Roots: {label}   WalkLen: {WALK_LEN}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log:     {log_file}")
    print(f"{'='*65}")

    start = time.time()
    with open(log_file, "w") as f:
        f.write(f"{'='*65}\n")
        f.write(f"GLASS GraphSAINT Random Walk Experiment\n")
        f.write(f"Dataset:   {DATASET}\n")
        f.write(f"Sampler:   {label}\n")
        f.write(f"num_roots: {r if r is not None else 'N/A (full adj, no sampling)'}\n")
        f.write(f"walk_len:  {WALK_LEN if r is not None else 'N/A'}\n")
        f.write(f"Repeat:    {REPEAT}\n")
        f.write(f"Device:    {DEVICE}\n")
        f.write(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command:   {' '.join(cmd)}\n")
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
        f.write(f"GLASS GraphSAINT Random Walk — Results Summary\n")
        f.write(f"{'='*65}\n")
        f.write(f"Dataset:   {DATASET}\n")
        f.write(f"Feature:   {FEATURE_FLAG}\n")
        f.write(f"walk_len:  {WALK_LEN}\n")
        f.write(f"Repeat:    {REPEAT}\n")
        f.write(f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*65}\n\n")

        f.write("[ Accuracy & Timing ]\n\n")
        h = f"{'Method':<22} {'Avg F1':>8} {'±Err':>7} {'Time/epoch':>12} {'Wall Time':>11}"
        f.write(h + "\n")
        f.write("-" * len(h) + "\n")
        for r, res in all_results.items():
            label = r_label(r)
            avg_f1     = f"{res['avg_f1']:.4f}"      if res['avg_f1']     is not None else "FAILED"
            std_err    = f"{res['std_err']:.4f}"     if res['std_err']    is not None else "N/A"
            train_time = f"{res['train_time']:.1f}s" if res['train_time'] is not None else "N/A"
            wall_time  = f"{res['wall_time']:.1f}s"
            f.write(f"{'GLASS-RW ('+label+')':<22} {avg_f1:>8} {std_err:>7} "
                    f"{train_time:>12} {wall_time:>11}\n")

        f.write("\n\n[ Edge Coverage — Memory Impact of Sampling ]\n\n")
        h2 = f"{'Method':<22} {'Edges Kept':>12} {'Total Edges':>13} {'% Kept':>8} {'% Reduced':>11}"
        f.write(h2 + "\n")
        f.write("-" * len(h2) + "\n")
        for r, res in all_results.items():
            label = r_label(r)
            if r is None:
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
            f.write(f"{'GLASS-RW ('+label+')':<22} {edges_kept:>12} {edges_total:>13} "
                    f"{pct_kept:>8} {pct_reduced:>11}\n")

        f.write("\n\n[ Notes ]\n\n")
        f.write("Sampler:    GraphSAINT random walk (Zeng et al., 2020)\n")
        f.write("num_roots:  random walk starting nodes per forward pass\n")
        f.write(f"walk_len:   {WALK_LEN} (steps per walk)\n")
        f.write("Kept edges: induced subgraph of visited nodes\n")
        f.write("            (includes inter-neighbor edges/triangles)\n")
        f.write("\n")
        f.write("Key difference from NeighborSampler (logs/Sampling/):\n")
        f.write("  NeighborSampler: per-node, k outgoing edges, no induced constraint.\n")
        f.write("  GraphSAINT-RW:   node set via walks, then full induced subgraph.\n")
        f.write("  EgoNet:          node set via 1-hop expansion, then full induced.\n")
        f.write("\nIndividual logs: logs/GraphSAINT/{dataset}_{method}.log\n")

    print(f"\nSummary written → {path}")
    return path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\nGLASS GraphSAINT Random Walk Experiments")
    print(f"Dataset:  {DATASET}  |  Feature: {FEATURE_FLAG}")
    print(f"Roots:    {[r_label(r) for r in R_VALUES]}  |  WalkLen: {WALK_LEN}")
    print(f"Repeat:   {REPEAT}  |  Device: {DEVICE}")

    all_results = {}
    total_start = time.time()

    for r in R_VALUES:
        label    = r_label(r)
        existing = log_path(r)
        if os.path.exists(existing) and os.path.getsize(existing) > 100:
            print(f"\n⊙ {label}: skipping — log exists at {existing}")
            all_results[r] = parse_existing_log(existing)
            continue

        result = run_experiment(r)
        all_results[r] = result

        if result["avg_f1"] is not None:
            edge_info = (f"  edges_kept={result['edge_pct']:.1f}%"
                         if result["edge_pct"] is not None else "")
            print(f"\n✓ {label}: F1={result['avg_f1']:.4f} ± {result['std_err']:.4f}"
                  f"  time={result['train_time']:.1f}s/epoch{edge_info}")
        else:
            print(f"\n✗ {label}: FAILED — check {log_path(r)}")

        write_summary(all_results)

    total = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"Done. Total time: {total/60:.1f} min")
    print(f"Summary: {summary_path()}")
    print(f"{'='*65}\n")
    print(open(summary_path()).read())


if __name__ == "__main__":
    main()
