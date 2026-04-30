"""
run_neighbor_sampling_experiments.py

Runs GLASS with neighbor sampling across a specified dataset and multiple k values.
Logs results to logs/ directory with structured filenames.

Usage:
    # Synthetic datasets:
    python run_neighbor_sampling_experiments.py --dataset density
    python run_neighbor_sampling_experiments.py --dataset cut_ratio
    python run_neighbor_sampling_experiments.py --dataset coreness
    python run_neighbor_sampling_experiments.py --dataset component

    # Real-world datasets:
    python run_neighbor_sampling_experiments.py --dataset ppi_bp
    python run_neighbor_sampling_experiments.py --dataset hpo_metab
    python run_neighbor_sampling_experiments.py --dataset hpo_neuro
    python run_neighbor_sampling_experiments.py --dataset em_user

Output per dataset:
    logs/Sampling/
        {dataset}_baseline.log
        {dataset}_k5.log
        {dataset}_k10.log
        {dataset}_k25.log
        {dataset}_summary.txt    ← clean table for thesis

IMPORTANT — before running:
    1. Add this to NeighborSampler._sample_and_renorm in impl/samplers.py,
       right after the line: keep_mask = ranks < self.k

        if not hasattr(self, '_logged_coverage'):
            self._logged_coverage = True
            kept  = keep_mask.sum().item()
            total = num_edges
            print(f"[EdgeCoverage] kept={kept}/{total} "
                  f"({kept/total*100:.1f}% of edges) "
                  f"reduction={1 - kept/total:.3f}")

    2. Make sure aggr=aggr is passed in buildModel in GLASSTest.py:
        conv = models.EmbZGConv(..., aggr=aggr, num_neighbors=num_neighbors, ...)
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

# k values — None is always prepended automatically as the baseline
DEFAULT_K_VALUES = [40, 20, 10, 5, 2]

REPEAT = 3
DEVICE = -1

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Run GLASS neighbor sampling experiments across k values."
)
parser.add_argument(
    "--dataset", type=str, required=True,
    help="Dataset: density, cut_ratio, coreness, component, "
         "ppi_bp, hpo_metab, hpo_neuro, em_user"
)
parser.add_argument(
    "--repeat", type=int, default=REPEAT,
    help=f"Repeats per experiment (default: {REPEAT})"
)
parser.add_argument(
    "--device", type=int, default=DEVICE,
    help="GPU device id, -1 for CPU (default: 0)"
)
parser.add_argument(
    "--k_values", type=int, nargs="+", default=None,
    help="k values to test, e.g. --k_values 5 10 25. "
         "Baseline (no sampling) always runs first automatically."
)
args = parser.parse_args()

DATASET = args.dataset
REPEAT  = args.repeat
DEVICE  = args.device
K_LIST  = args.k_values if args.k_values is not None else DEFAULT_K_VALUES
# None = baseline always runs first
K_VALUES = [None] + K_LIST

# ---------------------------------------------------------------------------
# Validate dataset
# ---------------------------------------------------------------------------

ALL_DATASETS = SYNTHETIC_DATASETS + REALWORLD_DATASETS
if DATASET not in ALL_DATASETS:
    print(f"ERROR: Unknown dataset '{DATASET}'")
    print(f"Valid: {ALL_DATASETS}")
    sys.exit(1)

FEATURE_FLAG = "--use_one" if DATASET in SYNTHETIC_DATASETS else "--use_nodeid"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOG_DIR = "logs/Sampling"
os.makedirs(LOG_DIR, exist_ok=True)

def k_label(k):
    return "baseline" if k is None else f"k{k}"

def log_path(k):
    return f"{LOG_DIR}/{DATASET}_{k_label(k)}.log"

def summary_path():
    return f"{LOG_DIR}/{DATASET}_summary.txt"

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

def build_command(k):
    cmd = [
        "python", "GLASSTest.py",
        FEATURE_FLAG,
        "--use_seed",
        "--use_maxzeroone",
        "--repeat", str(REPEAT),
        "--device", str(DEVICE),
        "--dataset", DATASET,
    ]
    if k is not None:
        cmd += ["--num_neighbors", str(k)]
    return cmd

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_experiment(k):
    label    = k_label(k)
    cmd      = build_command(k)
    log_file = log_path(k)

    print(f"\n{'='*65}")
    print(f"  Dataset: {DATASET}   Sampler: {label}   Repeat: {REPEAT}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log:     {log_file}")
    print(f"{'='*65}")

    start = time.time()

    with open(log_file, "w") as f:
        f.write(f"{'='*65}\n")
        f.write(f"GLASS Neighbor Sampling Experiment\n")
        f.write(f"Dataset:   {DATASET}\n")
        f.write(f"Sampler:   {label}\n")
        f.write(f"k:         {k if k is not None else 'N/A (full adj, no sampling)'}\n")
        f.write(f"Repeat:    {REPEAT}\n")
        f.write(f"Device:    {DEVICE}\n")
        f.write(f"Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command:   {' '.join(cmd)}\n")
        f.write(f"{'='*65}\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
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
# Parse results
# ---------------------------------------------------------------------------

def parse_results(lines, wall_time, returncode):
    result = {
        "avg_f1":      None,
        "std_err":     None,
        "train_time":  None,   # averaged across repeats
        "wall_time":   wall_time,
        "returncode":  returncode,
        # Edge coverage — from [EdgeCoverage] print in samplers.py
        "edges_kept":  None,
        "edges_total": None,
        "edge_pct":    None,   # percentage of edges kept
        "edge_reduction": None, # fraction dropped
    }

    train_time_sum   = 0.0
    train_time_count = 0

    for line in lines:
        # average 0.921 error 0.034
        m = re.search(r"average\s+([\d.]+)\s+error\s+([\d.]+)", line)
        if m:
            result["avg_f1"]  = float(m.group(1))
            result["std_err"] = float(m.group(2))

        # end: epoch 51, train time 33.77 s, val 0.984, tst 0.873
        m = re.search(
            r"end: epoch\s+\d+,\s+train time\s+([\d.]+)\s+s",
            line
        )
        if m:
            train_time_sum   += float(m.group(1))
            train_time_count += 1

        # [EdgeCoverage] kept=25000/59924 (41.7% of edges) reduction=0.583
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

    # Baseline: 100% edges kept by definition
    return result

def parse_existing_log(path):
    """Re-parse a previously written log file so its results appear in the summary."""
    with open(path) as f:
        lines = f.readlines()

    wall_time = 0.0
    returncode = 0
    for line in lines:
        m = re.search(r"Wall time:\s*([\d.]+)s", line)
        if m:
            wall_time = float(m.group(1))
        m = re.search(r"Exit code:\s*(-?\d+)", line)
        if m:
            returncode = int(m.group(1))

    return parse_results(lines, wall_time, returncode)

# ---------------------------------------------------------------------------
# Write summary
# ---------------------------------------------------------------------------

def write_summary(all_results):
    path = summary_path()

    with open(path, "w") as f:
        f.write(f"GLASS Neighbor Sampling — Results Summary\n")
        f.write(f"{'='*65}\n")
        f.write(f"Dataset:  {DATASET}\n")
        f.write(f"Feature:  {FEATURE_FLAG}\n")
        f.write(f"Repeat:   {REPEAT}\n")
        f.write(f"Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*65}\n\n")

        # --- Accuracy and timing table ---
        f.write("[ Accuracy & Timing ]\n\n")
        h = f"{'Method':<18} {'Avg F1':>8} {'±Err':>7} {'Time/epoch':>12} {'Wall Time':>11}"
        f.write(h + "\n")
        f.write("-" * len(h) + "\n")

        for k, r in all_results.items():
            label      = k_label(k)
            avg_f1     = f"{r['avg_f1']:.4f}"     if r['avg_f1']     is not None else "FAILED"
            std_err    = f"{r['std_err']:.4f}"    if r['std_err']    is not None else "N/A"
            train_time = f"{r['train_time']:.1f}s" if r['train_time'] is not None else "N/A"
            wall_time  = f"{r['wall_time']:.1f}s"
            f.write(f"{'GLASS ('+label+')':<18} {avg_f1:>8} {std_err:>7} "
                    f"{train_time:>12} {wall_time:>11}\n")

        # --- Edge coverage table ---
        f.write("\n\n[ Edge Coverage — Memory Impact of Sampling ]\n\n")
        h2 = f"{'Method':<18} {'Edges Kept':>12} {'Total Edges':>13} {'% Kept':>8} {'% Reduced':>11}"
        f.write(h2 + "\n")
        f.write("-" * len(h2) + "\n")

        total_edges = None
        for k, r in all_results.items():
            label = k_label(k)

            if k is None:
                # Baseline — all edges used
                # Get total edges from first sampling run that has edges_total
                te = next(
                    (v["edges_total"] for v in all_results.values()
                     if v["edges_total"] is not None), None
                )
                if te is not None:
                    total_edges = te
                    edges_kept  = f"{te:,}"
                    edges_total = f"{te:,}"
                    pct_kept    = "100.0%"
                    pct_reduced = "0.0%"
                else:
                    edges_kept = edges_total = pct_kept = pct_reduced = "N/A"
            else:
                if r["edges_kept"] is not None:
                    edges_kept  = f"{r['edges_kept']:,}"
                    edges_total = f"{r['edges_total']:,}"
                    pct_kept    = f"{r['edge_pct']:.1f}%"
                    pct_reduced = f"{r['edge_reduction']*100:.1f}%"
                else:
                    edges_kept = edges_total = pct_kept = pct_reduced = "N/A"

            f.write(f"{'GLASS ('+label+')':<18} {edges_kept:>12} {edges_total:>13} "
                    f"{pct_kept:>8} {pct_reduced:>11}\n")

        # --- Notes ---
        f.write("\n\n[ Notes ]\n\n")
        f.write("Avg F1:        mean micro-F1 across repeats\n")
        f.write("±Err:          standard error across repeats\n")
        f.write("Time/epoch:    average train time per epoch across repeats\n")
        f.write("Wall Time:     total clock time for all repeats\n")
        f.write("Edges Kept:    edges in forward pass after sampling (per layer, per batch)\n")
        f.write("% Reduced:     fraction of edges dropped — directly proportional to\n")
        f.write("               GPU memory savings in a SALIENT-style implementation\n")
        f.write("\n")
        f.write("Key thesis point:\n")
        f.write("  Current sampling is SLOWER due to Python-level argsort overhead.\n")
        f.write("  However, edge reduction shows the MEMORY BENEFIT of sampling.\n")
        f.write("  A SALIENT-style C++ sampler would eliminate the overhead while\n")
        f.write("  retaining the memory reduction — enabling GPU training on large\n")
        f.write("  graphs like Elliptic2 (196M edges) where full adj exceeds GPU RAM.\n")
        f.write("\n")
        f.write("Individual logs: logs/{dataset}_{method}.log\n")

    print(f"\nSummary written → {path}")
    return path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\nGLASS Neighbor Sampling Experiments")
    print(f"Dataset:  {DATASET}  |  Feature: {FEATURE_FLAG}")
    print(f"K values: {[k_label(k) for k in K_VALUES]}")
    print(f"Repeat:   {REPEAT}  |  Device: {DEVICE}")

    all_results  = {}
    total_start  = time.time()

    for k in K_VALUES:
        label = k_label(k)
        existing = log_path(k)
        if os.path.exists(existing) and os.path.getsize(existing) > 100:
            print(f"\n⊙ {label}: skipping — log already exists at {existing}")
            result = parse_existing_log(existing)
            all_results[k] = result
            continue

        result = run_experiment(k)
        all_results[k] = result

        if result["avg_f1"] is not None:
            edge_info = (f"  edges_kept={result['edge_pct']:.1f}%"
                         if result["edge_pct"] is not None else "")
            print(f"\n✓ {label}: F1={result['avg_f1']:.4f} ± {result['std_err']:.4f}"
                  f"  time={result['train_time']:.1f}s/epoch{edge_info}")
        else:
            print(f"\n✗ {label}: FAILED — check {log_path(k)}")

        # Write summary after every run — partial results safe if interrupted
        write_summary(all_results)

    total = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"Done. Total time: {total/60:.1f} min")
    print(f"Summary: {summary_path()}")
    print(f"{'='*65}\n")
    print(open(summary_path()).read())


if __name__ == "__main__":
    main()