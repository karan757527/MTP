"""
aggregate_seeds.py — collapse N-seed runs of one ablation cell into mean / std.

For a given experiment EXP and N seeds, reads each
    logs/<EXP>_seed{0..N-1}/train.jsonl
finds the final 'event=final' record (which contains best_val_prauc,
best_tst_prauc, etc.), computes mean and std across seeds, and writes
both a machine-readable JSON and a human-readable text block.

Usage:
    python scripts/aggregate_seeds.py --exp A8 --nseeds 5 --out logs/A8_summary.json
    python scripts/aggregate_seeds.py --all --nseeds 5 --out logs/all_summary.json

Robust to:
    * missing seed runs (skip that seed, continue)
    * partial / corrupt train.jsonl (skip non-JSON lines)
    * cells with zero successful seeds (records "no data" instead of crashing)
"""
import argparse
import json
import statistics
from pathlib import Path

EXPERIMENTS = ["A4", "A8b", "A7", "A8", "A5", "A2", "A1", "A6"]

KEYS = [
    "best_val_prauc",
    "best_tst_prauc",
    "best_tst_f1",
    "best_tst_rocauc",
    "best_epoch",
]


def load_final(jsonl_path: Path):
    """Return the last 'event=final' record from a JSONL file, or None."""
    if not jsonl_path.exists():
        return None
    last_final = None
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "final":
                    last_final = rec
    except OSError:
        return None
    return last_final


def stats(vals):
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None,
                "n": 0, "vals": []}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "min": vals[0], "max": vals[0],
                "n": 1, "vals": vals}
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals),
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
        "vals": vals,
    }


def aggregate_one(exp: str, nseeds: int) -> dict:
    collected = {k: [] for k in KEYS}
    per_seed = []
    for s in range(nseeds):
        p = Path(f"logs/{exp}_seed{s}/train.jsonl")
        rec = load_final(p)
        if rec is None:
            per_seed.append({"seed": s, "ok": False, "reason": "no final record"})
            continue
        per_seed.append({
            "seed": s,
            "ok": True,
            **{k: rec.get(k) for k in KEYS},
        })
        for k in KEYS:
            v = rec.get(k)
            if v is not None:
                collected[k].append(v)

    return {
        "experiment": exp,
        "nseeds_requested": nseeds,
        "nseeds_completed": sum(1 for r in per_seed if r["ok"]),
        "per_seed": per_seed,
        "agg": {k: stats(collected[k]) for k in KEYS},
    }


def print_one(summary: dict):
    exp = summary["experiment"]
    n_ok = summary["nseeds_completed"]
    n_req = summary["nseeds_requested"]
    print(f"[agg {exp}] n={n_ok}/{n_req} seeds completed")
    for k in KEYS:
        s = summary["agg"][k]
        if s["mean"] is None:
            print(f"  {k:18s}: no data")
        elif k == "best_epoch":
            print(f"  {k:18s}: mean={s['mean']:.1f}  std={s['std']:.1f}  "
                  f"min={s['min']}  max={s['max']}  n={s['n']}")
        else:
            print(f"  {k:18s}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"min={s['min']:.4f}  max={s['max']:.4f}  n={s['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None,
                    help="Single cell to aggregate (e.g. A8). Required unless --all.")
    ap.add_argument("--all", action="store_true",
                    help="Aggregate every cell in the canonical EXPERIMENTS list.")
    ap.add_argument("--nseeds", type=int, default=5,
                    help="Number of seeds expected per cell.")
    ap.add_argument("--out", required=True,
                    help="Output JSON path.")
    args = ap.parse_args()

    if args.all:
        all_summary = {"nseeds_requested": args.nseeds, "cells": {}}
        for exp in EXPERIMENTS:
            s = aggregate_one(exp, args.nseeds)
            all_summary["cells"][exp] = s
            print_one(s)
            print("")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_summary, f, indent=2)
        print(f"[wrote] {args.out}")

        # Print a compact one-table view for the runner_summary log.
        print("")
        print("=== CROSS-CELL TABLE (mean ± std of best_tst_prauc) ===")
        for exp in EXPERIMENTS:
            cell = all_summary["cells"][exp]
            agg = cell["agg"]["best_tst_prauc"]
            n = cell["nseeds_completed"]
            if agg["mean"] is None:
                print(f"  {exp:5s}  n={n}  no data")
            else:
                print(f"  {exp:5s}  n={n}  prauc = {agg['mean']:.4f} ± {agg['std']:.4f}"
                      f"  (range {agg['min']:.4f}..{agg['max']:.4f})")
    else:
        if not args.exp:
            ap.error("--exp is required unless --all is set")
        summary = aggregate_one(args.exp, args.nseeds)
        print_one(summary)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
