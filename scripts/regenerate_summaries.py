"""
regenerate_summaries.py — regenerate all *_summary.json files into
logs/ablations/summaries/ from logs/ablations/per_seed/.

Format is identical to scripts/aggregate_seeds.py (same KEYS, same statistics
functions, same dict layout, same json.dump indent). Only the input path
differs to match the reorganised log layout.

Usage:
    python scripts/regenerate_summaries.py
    python scripts/regenerate_summaries.py --nseeds 10
    python scripts/regenerate_summaries.py --cells A1 A2 A8
"""
import argparse
import json
import statistics
from pathlib import Path

PER_SEED_ROOT = Path("/data/cs24m035/BPTT/GLASS/logs/ablations/per_seed")
SUMMARY_ROOT = Path("/data/cs24m035/BPTT/GLASS/logs/ablations/summaries")

CELLS = ["A1", "A2", "A4", "A5", "A6", "A7", "A8", "A8b",
         "A9", "A9b", "A9c", "A10", "A11"]

KEYS = [
    "best_val_prauc",
    "best_tst_prauc",
    "best_tst_f1",
    "best_tst_rocauc",
    "best_epoch",
]


def load_final(jsonl_path: Path):
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


def load_timing(jsonl_path: Path):
    """Walk all 'epoch' events and accumulate wall time + epoch count.

    Returns (wall_time_s, n_epochs). wall_time_s = sum of (trn+val+tst)_time_s
    across every epoch record. Epochs without timing fields contribute 0.
    """
    if not jsonl_path.exists():
        return 0.0, 0
    wall = 0.0
    n_ep = 0
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
                if rec.get("event") != "epoch":
                    continue
                n_ep += 1
                wall += float(rec.get("trn_time_s", 0) or 0)
                wall += float(rec.get("val_time_s", 0) or 0)
                wall += float(rec.get("tst_time_s", 0) or 0)
    except OSError:
        return 0.0, 0
    return wall, n_ep


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
    timing_collected = {"wall_time_s": [], "n_epochs": []}
    per_seed = []
    for s in range(nseeds):
        p = PER_SEED_ROOT / f"{exp}_seed{s}" / "train.jsonl"
        rec = load_final(p)
        wall_s, n_ep = load_timing(p)

        if rec is None:
            # Even for failed seeds, still record any partial timing observed.
            per_seed.append({
                "seed": s, "ok": False, "reason": "no final record",
                "wall_time_s": wall_s, "n_epochs": n_ep,
            })
            continue
        per_seed.append({
            "seed": s,
            "ok": True,
            **{k: rec.get(k) for k in KEYS},
            "wall_time_s": wall_s,
            "n_epochs": n_ep,
        })
        for k in KEYS:
            v = rec.get(k)
            if v is not None:
                collected[k].append(v)
        # Aggregate timing only over OK seeds (matches metric aggregation).
        timing_collected["wall_time_s"].append(wall_s)
        timing_collected["n_epochs"].append(n_ep)

    agg = {k: stats(collected[k]) for k in KEYS}
    agg["wall_time_s"] = stats(timing_collected["wall_time_s"])
    agg["n_epochs"] = stats(timing_collected["n_epochs"])
    return {
        "experiment": exp,
        "nseeds_requested": nseeds,
        "nseeds_completed": sum(1 for r in per_seed if r["ok"]),
        "per_seed": per_seed,
        "agg": agg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=7,
                    help="Target seed count to look for (default: 7).")
    ap.add_argument("--cells", nargs="*", default=None,
                    help="Cells to regenerate (default: all 13).")
    args = ap.parse_args()

    cells = args.cells or CELLS
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"=== regenerate_summaries — nseeds={args.nseeds} ===")
    print(f"  reading from: {PER_SEED_ROOT}")
    print(f"  writing to:   {SUMMARY_ROOT}")
    print("")
    for c in cells:
        s = aggregate_one(c, args.nseeds)
        out = SUMMARY_ROOT / f"{c}_summary.json"
        with open(out, "w") as f:
            json.dump(s, f, indent=2)
        agg = s["agg"]["best_tst_prauc"]
        wt = s["agg"]["wall_time_s"]
        if agg["mean"] is None:
            tail = "  no data"
        else:
            wall_min = wt["mean"] / 60 if wt["mean"] is not None else 0
            tail = (f"  best_tst_prauc = {agg['mean']:.4f} ± {agg['std']:.4f}"
                    f"  wall ≈ {wall_min:.1f} min/seed")
        print(f"  {c:5s}  n={s['nseeds_completed']}/{args.nseeds}  → {out.name}{tail}")


if __name__ == "__main__":
    main()
