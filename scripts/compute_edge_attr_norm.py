#!/usr/bin/env python3
"""Build a normalized edge_attr tensor with a presence-mask column for A9c.

Reads:  dataset_/elliptic2/processed/elliptic2_k2_edge_attr.pt   (E, 95) f16
Writes: dataset_/elliptic2/processed/elliptic2_k2_edge_attr_norm.pt (E, 96) f16
        dataset_/elliptic2/processed/elliptic2_k2_edge_attr_normstats.json

Recipe:
  - Per-column mean and std are computed over NON-ZERO rows only (rows where
    any column is non-zero — i.e. edges that have observed transaction
    features).
  - Z-score: (x - mean) / (std + eps), applied to non-zero rows only.
  - Zero rows (~29% of edges) stay as exact zero across the first 95 cols.
  - A 96th column is appended: 1.0 for non-zero rows, 0.0 for zero rows.
    This lets the model distinguish "no info" from "small info".

Why:
  - Released edge_attr is unnormalized: per-col mean spans -0.07 to 82, std
    spans 0.10 to 30. Linear(95, H) gradients are dominated by high-magnitude
    columns; low-magnitude features are drowned out.
  - 29% of edges have all-zero attrs (placeholder for missing info). Without
    a presence mask the model can't tell apart the missing-info case.

Idempotent: skips if output exists. Delete the .pt file to regenerate.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import torch

PROCESSED = Path("/data/cs24m035/BPTT/GLASS/dataset_/elliptic2/processed")
SRC       = PROCESSED / "elliptic2_k2_edge_attr.pt"
DST       = PROCESSED / "elliptic2_k2_edge_attr_norm.pt"
DST_STATS = PROCESSED / "elliptic2_k2_edge_attr_normstats.json"

EPS = 1e-6
CHUNK = 10_000_000   # 10M rows × 95 f16 cols ≈ 1.8 GB per chunk

NEEDED_BYTES = 75 * 1024**3   # ~75 GB headroom for the new file


def main():
    if DST.exists() and DST_STATS.exists():
        print(f"[norm] {DST.name} already exists — skipping. Delete to regenerate.")
        return 0

    free = shutil.disk_usage(PROCESSED).free
    print(f"[norm] disk free at {PROCESSED}: {free / 1e9:.1f} GB")
    if free < NEEDED_BYTES:
        print(f"[norm] ABORT: need ~{NEEDED_BYTES / 1e9:.0f} GB free, have {free / 1e9:.1f} GB")
        return 1

    print(f"[norm] loading {SRC}", flush=True)
    t0 = time.time()
    blob = torch.load(SRC, weights_only=False)
    x = blob["edge_attr"]
    print(f"[norm] loaded in {time.time() - t0:.1f}s shape={tuple(x.shape)} dtype={x.dtype}", flush=True)

    E, F = x.shape

    # ---- pass 1: per-col mean/std over non-zero rows ----
    print(f"[norm] pass 1: per-col stats over non-zero rows (chunk={CHUNK:,})", flush=True)
    t1 = time.time()
    sum_x  = torch.zeros(F, dtype=torch.float64)
    sum_xx = torch.zeros(F, dtype=torch.float64)
    n_nz = 0
    for ci, i in enumerate(range(0, E, CHUNK)):
        j = min(i + CHUNK, E)
        chunk = x[i:j].float()
        nz = chunk.abs().sum(dim=1) > 0
        if nz.any():
            nz_chunk = chunk[nz]
            sum_x  += nz_chunk.sum(dim=0).double()
            sum_xx += (nz_chunk * nz_chunk).sum(dim=0).double()
            n_nz   += int(nz.sum().item())
        if ci % 5 == 0:
            print(f"  pass1 {j:,}/{E:,}  nz_so_far={n_nz:,}  t={time.time() - t1:.0f}s", flush=True)
    mean = (sum_x / n_nz).float()
    var  = (sum_xx / n_nz - (sum_x / n_nz) ** 2).clamp(min=0).float()
    std  = var.sqrt()
    print(f"[norm] pass1 done in {time.time() - t1:.1f}s  n_nonzero={n_nz:,} ({100 * n_nz / E:.2f}%)", flush=True)
    print(f"[norm] mean range: {mean.min().item():.4f} to {mean.max().item():.4f}")
    print(f"[norm] std  range: {std.min().item():.4f} to {std.max().item():.4f}")

    # ---- pass 2: build (E, F+1) normalized + presence-mask tensor ----
    print(f"[norm] pass 2: writing normalized tensor with presence-mask col", flush=True)
    out = torch.zeros(E, F + 1, dtype=torch.float16)
    inv_std = 1.0 / (std + EPS)
    t2 = time.time()
    for ci, i in enumerate(range(0, E, CHUNK)):
        j = min(i + CHUNK, E)
        chunk = x[i:j].float()
        nz = chunk.abs().sum(dim=1) > 0
        normed = torch.zeros_like(chunk)
        if nz.any():
            normed[nz] = (chunk[nz] - mean) * inv_std
        out[i:j, :F] = normed.half()
        out[i:j, F]  = nz.half()
        if ci % 5 == 0:
            print(f"  pass2 {j:,}/{E:,}  t={time.time() - t2:.0f}s", flush=True)
    print(f"[norm] pass2 done in {time.time() - t2:.1f}s", flush=True)

    # ---- save ----
    print(f"[norm] saving {DST}", flush=True)
    t3 = time.time()
    new_blob = {**blob, "edge_attr": out,
                "norm_recipe": "z-score per col over non-zero rows; zero rows kept zero; col F+1 = presence mask"}
    torch.save(new_blob, DST)
    print(f"[norm] saved in {time.time() - t3:.1f}s  size={DST.stat().st_size / 1e9:.2f} GB", flush=True)

    with open(DST_STATS, "w") as f:
        json.dump({
            "src_path": str(SRC),
            "dst_path": str(DST),
            "E": int(E),
            "F_orig": int(F),
            "F_new": int(F + 1),
            "n_nonzero": int(n_nz),
            "n_zero": int(E - n_nz),
            "frac_nonzero": float(n_nz / E),
            "per_col_mean_nz": mean.tolist(),
            "per_col_std_nz": std.tolist(),
            "edge_index_sha256": blob.get("edge_index_sha256"),
            "compute_time_s": float(time.time() - t0),
        }, f, indent=2)
    print(f"[norm] all done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
