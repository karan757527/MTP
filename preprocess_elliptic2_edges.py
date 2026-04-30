"""
preprocess_elliptic2_edges.py — build (E, 95) edge_attr tensor aligned to the
existing preprocessed Elliptic2 edge_index.

WHY THIS SCRIPT EXISTS
----------------------
The original preprocess (preprocess_elliptic2.py) saved edge_weight = ones(E)
and dropped the 95-dim edge features from background_edges.csv. A9 (the novel
edge-features ablation cell) needs those features.

Re-running the full preprocess would take hours and duplicate data. Instead
this script:
  1. Loads the existing processed blob (elliptic2_k2.pt) to reuse its
     edge_index, global_ids, and node counts. No BFS / no induction re-done.
  2. Builds a (src*N + dst) -> position lookup over that edge_index.
  3. Streams background_edges.csv in chunks, mapping each row's clIds to
     (local_src, local_dst) via the same clId -> row -> local chain used
     by the original preprocess.
  4. For each raw directed CSV row, writes its 95-dim features into BOTH
     the (src,dst) slot and the (dst,src) slot of an (E, 95) edge_attr
     tensor (edge_index is undirected so every raw edge has two slots).
  5. z-scores per column and saves edge_attr.pt + stats.

MEMORY
------
edge_attr: 382M * 95 * 2 bytes (float16) ~= 72 GB.
lookup keys + order: 382M * 16 bytes ~= 6.1 GB.
global_to_local map: 49.3M * 8 bytes ~= 0.4 GB.
Peak ~= 80 GB. Host has 118 GB free.

OUTPUT
------
  dataset_/elliptic2/processed/elliptic2_k2_edge_attr.pt

Contents:
  edge_attr       : (E, 95) float16, z-scored per column, aligned to
                    elliptic2_k2.pt's edge_index
  edge_attr_mean  : (95,) float32 — pre-zscore column means
  edge_attr_std   : (95,) float32 — pre-zscore column stds
  edge_index_sha  : sha256 of edge_index bytes for integrity
  n_hit           : number of edge_index slots populated by the CSV
  n_missing       : number of edge_index slots that never got a CSV hit

USAGE
-----
  python preprocess_elliptic2_edges.py
    --kaggle_dir dataset_/elliptic2/kaggle
    --chunksize 10000000
    [--max_rows N]   # smoke test on a CSV prefix
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


N_TOTAL_BG = 49_299_864
N_EDGE_FEATS = 95


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_clid_to_row_lookup(bg_nodes_path: Path):
    """Returns (sorted_clids, order) such that row = order[searchsorted(sorted_clids, q)]."""
    log(f"  reading clId column from {bg_nodes_path.name}...")
    t0 = time.time()
    df = pd.read_csv(bg_nodes_path, usecols=["clId"], dtype={"clId": np.int64})
    clids = df["clId"].to_numpy()
    del df
    assert len(clids) == N_TOTAL_BG, (
        f"background_nodes has {len(clids):,} rows, expected {N_TOTAL_BG:,}"
    )
    log(f"  loaded {len(clids):,} clIds in {time.time()-t0:.1f}s")

    log("  sorting clIds...")
    t0 = time.time()
    order = np.argsort(clids, kind="stable").astype(np.int64)
    sorted_clids = clids[order]
    del clids
    assert (np.diff(sorted_clids) > 0).all(), "non-unique clId"
    log(f"  sorted in {time.time()-t0:.1f}s")
    return sorted_clids, order


def map_clids_to_rows(query, sorted_clids, order):
    """Vectorized clId -> row_in_background. Returns -1 for unknown clIds
    (callers will treat missing as "edge endpoint not in induced set")."""
    idx = np.searchsorted(sorted_clids, query)
    idx_clipped = np.clip(idx, 0, len(sorted_clids) - 1)
    hit = sorted_clids[idx_clipped] == query
    rows = np.where(hit, order[idx_clipped], -1).astype(np.int64)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_dir", type=str,
                    default="dataset_/elliptic2/kaggle",
                    help="Dir holding background_nodes.csv and background_edges.csv")
    ap.add_argument("--processed_dir", type=str,
                    default="dataset_/elliptic2/processed",
                    help="Dir holding the existing elliptic2_k2.pt blob; "
                         "output edge_attr file goes here too.")
    ap.add_argument("--chunksize", type=int, default=10_000_000,
                    help="Rows per CSV chunk. 10M ~ 3.8 GB in memory.")
    ap.add_argument("--max_rows", type=int, default=None,
                    help="Smoke-test: stop after reading this many rows.")
    ap.add_argument("--out_name", type=str,
                    default="elliptic2_k2_edge_attr.pt")
    args = ap.parse_args()

    kaggle_dir = Path(args.kaggle_dir)
    processed_dir = Path(args.processed_dir)
    bg_nodes = kaggle_dir / "background_nodes.csv"
    bg_edges = kaggle_dir / "background_edges.csv"
    existing_blob = processed_dir / "elliptic2_k2.pt"
    out_path = processed_dir / args.out_name

    for p in (bg_nodes, bg_edges, existing_blob):
        if not p.exists():
            raise FileNotFoundError(p)

    # ------------------------------------------------------------------
    # 1. Load existing preprocessed blob -- reuse its edge_index.
    # ------------------------------------------------------------------
    log(f"Step 1: loading existing blob {existing_blob}...")
    t0 = time.time()
    blob = torch.load(existing_blob, map_location="cpu", weights_only=False)
    log(f"  loaded in {time.time()-t0:.1f}s")

    edge_index = blob["edge_index"]                          # (2, E) long
    global_ids = blob["global_ids"]                          # (N_local,) long
    n_local = int(blob["num_nodes"])
    E = int(edge_index.shape[1])
    log(f"  edge_index: {tuple(edge_index.shape)}")
    log(f"  n_local:    {n_local:,}")
    log(f"  E:          {E:,}")

    # Hash edge_index now so we can detect any mismatch between this
    # edge_attr tensor and the blob the driver will pair it with.
    ei_bytes = edge_index.contiguous().numpy().tobytes()
    edge_index_sha = hashlib.sha256(ei_bytes).hexdigest()
    log(f"  edge_index sha256: {edge_index_sha}")
    del blob, ei_bytes

    ei_np = edge_index.numpy()
    src_local_arr = ei_np[0].astype(np.int64, copy=False)
    dst_local_arr = ei_np[1].astype(np.int64, copy=False)
    del ei_np

    # ------------------------------------------------------------------
    # 2. Build (src, dst) -> position lookup over edge_index.
    #    Key = src * n_local + dst  (fits in int64 since n_local < 2^32).
    # ------------------------------------------------------------------
    log("Step 2: building (src,dst) -> position lookup over edge_index...")
    t0 = time.time()
    keys = src_local_arr * np.int64(n_local) + dst_local_arr
    order_e = np.argsort(keys, kind="stable").astype(np.int64)
    sorted_keys = keys[order_e]
    # Sanity: undirected -> no duplicate (src,dst) pairs in edge_index.
    dups = int((np.diff(sorted_keys) == 0).sum())
    if dups > 0:
        log(f"  WARNING: {dups:,} duplicate (src,dst) pairs in edge_index")
    del keys
    log(f"  lookup built in {time.time()-t0:.1f}s "
        f"(sorted_keys {sorted_keys.nbytes/1e9:.1f} GB, "
        f"order {order_e.nbytes/1e9:.1f} GB)")

    # ------------------------------------------------------------------
    # 3. Build clId -> row lookup and global_to_local inverse map.
    # ------------------------------------------------------------------
    log("Step 3: building clId -> row lookup...")
    sorted_clids, cl_order = build_clid_to_row_lookup(bg_nodes)

    log("  building global_to_local inverse map...")
    global_to_local = np.full(N_TOTAL_BG, -1, dtype=np.int64)
    global_to_local[global_ids.numpy().astype(np.int64)] = np.arange(n_local,
                                                                      dtype=np.int64)

    # ------------------------------------------------------------------
    # 4. Allocate output edge_attr tensor (float16 to fit in RAM).
    # ------------------------------------------------------------------
    log(f"Step 4: allocating edge_attr ({E:,} x {N_EDGE_FEATS}) float16 "
        f"~ {E * N_EDGE_FEATS * 2 / 1e9:.1f} GB...")
    edge_attr = np.zeros((E, N_EDGE_FEATS), dtype=np.float16)
    hit_mask = np.zeros(E, dtype=bool)

    # Streaming sums for z-score (keep in float64 for numerical safety).
    col_sum = np.zeros(N_EDGE_FEATS, dtype=np.float64)
    col_sumsq = np.zeros(N_EDGE_FEATS, dtype=np.float64)
    n_hit_directed = 0

    # ------------------------------------------------------------------
    # 5. Stream CSV chunks, map, lookup, scatter.
    # ------------------------------------------------------------------
    log(f"Step 5: streaming {bg_edges.name} in chunks of {args.chunksize:,}...")
    feat_cols = [f"feat#{i}" for i in range(1, N_EDGE_FEATS + 1)]
    # We don't know the exact header names the CSV uses for the 95 features --
    # read the header row once to confirm column ordering.
    header = pd.read_csv(bg_edges, nrows=0).columns.tolist()
    log(f"  CSV header ({len(header)} cols): first 8 = {header[:8]}")
    if "clId1" not in header or "clId2" not in header:
        raise RuntimeError(f"Unexpected CSV header: {header[:6]}...")
    # Feature columns = everything that isn't clId1, clId2, txId.
    non_feat = {"clId1", "clId2", "txId"}
    feat_cols = [c for c in header if c not in non_feat]
    if len(feat_cols) != N_EDGE_FEATS:
        raise RuntimeError(
            f"Expected {N_EDGE_FEATS} feature columns, found {len(feat_cols)}. "
            f"Header: {header}"
        )
    log(f"  feature columns: {feat_cols[0]}..{feat_cols[-1]} ({len(feat_cols)} cols)")

    usecols = ["clId1", "clId2"] + feat_cols
    dtype_map = {"clId1": np.int64, "clId2": np.int64}
    for c in feat_cols:
        dtype_map[c] = np.float32

    overall_t0 = time.time()
    rows_read = 0
    rows_hit = 0
    rows_dropped = 0

    chunk_iter = pd.read_csv(
        bg_edges,
        chunksize=args.chunksize,
        usecols=usecols,
        dtype=dtype_map,
    )

    for chunk_idx, chunk in enumerate(chunk_iter):
        if args.max_rows is not None and rows_read >= args.max_rows:
            break
        ct0 = time.time()
        if args.max_rows is not None:
            remaining = args.max_rows - rows_read
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        clid1 = chunk["clId1"].to_numpy()
        clid2 = chunk["clId2"].to_numpy()
        feats = chunk[feat_cols].to_numpy(dtype=np.float32, copy=False)
        del chunk

        rows_read += len(clid1)

        # clId -> global row
        row1 = map_clids_to_rows(clid1, sorted_clids, cl_order)
        row2 = map_clids_to_rows(clid2, sorted_clids, cl_order)
        del clid1, clid2

        # global row -> local (in induced subset)
        valid = (row1 >= 0) & (row2 >= 0)
        # Every valid row should also remap to a local id >= 0 (induced
        # subgraph includes 94.3% of nodes), but guard anyway.
        local1 = np.where(valid, global_to_local[np.clip(row1, 0, N_TOTAL_BG - 1)], -1)
        local2 = np.where(valid, global_to_local[np.clip(row2, 0, N_TOTAL_BG - 1)], -1)
        in_induced = (local1 >= 0) & (local2 >= 0)
        del row1, row2, valid

        n_in = int(in_induced.sum())
        if n_in == 0:
            rows_dropped += len(local1)
            del local1, local2, feats, in_induced
            continue

        local1 = local1[in_induced]
        local2 = local2[in_induced]
        feats = feats[in_induced]
        del in_induced

        # Forward lookup: (local1, local2) -> position in edge_index.
        fwd_key = local1 * np.int64(n_local) + local2
        rev_key = local2 * np.int64(n_local) + local1

        fwd_idx = np.searchsorted(sorted_keys, fwd_key)
        fwd_idx_c = np.clip(fwd_idx, 0, len(sorted_keys) - 1)
        fwd_hit = sorted_keys[fwd_idx_c] == fwd_key
        fwd_pos = np.where(fwd_hit, order_e[fwd_idx_c], -1)

        rev_idx = np.searchsorted(sorted_keys, rev_key)
        rev_idx_c = np.clip(rev_idx, 0, len(sorted_keys) - 1)
        rev_hit = sorted_keys[rev_idx_c] == rev_key
        rev_pos = np.where(rev_hit, order_e[rev_idx_c], -1)
        del fwd_key, rev_key, fwd_idx, fwd_idx_c, rev_idx, rev_idx_c

        # Scatter into edge_attr. Casting float32 -> float16 is where
        # precision is lost; features are already normalised in the raw
        # CSV, so the half-precision range is more than adequate.
        feats_f16 = feats.astype(np.float16)
        any_hit = fwd_hit | rev_hit
        if fwd_hit.any():
            edge_attr[fwd_pos[fwd_hit]] = feats_f16[fwd_hit]
            hit_mask[fwd_pos[fwd_hit]] = True
        if rev_hit.any():
            edge_attr[rev_pos[rev_hit]] = feats_f16[rev_hit]
            hit_mask[rev_pos[rev_hit]] = True

        # Streaming sums for z-score -- use float32 vals before cast,
        # weighted by number of slots this row populated (fwd + rev).
        weight = fwd_hit.astype(np.float64) + rev_hit.astype(np.float64)
        col_sum += (feats.astype(np.float64) * weight[:, None]).sum(axis=0)
        col_sumsq += (feats.astype(np.float64) ** 2 * weight[:, None]).sum(axis=0)
        n_hit_directed += int(weight.sum())
        rows_hit += int(any_hit.sum())
        rows_dropped += int((~any_hit).sum())
        del fwd_pos, rev_pos, fwd_hit, rev_hit, any_hit, local1, local2
        del feats, feats_f16, weight

        ct = time.time() - ct0
        log(f"  chunk {chunk_idx+1}: read={len(fwd_pos) if False else args.chunksize:,} "
            f"total_read={rows_read:,} hit_rows={rows_hit:,} "
            f"hit_slots={int(hit_mask.sum()):,}/{E:,} "
            f"({100*hit_mask.sum()/E:.1f}%) t={ct:.1f}s")

    total_t = time.time() - overall_t0
    log(f"Step 5 DONE in {total_t/60:.1f} min.")
    log(f"  rows_read    = {rows_read:,}")
    log(f"  rows_hit     = {rows_hit:,}  ({100*rows_hit/max(rows_read,1):.1f}%)")
    log(f"  rows_dropped = {rows_dropped:,}")
    log(f"  slots_hit    = {int(hit_mask.sum()):,}/{E:,} "
        f"({100*hit_mask.sum()/E:.1f}%)")

    # ------------------------------------------------------------------
    # 6. z-score.
    # ------------------------------------------------------------------
    log("Step 6: computing per-column mean/std and z-scoring...")
    denom = max(n_hit_directed, 1)
    col_mean = (col_sum / denom).astype(np.float32)
    col_var = (col_sumsq / denom - (col_sum / denom) ** 2)
    col_var = np.maximum(col_var, 1e-8)
    col_std = np.sqrt(col_var).astype(np.float32)
    log(f"  mean range: [{col_mean.min():.4g}, {col_mean.max():.4g}]")
    log(f"  std  range: [{col_std.min():.4g}, {col_std.max():.4g}]")

    # Do z-score in float32 in-place chunks to keep peak memory low.
    CHUNK_E = 20_000_000
    for s in range(0, E, CHUNK_E):
        e = min(s + CHUNK_E, E)
        block = edge_attr[s:e].astype(np.float32)
        block = (block - col_mean) / col_std
        edge_attr[s:e] = block.astype(np.float16)

    # ------------------------------------------------------------------
    # 7. Save.
    # ------------------------------------------------------------------
    log(f"Step 7: saving to {out_path}...")
    out = {
        "edge_attr": torch.from_numpy(edge_attr),
        "edge_attr_mean": torch.from_numpy(col_mean),
        "edge_attr_std": torch.from_numpy(col_std),
        "edge_index_sha256": edge_index_sha,
        "n_slots_hit": int(hit_mask.sum()),
        "n_slots_total": int(E),
        "rows_read": int(rows_read),
        "rows_hit": int(rows_hit),
        "rows_dropped": int(rows_dropped),
    }
    torch.save(out, out_path)
    size_gb = out_path.stat().st_size / 1e9
    log(f"  wrote {out_path} ({size_gb:.1f} GB)")

    stats = {
        "edge_index_sha256": edge_index_sha,
        "E": E,
        "n_local": n_local,
        "n_slots_hit": int(hit_mask.sum()),
        "n_slots_total": int(E),
        "rows_read": int(rows_read),
        "rows_hit": int(rows_hit),
        "rows_dropped": int(rows_dropped),
        "col_mean_min": float(col_mean.min()),
        "col_mean_max": float(col_mean.max()),
        "col_std_min":  float(col_std.min()),
        "col_std_max":  float(col_std.max()),
        "total_time_min": total_t / 60.0,
    }
    stats_path = processed_dir / "elliptic2_k2_edge_attr_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log(f"  wrote {stats_path}")
    log("DONE.")


if __name__ == "__main__":
    main()
