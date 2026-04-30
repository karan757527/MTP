"""
preprocess_elliptic2.py — one-shot preprocessing for GLASS ablation on Elliptic2.

WHAT THIS DOES
--------------
Builds a 2-hop induced subgraph around the 90,745 labeled nodes (the only nodes
that can possibly influence any prediction in a 2-layer GNN), and packages
everything GLASS's BaseGraph needs into a single .pt file that we can carry
back to the training machine.

WHY 2-HOP
---------
A k-layer GNN's receptive field is exactly k hops. For a 2-layer GLASS model,
nodes more than 2 hops from any labeled node can NEVER influence any subgraph
prediction, so excluding them is mathematically lossless for A1 (2 layers).
For the 1-layer variants (A4, A5, A7, A8) the extra hop is wasted compute but
not incorrect. Running the full 49M-node background graph is infeasible
(activations blow past 100 GB even at hidden=16).

CONFIRMED INPUT FORMATS (from inspect_elliptic2.py + inspect_torch.py)
---------------------------------------------------------------------
  raw_emb.pt           : torch.float32  [49_299_864, 43]
                         Pre-normalized background-node features indexed by
                         *row* in background_nodes.csv. Values look like
                         [-20, 170] floats — RevTrack has already done the
                         feature transform.
  node_idx_map.pt      : torch.int64    [90_745]
                         Maps dense_idx ∈ [0, 90744] → row_in_raw_emb.
                         All unique, min=0, max=49_298_585.
  data_df.pkl          : pandas DataFrame (110902, 17)
                         Relevant cols:
                           node_ids_mapped : list[int] in [0, 90744]
                           labels          : int64 {0, 1}   (2578 positives)
                           split           : str {'TRN','VAL','TST'}
  background_nodes.csv : 5.35 GB, 49_299_864 rows, 44 cols
                         First col = 'clId' (int64). Row index in this file
                         is the same as row index in raw_emb.
                         Remaining 43 cols are the *raw integer* features —
                         we don't need these, raw_emb has the normalized form.
  background_edges.csv : 82.88 GB, 196_215_606 rows, 98 cols
                         First 3 cols: 'clId1', 'clId2', 'txId' (all int64)
                         then 95 edge features. Edges are between clIds,
                         NOT between row indices — we have to map clId → row
                         before we can use them for BFS on raw_emb.

PIPELINE
--------
  1. Load RevTrack files (node_idx_map, data_df).
  2. Load only the clId column of background_nodes.csv → build
     clId→row lookup (via argsort + searchsorted; ~800 MB).
  3. Stream background_edges.csv (usecols=clId1,clId2 only),
     map clId1/clId2 → row1/row2. Output: int32 arrays of length 196M.
  4. Undirect (append reversed edges).
  5. Build CSR (rowptr + col) sorted by src.
  6. BFS k=2 hops from the 90,745 seeds (node_idx_map values).
  7. Induce edges among the visited subset; remap to local IDs.
  8. Load raw_emb, slice to induced subset, free.
  9. Build pos (padded subgraph membership): for each data_df row,
     remap node_ids_mapped → global rows → k2-local.
 10. Build y (labels) and mask (split).
 11. Save .pt + .json.

MEMORY FOOTPRINT (approx peak on the processing machine)
--------------------------------------------------------
  clId lookup:  49.3M × 8  + 49.3M × 8           ≈ 0.8 GB
  edges raw:    196M × 2 × 8                     ≈ 3.1 GB  (pandas int64)
  edges mapped: 196M × 2 × 4                     ≈ 1.6 GB  (int32)
  undirected:   392M × 2 × 4                     ≈ 3.1 GB
  CSR col:      392M × 4                         ≈ 1.6 GB
  CSR rowptr:   49.3M × 8                        ≈ 0.4 GB
  visited:      49.3M × 1                        ≈ 0.05 GB
  raw_emb:      49.3M × 43 × 4  (transient)      ≈ 8.5 GB
  ------
  peak ≈ 20-25 GB comfortably (host has 76 GB available)

USAGE
-----
  python preprocess_elliptic2.py

Paths are hardcoded in the CONFIG block below, matching the actual locations
on the processing machine. Edit if they move. Expected runtime: ~15-30 min,
dominated by (a) the edges CSV read and (b) the argsort step in CSR build.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


# ============================================================================
# CONFIG — paths on the processing machine
# ============================================================================
KAGGLE_DIR = Path("/lfs/usrhome/mtech/cs24m021/scratch/job1727931/extracted")
RAW_DIR    = Path("/lfs/usrhome/mtech/cs24m021/raw_local")
OUT_DIR    = Path("/lfs/usrhome/mtech/cs24m021/elliptic2_processed")

BG_NODES = KAGGLE_DIR / "background_nodes.csv"
BG_EDGES = KAGGLE_DIR / "background_edges.csv"
RAW_EMB      = RAW_DIR / "raw_emb.pt"
NODE_IDX_MAP = RAW_DIR / "node_idx_map.pt"
DATA_DF      = RAW_DIR / "data_df.pkl"

K_HOPS = 2

# Known from the inspection probe — used as an assertion target.
N_TOTAL_BG = 49_299_864


# ============================================================================
# Helpers
# ============================================================================
def log(msg: str) -> None:
    """Timestamped stdout print, flushed so progress shows live over SSH."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gather_neighbors_vectorized(frontier: np.ndarray,
                                rowptr: np.ndarray,
                                col: np.ndarray) -> np.ndarray:
    """
    For each u in `frontier`, return all neighbors col[rowptr[u]:rowptr[u+1]],
    concatenated. Fully vectorized — no Python loop over nodes.

    Trick: build a 'repeat-and-offset' index into col.
      For node u with degree d_u starting at rowptr[u], we want col indices
      rowptr[u], rowptr[u]+1, ..., rowptr[u]+d_u-1. We construct them by
      repeating each rowptr[u] d_u times and adding a per-node local arange.
    """
    starts = rowptr[frontier]
    ends   = rowptr[frontier + 1]
    lens   = ends - starts
    total  = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=col.dtype)
    # Offset of each node's slice within the concatenated output.
    offsets = np.cumsum(lens) - lens                      # shape [len(frontier)]
    rep_starts = np.repeat(starts, lens)                  # shape [total]
    local_idx  = np.arange(total, dtype=np.int64) \
               - np.repeat(offsets, lens)                 # shape [total]
    return col[rep_starts + local_idx]


def build_clid_to_row_lookup(bg_nodes_path: Path):
    """
    Reads ONLY the clId column from background_nodes.csv (≈400 MB in memory)
    and returns (sorted_clids, order) such that:

        row_of(q) = order[np.searchsorted(sorted_clids, q)]

    We use searchsorted instead of a Python dict because a dict over 49M
    int64 keys would eat ~5 GB and be much slower to build.
    """
    log("  reading clId column from background_nodes.csv…")
    t0 = time.time()
    # usecols=['clId'] + c engine → pandas skips every other column fast.
    df = pd.read_csv(bg_nodes_path, usecols=["clId"], dtype={"clId": np.int64})
    clids = df["clId"].to_numpy()
    del df
    assert len(clids) == N_TOTAL_BG, (
        f"background_nodes has {len(clids):,} rows, "
        f"expected {N_TOTAL_BG:,}"
    )
    log(f"  loaded {len(clids):,} clIds in {time.time()-t0:.1f}s")

    log("  sorting clIds for searchsorted lookup…")
    t0 = time.time()
    order = np.argsort(clids, kind="stable").astype(np.int64)
    sorted_clids = clids[order]
    del clids
    # Enforce uniqueness — if two rows share the same clId, the lookup is
    # ambiguous and downstream edge mapping is silently wrong.
    assert (np.diff(sorted_clids) > 0).all(), (
        "Non-unique clId in background_nodes.csv — cannot build lookup."
    )
    log(f"  sorted in {time.time()-t0:.1f}s "
        f"(clId range [{int(sorted_clids[0])}, {int(sorted_clids[-1])}])")
    return sorted_clids, order


def map_clids_to_rows(query: np.ndarray,
                      sorted_clids: np.ndarray,
                      order: np.ndarray) -> np.ndarray:
    """
    Map an array of clId values to their row indices in background_nodes.
    Vectorized via numpy searchsorted. Runs at ~50M lookups/sec.
    Raises if any clId is missing.
    """
    idx = np.searchsorted(sorted_clids, query)
    # Bounds check + exact match check catches any clId that isn't in nodes.
    if (idx >= len(sorted_clids)).any() or (sorted_clids[idx] != query).any():
        bad = query[(idx >= len(sorted_clids)) |
                    (sorted_clids[np.clip(idx, 0, len(sorted_clids) - 1)]
                     != query)]
        raise RuntimeError(
            f"{len(bad):,} edge endpoints reference clIds not present in "
            f"background_nodes. First few: {bad[:5].tolist()}"
        )
    return order[idx]


# ============================================================================
# Main
# ============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load RevTrack files
    # ------------------------------------------------------------------
    log("Step 1: loading RevTrack files…")
    node_idx_map = torch.load(NODE_IDX_MAP, map_location="cpu",
                              weights_only=False)
    if not torch.is_tensor(node_idx_map):
        node_idx_map = torch.as_tensor(node_idx_map)
    node_idx_map = node_idx_map.long()
    log(f"  node_idx_map: shape={tuple(node_idx_map.shape)}, "
        f"range=[{int(node_idx_map.min())}, {int(node_idx_map.max())}]")
    assert int(node_idx_map.max()) < N_TOTAL_BG

    data_df = pd.read_pickle(DATA_DF)
    log(f"  data_df: shape={data_df.shape}, "
        f"splits={sorted(data_df['split'].unique().tolist())}")
    required = {"node_ids_mapped", "labels", "split"}
    missing = required - set(data_df.columns)
    assert not missing, f"data_df missing required columns: {missing}"

    # ------------------------------------------------------------------
    # Step 2 — Build clId → row_in_background lookup
    # ------------------------------------------------------------------
    log("Step 2: building clId → row lookup from background_nodes.csv…")
    sorted_clids, order = build_clid_to_row_lookup(BG_NODES)

    # ------------------------------------------------------------------
    # Step 3 — Load edges (clId1, clId2 only) and map to row indices
    # ------------------------------------------------------------------
    log("Step 3: loading background_edges.csv (clId1, clId2 only)…")
    t0 = time.time()
    df_edges = pd.read_csv(
        BG_EDGES,
        usecols=["clId1", "clId2"],
        dtype={"clId1": np.int64, "clId2": np.int64},
    )
    log(f"  loaded {len(df_edges):,} edges in {time.time()-t0:.1f}s")
    src_cl = df_edges["clId1"].to_numpy()
    dst_cl = df_edges["clId2"].to_numpy()
    del df_edges

    log("  mapping clId1 → row…")
    t0 = time.time()
    src_row = map_clids_to_rows(src_cl, sorted_clids, order).astype(np.int32)
    del src_cl
    log(f"    done in {time.time()-t0:.1f}s")

    log("  mapping clId2 → row…")
    t0 = time.time()
    dst_row = map_clids_to_rows(dst_cl, sorted_clids, order).astype(np.int32)
    del dst_cl, sorted_clids, order
    log(f"    done in {time.time()-t0:.1f}s")
    log(f"  row range: src[{src_row.min()},{src_row.max()}] "
        f"dst[{dst_row.min()},{dst_row.max()}]")

    # ------------------------------------------------------------------
    # Step 4 — Undirect
    # ------------------------------------------------------------------
    log("Step 4: making edges undirected…")
    src_u = np.concatenate([src_row, dst_row])
    dst_u = np.concatenate([dst_row, src_row])
    del src_row, dst_row
    log(f"  undirected edge count: {len(src_u):,}")

    # ------------------------------------------------------------------
    # Step 5 — Build CSR (rowptr + col)
    # ------------------------------------------------------------------
    log("Step 5: building CSR adjacency…")
    t0 = time.time()
    # argsort on 392M int32 — slow but feasible (~1-3 min).
    order_e = np.argsort(src_u, kind="stable")
    src_sorted = src_u[order_e]
    dst_sorted = dst_u[order_e]
    del order_e, src_u, dst_u
    log(f"  sorted edges in {time.time()-t0:.1f}s")

    deg = np.bincount(src_sorted, minlength=N_TOTAL_BG).astype(np.int64)
    rowptr = np.empty(N_TOTAL_BG + 1, dtype=np.int64)
    rowptr[0] = 0
    np.cumsum(deg, out=rowptr[1:])
    col = dst_sorted  # int32
    del src_sorted, dst_sorted, deg
    log(f"  CSR: rowptr[{len(rowptr):,}], col[{len(col):,}]")

    # ------------------------------------------------------------------
    # Step 6 — BFS k=2 from the labeled seeds
    # ------------------------------------------------------------------
    log(f"Step 6: BFS {K_HOPS}-hop from {len(node_idx_map):,} seeds…")
    seeds = node_idx_map.numpy().astype(np.int64)
    visited = np.zeros(N_TOTAL_BG, dtype=bool)
    visited[seeds] = True
    frontier = seeds

    for hop in range(K_HOPS):
        t0 = time.time()
        # frontier values are global row indices (int64); CSR indexing handles
        # either dtype, but rowptr[frontier+1] requires int/long.
        nbrs = gather_neighbors_vectorized(frontier, rowptr, col)
        # nbrs is int32 — cast to int64 for bool-index assignment safety.
        nbrs64 = nbrs.astype(np.int64, copy=False)
        was_visited = visited.copy()
        visited[nbrs64] = True
        new_mask = visited & ~was_visited
        frontier = np.nonzero(new_mask)[0]
        log(f"  hop {hop+1}: +{len(frontier):,} new, "
            f"total visited={int(visited.sum()):,} "
            f"({time.time()-t0:.1f}s)")

    S_global = np.nonzero(visited)[0].astype(np.int64)
    n_local = len(S_global)
    log(f"  induced subset: {n_local:,} nodes "
        f"({100 * n_local / N_TOTAL_BG:.3f}% of full graph)")

    # ------------------------------------------------------------------
    # Step 7 — Induce edges, remap to local IDs
    # ------------------------------------------------------------------
    log("Step 7: inducing edges among visited subset…")
    # Dense global→local map. 49M int64 = 392 MB.
    global_to_local = np.full(N_TOTAL_BG, -1, dtype=np.int64)
    global_to_local[S_global] = np.arange(n_local, dtype=np.int64)

    # Every edge leaving S (we use undirected CSR so this gets both directions).
    dst_out_of_S = gather_neighbors_vectorized(S_global, rowptr, col)
    lens = rowptr[S_global + 1] - rowptr[S_global]
    src_out_of_S = np.repeat(S_global, lens)

    # Keep only edges whose *dst* is also in S.
    keep = visited[dst_out_of_S.astype(np.int64, copy=False)]
    src_k = src_out_of_S[keep]
    dst_k = dst_out_of_S[keep].astype(np.int64, copy=False)
    del dst_out_of_S, src_out_of_S, lens, keep, rowptr, col
    log(f"  induced directed edges: {len(src_k):,}")

    src_local = global_to_local[src_k]
    dst_local = global_to_local[dst_k]
    del src_k, dst_k
    assert (src_local >= 0).all() and (dst_local >= 0).all()

    edge_index = torch.from_numpy(np.stack([src_local, dst_local])).long()
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    del src_local, dst_local
    log(f"  edge_index: {tuple(edge_index.shape)}")

    # ------------------------------------------------------------------
    # Step 8 — Slice raw_emb to the induced subset
    # ------------------------------------------------------------------
    log("Step 8: slicing raw_emb to induced subset…")
    raw_emb = torch.load(RAW_EMB, map_location="cpu", weights_only=False)
    log(f"  raw_emb: shape={tuple(raw_emb.shape)}, dtype={raw_emb.dtype}")
    assert raw_emb.shape[0] == N_TOTAL_BG, (
        f"raw_emb has {raw_emb.shape[0]} rows, expected {N_TOTAL_BG}"
    )
    x_features = raw_emb[torch.from_numpy(S_global).long()].contiguous()
    del raw_emb
    log(f"  x_features: shape={tuple(x_features.shape)}, "
        f"dtype={x_features.dtype}")

    # ------------------------------------------------------------------
    # Step 9 — Build padded pos tensor from data_df.node_ids_mapped
    # ------------------------------------------------------------------
    log("Step 9: building padded subgraph membership tensor…")
    # node_ids_mapped entries are dense_idx ∈ [0, 90744].
    # Chain: dense_idx → global row (via node_idx_map) → k2-local (via g2l).
    node_idx_map_np = node_idx_map.numpy().astype(np.int64)
    subg_nodes_local = []
    for nlist in data_df["node_ids_mapped"]:
        arr = np.asarray(nlist, dtype=np.int64).ravel()
        global_ids = node_idx_map_np[arr]
        local_ids = global_to_local[global_ids]
        # Every seed is in visited by construction, so every subgraph node
        # MUST remap cleanly. This assert catches data corruption / ID
        # mismatch early.
        if (local_ids < 0).any():
            bad = int((local_ids < 0).sum())
            raise RuntimeError(
                f"{bad} subgraph nodes failed to remap — node_idx_map / "
                f"background graph ID spaces do not align."
            )
        subg_nodes_local.append(torch.from_numpy(local_ids).long())

    pos = pad_sequence(subg_nodes_local, batch_first=True, padding_value=-1)
    log(f"  pos: shape={tuple(pos.shape)} "
        f"(num_subgraphs × max_subgraph_size)")

    # ------------------------------------------------------------------
    # Step 10 — y and mask
    # ------------------------------------------------------------------
    log("Step 10: building y and mask…")
    y = torch.as_tensor(
        np.asarray(data_df["labels"].values, dtype=np.float32)
    )
    log(f"  y: shape={tuple(y.shape)}, "
        f"positives={int((y > 0.5).sum())}, "
        f"negatives={int((y <= 0.5).sum())}")

    split_to_int = {"TRN": 0, "VAL": 1, "TST": 2}
    try:
        mask_list = [split_to_int[s] for s in data_df["split"].values]
    except KeyError as e:
        raise RuntimeError(
            f"Unexpected split label: {e}. "
            f"Observed unique: {data_df['split'].unique().tolist()}"
        )
    mask = torch.tensor(mask_list, dtype=torch.long)
    log(f"  mask: train={int((mask == 0).sum())}, "
        f"val={int((mask == 1).sum())}, "
        f"test={int((mask == 2).sum())}")

    # ------------------------------------------------------------------
    # Step 11 — Save outputs
    # ------------------------------------------------------------------
    log("Step 11: saving artifacts…")
    out_dict = {
        # Core tensors GLASS's BaseGraph expects:
        "x_features":  x_features,                       # [n_local, 43] float32
        "edge_index":  edge_index,                       # [2, E]        long
        "edge_weight": edge_weight,                      # [E]           float32
        "pos":         pos,                              # [S, max_size] long, -1 pad
        "y":           y,                                # [S]           float32
        "mask":        mask,                             # [S]           long {0,1,2}
        # Metadata for audit / reproducibility:
        "num_nodes":   int(n_local),
        "k_hops":      int(K_HOPS),
        "global_ids":  torch.from_numpy(S_global).long(),  # k2-local → bg row
    }
    out_pt = OUT_DIR / f"elliptic2_k{K_HOPS}.pt"
    torch.save(out_dict, out_pt)
    log(f"  wrote {out_pt}  ({out_pt.stat().st_size / 1e9:.2f} GB)")

    n_pos = int((y > 0.5).sum())
    n_neg = int((y <= 0.5).sum())
    stats = {
        "k_hops": int(K_HOPS),
        "num_nodes_induced": int(n_local),
        "num_nodes_full_bg": int(N_TOTAL_BG),
        "induced_fraction": n_local / N_TOTAL_BG,
        "num_edges_induced": int(edge_index.shape[1]),
        "feat_dim": int(x_features.shape[1]),
        "num_subgraphs": int(pos.shape[0]),
        "max_subgraph_size": int(pos.shape[1]),
        "num_positives": n_pos,
        "num_negatives": n_neg,
        "pos_weight_suggested": float(n_neg / max(n_pos, 1)),
        "split_train": int((mask == 0).sum()),
        "split_val":   int((mask == 1).sum()),
        "split_test":  int((mask == 2).sum()),
    }
    out_json = OUT_DIR / "elliptic2_stats.json"
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    log(f"  wrote {out_json}")

    log("DONE.")
    log("Summary:")
    for k, v in stats.items():
        log(f"  {k}: {v}")


if __name__ == "__main__":
    main()
