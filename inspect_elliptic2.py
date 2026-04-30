"""
inspect_elliptic2.py — read-only diagnostic script.

Prints everything needed to finalize preprocess_elliptic2.py without guessing:
  - background_nodes.csv / background_edges.csv columns, dtypes, row counts, samples
  - raw_emb.pt / node_idx_map.pt / data_df.pkl shapes, dtypes, sample values
  - Processing machine specs (RAM, disk, library versions)

Does NOT load full CSVs — uses chunked/sampled reads only. Safe to run on the
processing machine without worrying about memory. Should take ~1-2 minutes max
(row-counting the 196M-edge CSV is the slowest part).

Just run:  python inspect_elliptic2.py
Paste the entire stdout back.
"""

import os
import sys
import shutil
import platform
from pathlib import Path

# ---- Paths (from the user) ----
KAGGLE_DIR = Path("/lfs/usrhome/mtech/cs24m021/scratch/job1727931/extracted")
RAW_DIR    = Path("/lfs/usrhome/mtech/cs24m021/raw_local")

BG_NODES = KAGGLE_DIR / "background_nodes.csv"
BG_EDGES = KAGGLE_DIR / "background_edges.csv"

RAW_EMB      = RAW_DIR / "raw_emb.pt"
NODE_IDX_MAP = RAW_DIR / "node_idx_map.pt"
DATA_DF      = RAW_DIR / "data_df.pkl"


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def safe(fn, *args, **kwargs):
    """Run a probe; print the exception instead of crashing the whole script."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  !! ERROR: {type(e).__name__}: {e}")
        return None


# ============================================================================
# 0. System + library versions
# ============================================================================
banner("0. SYSTEM")
print(f"  python:   {sys.version.split()[0]}")
print(f"  platform: {platform.platform()}")
try:
    import numpy as np
    print(f"  numpy:    {np.__version__}")
except Exception as e:
    print(f"  numpy:    MISSING ({e})")
try:
    import pandas as pd
    print(f"  pandas:   {pd.__version__}")
except Exception as e:
    print(f"  pandas:   MISSING ({e})")
try:
    import torch
    print(f"  torch:    {torch.__version__}")
except Exception as e:
    print(f"  torch:    MISSING ({e})")

# RAM
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemTotal", "MemAvailable")):
                print(f"  {line.strip()}")
except Exception as e:
    print(f"  meminfo:  could not read ({e})")

# Disk free on the work dirs
for p in [KAGGLE_DIR, RAW_DIR]:
    try:
        usage = shutil.disk_usage(p)
        print(f"  disk@{p}: free={usage.free/1e9:.1f} GB "
              f"total={usage.total/1e9:.1f} GB")
    except Exception as e:
        print(f"  disk@{p}: could not stat ({e})")


# ============================================================================
# 1. background_nodes.csv
# ============================================================================
banner("1. background_nodes.csv")
print(f"  path: {BG_NODES}")
print(f"  exists: {BG_NODES.exists()}")
if BG_NODES.exists():
    print(f"  size:   {BG_NODES.stat().st_size / 1e9:.2f} GB")

    # Read 10 rows with default dtype inference
    def peek_nodes():
        import pandas as pd
        df = pd.read_csv(BG_NODES, nrows=10)
        print(f"  columns (in order): {list(df.columns)}")
        print(f"  inferred dtypes:")
        for col, dt in df.dtypes.items():
            print(f"    {col!r}: {dt}")
        print(f"  first 5 rows:")
        with pd.option_context("display.max_columns", None,
                               "display.width", 200):
            print(df.head(5).to_string(index=True))
        print(f"  row 0 as dict: {df.iloc[0].to_dict()}")
        return df
    safe(peek_nodes)

    # Count total rows without loading everything
    def count_rows():
        # Fast row count via buffered line read
        cnt = 0
        with open(BG_NODES, "rb") as f:
            for _ in f:
                cnt += 1
        # Subtract 1 for header
        return cnt - 1
    print("  counting rows (may take ~30s)…", flush=True)
    n = safe(count_rows)
    if n is not None:
        print(f"  total data rows: {n:,}")
        print(f"  expected ~49,299,864 → match: {n == 49_299_864}")


# ============================================================================
# 2. background_edges.csv
# ============================================================================
banner("2. background_edges.csv")
print(f"  path: {BG_EDGES}")
print(f"  exists: {BG_EDGES.exists()}")
if BG_EDGES.exists():
    print(f"  size:   {BG_EDGES.stat().st_size / 1e9:.2f} GB")

    def peek_edges():
        import pandas as pd
        df = pd.read_csv(BG_EDGES, nrows=10)
        print(f"  columns (in order): {list(df.columns)}")
        print(f"  inferred dtypes:")
        for col, dt in df.dtypes.items():
            print(f"    {col!r}: {dt}")
        print(f"  first 5 rows:")
        with pd.option_context("display.max_columns", None,
                               "display.width", 200):
            print(df.head(5).to_string(index=True))
        print(f"  row 0 as dict: {df.iloc[0].to_dict()}")
        return df
    safe(peek_edges)

    print("  counting rows (slow — a few minutes on 196M edges)…",
          flush=True)

    def count_edge_rows():
        cnt = 0
        with open(BG_EDGES, "rb") as f:
            for _ in f:
                cnt += 1
        return cnt - 1
    n = safe(count_edge_rows)
    if n is not None:
        print(f"  total data rows: {n:,}")
        print(f"  expected ~196M → order-of-magnitude check: "
              f"{1e8 < n < 5e8}")


# ============================================================================
# 3. raw_emb.pt
# ============================================================================
banner("3. raw_emb.pt")
print(f"  path: {RAW_EMB}")
print(f"  exists: {RAW_EMB.exists()}")
if RAW_EMB.exists():
    print(f"  size:   {RAW_EMB.stat().st_size / 1e9:.2f} GB")

    def peek_raw_emb():
        import torch
        obj = torch.load(RAW_EMB, map_location="cpu", weights_only=False)
        print(f"  type: {type(obj).__name__}")
        if torch.is_tensor(obj):
            print(f"  shape:  {tuple(obj.shape)}")
            print(f"  dtype:  {obj.dtype}")
            print(f"  device: {obj.device}")
            print(f"  row[0]  (first 10 vals): "
                  f"{obj[0, :10].tolist() if obj.ndim >= 2 else obj[:10].tolist()}")
            print(f"  has NaN: {bool(torch.isnan(obj).any())}")
            print(f"  min/max (sampled 1000 rows): "
                  f"{float(obj[:1000].min()):.3f} / {float(obj[:1000].max()):.3f}")
        elif isinstance(obj, dict):
            print(f"  dict keys: {list(obj.keys())}")
            for k, v in obj.items():
                if hasattr(v, "shape"):
                    print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    print(f"    {k}: {type(v).__name__}")
        else:
            print(f"  unexpected wrapper: {obj!r}")
    safe(peek_raw_emb)


# ============================================================================
# 4. node_idx_map.pt
# ============================================================================
banner("4. node_idx_map.pt")
print(f"  path: {NODE_IDX_MAP}")
print(f"  exists: {NODE_IDX_MAP.exists()}")
if NODE_IDX_MAP.exists():
    print(f"  size:   {NODE_IDX_MAP.stat().st_size / 1e6:.2f} MB")

    def peek_nim():
        import torch
        obj = torch.load(NODE_IDX_MAP, map_location="cpu",
                         weights_only=False)
        print(f"  type: {type(obj).__name__}")
        if torch.is_tensor(obj):
            print(f"  shape:  {tuple(obj.shape)}")
            print(f"  dtype:  {obj.dtype}")
            print(f"  min:    {int(obj.min())}")
            print(f"  max:    {int(obj.max())}")
            print(f"  first 10: {obj[:10].tolist()}")
            print(f"  last 10:  {obj[-10:].tolist()}")
            import torch as _t
            uniq = _t.unique(obj)
            print(f"  unique count: {len(uniq)} "
                  f"(expected {obj.numel()} if no duplicates)")
        else:
            print(f"  unexpected wrapper: {obj!r}")
    safe(peek_nim)


# ============================================================================
# 5. data_df.pkl
# ============================================================================
banner("5. data_df.pkl")
print(f"  path: {DATA_DF}")
print(f"  exists: {DATA_DF.exists()}")
if DATA_DF.exists():
    print(f"  size:   {DATA_DF.stat().st_size / 1e6:.2f} MB")

    def peek_df():
        import pandas as pd
        df = pd.read_pickle(DATA_DF)
        print(f"  shape: {df.shape}")
        print(f"  columns: {list(df.columns)}")
        print(f"  dtypes:")
        for col, dt in df.dtypes.items():
            print(f"    {col!r}: {dt}")

        # labels unique
        if "labels" in df.columns:
            uniq = df["labels"].unique()
            print(f"  labels.unique(): {uniq}  (count={len(uniq)})")
            try:
                vc = df["labels"].value_counts()
                print(f"  labels.value_counts():\n{vc.to_string()}")
            except Exception as e:
                print(f"  labels.value_counts() failed: {e}")

        # split unique
        if "split" in df.columns:
            uniq = df["split"].unique()
            print(f"  split.unique(): {uniq}  (count={len(uniq)})")
            try:
                vc = df["split"].value_counts()
                print(f"  split.value_counts():\n{vc.to_string()}")
            except Exception as e:
                print(f"  split.value_counts() failed: {e}")

        # node_ids_mapped format
        if "node_ids_mapped" in df.columns:
            s0 = df["node_ids_mapped"].iloc[0]
            print(f"  node_ids_mapped[0] type: {type(s0).__name__}")
            print(f"  node_ids_mapped[0] value: {s0}")
            # length stats
            try:
                lens = df["node_ids_mapped"].apply(len)
                print(f"  node_ids_mapped len stats: "
                      f"min={lens.min()} max={lens.max()} "
                      f"mean={lens.mean():.1f} median={lens.median():.1f}")
            except Exception as e:
                print(f"  len stats failed: {e}")

            # value range — sanity for the [0, 90745) assumption
            try:
                import numpy as np
                flat = np.concatenate([
                    np.asarray(x, dtype=np.int64).ravel()
                    for x in df["node_ids_mapped"].iloc[:1000]
                ])
                print(f"  node_ids_mapped[:1000] flat range: "
                      f"[{flat.min()}, {flat.max()}]")
            except Exception as e:
                print(f"  flat range failed: {e}")

        # Show a couple of full rows
        print(f"  first 2 rows (all cols):")
        with pd.option_context("display.max_columns", None,
                               "display.width", 200,
                               "display.max_colwidth", 120):
            print(df.head(2).to_string(index=True))
    safe(peek_df)


print("\nDONE — paste everything above in a single reply.")
