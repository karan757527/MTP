"""
inspect_torch.py — torch-only follow-up probe.

Run in any env that has torch installed. Paste stdout back.
"""
from pathlib import Path
import torch

RAW_DIR = Path("/lfs/usrhome/mtech/cs24m021/raw_local")
RAW_EMB      = RAW_DIR / "raw_emb.pt"
NODE_IDX_MAP = RAW_DIR / "node_idx_map.pt"


def banner(t):
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70)


# ---- raw_emb.pt ----
banner("raw_emb.pt")
obj = torch.load(RAW_EMB, map_location="cpu", weights_only=False)
print(f"type: {type(obj).__name__}")
if torch.is_tensor(obj):
    print(f"shape:  {tuple(obj.shape)}")
    print(f"dtype:  {obj.dtype}")
    print(f"numel:  {obj.numel():,}")
    print(f"row[0] first 10: {obj[0, :10].tolist() if obj.ndim >= 2 else obj[:10].tolist()}")
    print(f"row[1] first 10: {obj[1, :10].tolist() if obj.ndim >= 2 else None}")
    if obj.ndim >= 2:
        s = obj[:1000]
        print(f"min/max (first 1000 rows): {float(s.min()):.4f} / {float(s.max()):.4f}")
elif isinstance(obj, dict):
    print(f"dict keys: {list(obj.keys())}")
    for k, v in obj.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} = {v!r}"[:200])
else:
    print(f"unexpected wrapper: {type(obj)}")


# ---- node_idx_map.pt ----
banner("node_idx_map.pt")
obj = torch.load(NODE_IDX_MAP, map_location="cpu", weights_only=False)
print(f"type: {type(obj).__name__}")
if torch.is_tensor(obj):
    print(f"shape:  {tuple(obj.shape)}")
    print(f"dtype:  {obj.dtype}")
    print(f"numel:  {obj.numel():,}")
    print(f"min:    {int(obj.min())}")
    print(f"max:    {int(obj.max())}")
    print(f"first 10: {obj[:10].tolist()}")
    print(f"last 10:  {obj[-10:].tolist()}")
    uniq = torch.unique(obj)
    print(f"unique count: {len(uniq):,} (total {obj.numel():,})")
elif isinstance(obj, dict):
    print(f"dict, len={len(obj)}")
    items = list(obj.items())[:5]
    for k, v in items:
        print(f"  sample: {k!r} -> {v!r}  (key type {type(k).__name__}, val type {type(v).__name__})")
    print(f"  ...")
elif isinstance(obj, list):
    print(f"list, len={len(obj)}")
    print(f"first 10: {obj[:10]}")
    print(f"last 10:  {obj[-10:]}")
else:
    print(f"unexpected wrapper: {type(obj)}")
    print(repr(obj)[:500])


# ---- cross-check: does node_idx_map align with raw_emb row indexing? ----
banner("cross-check")
# If raw_emb has ~49.3M rows, it's the full background features.
# If raw_emb has ~90k rows, it's already indexed by dense idx.
try:
    if torch.is_tensor(obj):
        print(f"node_idx_map is a tensor — "
              f"likely maps dense_idx -> clId OR dense_idx -> row_in_background")
except Exception as e:
    print(f"err: {e}")

print("\nDONE")
