"""
smoke_test_elliptic2.py — staged sanity checks for the Elliptic2 pipeline.

Run these stages in order BEFORE committing to a full ablation run. Each
stage is cheap and isolates one failure mode.

Stages:
  1. file_check  — .pt file exists, shapes/dtypes match expectations.
                   (CPU-only, ~20s, ~18 GB RAM)
  2. synthetic   — build a 10k-node fake graph, run model forward+backward.
                   Validates the model code (input_channels path, norm_type,
                   checkpointing) without paying the full-graph memory cost.
                   (GPU, <10s, <1 GB)
  3. gpu_load    — load full Elliptic2, move to GPU, report peak memory.
                   Does NOT run the model. Use to check whether the dataset
                   even fits on the device before wasting time on training.
                   (GPU, ~60s, ~25 GB VRAM expected)
  4. dry_run     — full pipeline: data + model + 1 train batch + 1 eval batch.
                   Equivalent to `run_ablation_elliptic2.py --dry_run`.
                   (GPU, ~5 min, ~45 GB VRAM expected with grad_checkpoint)

Usage:
  CUDA_VISIBLE_DEVICES=0 python smoke_test_elliptic2.py --stage file_check
  CUDA_VISIBLE_DEVICES=0 python smoke_test_elliptic2.py --stage synthetic
  CUDA_VISIBLE_DEVICES=0 python smoke_test_elliptic2.py --stage gpu_load
  CUDA_VISIBLE_DEVICES=0 python smoke_test_elliptic2.py --stage dry_run

Run from GLASS/ directory.
"""
import argparse
import functools
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


PROCESSED_PT = Path("dataset_/elliptic2/processed/elliptic2_k2.pt")


# ----------------------------------------------------------------------------
# Stage 1: file_check
# ----------------------------------------------------------------------------


def stage_file_check():
    print(f"[file_check] path: {PROCESSED_PT}")
    if not PROCESSED_PT.exists():
        print("[file_check] FAIL: file does not exist")
        sys.exit(1)
    size_gb = PROCESSED_PT.stat().st_size / 1e9
    print(f"[file_check] size: {size_gb:.2f} GB")

    t0 = time.time()
    blob = torch.load(PROCESSED_PT, map_location="cpu", weights_only=False)
    print(f"[file_check] loaded in {time.time()-t0:.1f}s")

    required = ["x_features", "edge_index", "edge_weight", "pos", "y", "mask"]
    for k in required:
        if k not in blob:
            print(f"[file_check] FAIL: missing key {k!r}")
            sys.exit(1)

    x = blob["x_features"]
    ei = blob["edge_index"]
    ew = blob["edge_weight"]
    pos = blob["pos"]
    y = blob["y"]
    mask = blob["mask"]

    print(f"[file_check] x_features: shape={tuple(x.shape)} dtype={x.dtype}")
    print(f"[file_check] edge_index: shape={tuple(ei.shape)} dtype={ei.dtype}")
    print(f"[file_check] edge_weight: shape={tuple(ew.shape)} dtype={ew.dtype}")
    print(f"[file_check] pos: shape={tuple(pos.shape)} dtype={pos.dtype}")
    print(f"[file_check] y: shape={tuple(y.shape)} dtype={y.dtype}")
    print(f"[file_check] mask: shape={tuple(mask.shape)} dtype={mask.dtype}")

    # Sanity checks on the shapes / contents.
    assert x.ndim == 2 and x.shape[1] == 43, f"bad x shape: {x.shape}"
    assert ei.ndim == 2 and ei.shape[0] == 2, f"bad ei shape: {ei.shape}"
    assert ew.shape[0] == ei.shape[1], f"ew/ei edge count mismatch"
    assert y.shape[0] == mask.shape[0] == pos.shape[0] == 110902, \
        f"subgraph count mismatch: y={y.shape} mask={mask.shape} pos={pos.shape}"

    n_pos = int((y > 0.5).sum())
    n_neg = int((y < 0.5).sum())
    print(f"[file_check] labels: pos={n_pos} neg={n_neg} ratio={n_neg/max(n_pos,1):.1f}:1")
    for s, name in zip([0, 1, 2], ["train", "val", "test"]):
        cnt = int((mask == s).sum())
        print(f"[file_check] split {name}: {cnt}")
    print(f"[file_check] num_nodes: {x.shape[0]:,}")
    print(f"[file_check] num_edges (directed): {ei.shape[1]:,}")
    print(f"[file_check] pos range: min={int(pos[pos >= 0].min())} "
          f"max={int(pos.max())}")
    print("[file_check] OK")


# ----------------------------------------------------------------------------
# Stage 2: synthetic — tiny graph, full model
# ----------------------------------------------------------------------------


def stage_synthetic():
    from impl import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[synthetic] device={device}")

    N = 10_000
    E = 50_000
    D = 43
    H = 32
    B = 64  # subgraphs per batch
    SG_SIZE = 10

    torch.manual_seed(0)
    x = torch.randn(N, D, device=device).unsqueeze(1)  # (N, 1, D)
    src = torch.randint(0, N, (E,), device=device)
    dst = torch.randint(0, N, (E,), device=device)
    edge_index = torch.stack([src, dst], dim=0).long()
    edge_weight = torch.ones(E, device=device)

    # Build random "subgraphs": B subgraphs, each with SG_SIZE node ids
    pos = torch.randint(0, N, (B, SG_SIZE), device=device).long()
    y = torch.randint(0, 2, (B,), device=device).float()

    # --- build model the same way the runner does ---
    for layers in [1, 2]:
        for norm in ["graph", "layer", "batch"]:
            for ckpt in [False, True]:
                conv = models.EmbZGConv(
                    hidden_channels=H,
                    output_channels=H,
                    num_layers=layers,
                    max_deg=0,
                    input_channels=D,
                    norm_type=norm,
                    use_checkpoint=ckpt,
                    activation=nn.ELU(inplace=True),
                    jk=False,
                    dropout=0.0,
                    conv=functools.partial(
                        models.GLASSConv, aggr="sum", z_ratio=0.8, dropout=0.0,
                    ),
                    gn=True,
                    num_neighbors=None,
                    aggr="sum",
                )
                mlp = nn.Linear(H, 1)
                pool = models.SizePool()
                gnn = models.GLASS(
                    conv, nn.ModuleList([mlp]), nn.ModuleList([pool])
                ).to(device)

                from impl.utils import MaxZOZ
                z = MaxZOZ(x.reshape(N, D), pos)

                gnn.train()
                pred = gnn(x, edge_index, edge_weight, pos, z=z, id=0)
                loss = nn.BCEWithLogitsLoss()(pred.flatten(), y.flatten())
                loss.backward()
                print(f"[synthetic] layers={layers} norm={norm:5s} "
                      f"ckpt={str(ckpt):5s}  loss={loss.item():.4f} "
                      f"pred={tuple(pred.shape)}")
                del gnn, conv, pred, loss
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    print("[synthetic] OK")


# ----------------------------------------------------------------------------
# Stage 3: gpu_load — just load the data onto the GPU
# ----------------------------------------------------------------------------


def stage_gpu_load():
    if not torch.cuda.is_available():
        print("[gpu_load] SKIP: no CUDA device")
        return
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()

    import datasets
    t0 = time.time()
    baseG = datasets.load_dataset("elliptic2")
    baseG.y = baseG.y.to(torch.float)
    print(f"[gpu_load] loaded in {time.time()-t0:.1f}s (CPU)")

    t0 = time.time()
    baseG.to(device)
    torch.cuda.synchronize()
    print(f"[gpu_load] moved in {time.time()-t0:.1f}s")
    print(f"[gpu_load] peak allocated: "
          f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"[gpu_load] current allocated: "
          f"{torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("[gpu_load] OK")


# ----------------------------------------------------------------------------
# Stage 4: dry_run — defer to the runner
# ----------------------------------------------------------------------------


def stage_dry_run():
    import subprocess
    cmd = [
        sys.executable, "run_ablation_elliptic2.py",
        "--config", "config/ablation/A2.yml",
        "--log_dir", "logs/smoke_dry_run",
        "--grad_checkpoint",
        "--dry_run",
    ]
    print(f"[dry_run] exec: {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"[dry_run] FAIL: rc={r.returncode}")
        sys.exit(r.returncode)
    print("[dry_run] OK")


STAGES = {
    "file_check": stage_file_check,
    "synthetic": stage_synthetic,
    "gpu_load": stage_gpu_load,
    "dry_run": stage_dry_run,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES.keys()))
    args = ap.parse_args()
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
