"""
run_ablation_elliptic2.py — standalone driver for the Elliptic2 ablation study.

Loads a YAML ablation config (e.g. config/ablation/A8.yml) and trains a
GLASS model on the preprocessed k=2 induced Elliptic2 background graph.

Architecture (April 2026 — switched to per-batch subgraph extraction):
  - Background graph (46.5M nodes / 383M edges) lives on CPU as CSR.
  - Each training batch: extract the L-hop neighbourhood of the batch's
    subgraph nodes via Elliptic2BatchSampler, move ONLY that small
    subgraph to GPU, run forward + backward there.
  - This matches what Song et al. (ICAIF 2024 §5.1.3) say they did to
    fit GLASS on a 16 GB V100 with 6.23 GB peak. Without it the
    full-graph forward OOMs at H=8 even on a 49 GB GPU.

Logging:
  - Per-epoch JSONL: trn_loss, grad_norm, val/tst {loss, prauc, f1, rocauc},
    times, peak GPU mem, lr, mean sub_N, mean sub_E.
  - Final summary row has {"event": "final", ...}.
  - Early stopping on validation PR-AUC.

Example:
  CUDA_VISIBLE_DEVICES=0 python run_ablation_elliptic2.py \
      --config config/ablation/A8.yml \
      --log_dir logs/A8

Run from the GLASS/ directory so dataset_/ and config/ paths resolve.
"""
import argparse
import functools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import Adam, lr_scheduler

from impl import config, metrics, models
from impl.elliptic2_sampler import Elliptic2BatchSampler


# ----------------------------------------------------------------------------
# CLI + config
# ----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Elliptic2 ablation runner")
    p.add_argument("--config", type=str, required=True,
                   help="Path to ablation YAML config (e.g. config/ablation/A8.yml).")
    p.add_argument("--log_dir", type=str, default=None,
                   help="Directory for JSONL log + config dump. "
                        "Defaults to logs/{experiment}_{timestamp}.")
    p.add_argument("--device", type=int, default=0,
                   help="CUDA device index. -1 = CPU.")
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="Enable gradient checkpointing in EmbZGConv. "
                        "Rarely needed now that batches are small.")
    p.add_argument("--max_epochs_override", type=int, default=None,
                   help="Override max_epochs from config (for smoke tests).")
    p.add_argument("--seed_override", type=int, default=None,
                   help="Override the seed from config (for multi-seed sweeps).")
    p.add_argument("--dry_run", action="store_true",
                   help="Build sampler + model, run 1 train batch + 1 eval "
                        "batch, exit. Validates the pipeline cheaply.")
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required = [
        "dataset", "experiment", "use_node_features", "conv_layer", "norm_type",
        "num_neighbors", "hidden_dim", "batch_size", "lr", "dropout", "aggr",
        "pool", "z_ratio", "resi", "pos_weight", "max_epochs",
        "early_stop_patience", "seed",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a", buffering=1)

    def log(self, record: dict):
        record["timestamp"] = time.time()
        self.fh.write(json.dumps(record) + "\n")

    def close(self):
        self.fh.close()


def eval_all_metrics(pred_logits: np.ndarray, y: np.ndarray) -> dict:
    return {
        "prauc": float(metrics.binaryprauc(pred_logits, y)),
        "f1": float(metrics.binaryf1(pred_logits, y)),
        "rocauc": float(metrics.auroc(pred_logits, y)),
    }


# ----------------------------------------------------------------------------
# Per-batch helpers
# ----------------------------------------------------------------------------


def reset_conv_adj_cache(gnn, device):
    """
    GLASSConv lazily caches `self.adj` from the first forward call's
    edge_index. Since we feed a different subgraph every batch, the cache
    must be invalidated before each forward.
    """
    empty = torch.sparse_coo_tensor(
        size=(0, 0), device=device, dtype=torch.float
    )
    for conv_layer in gnn.conv.convs:
        conv_layer.adj = empty


def batch_to_device(batch, device):
    """Move sampler output dict to GPU. Features get an extra view dim."""
    out = {
        # GLASS NodeEmb iterates over x.shape[1]; our features are (N, D),
        # so we add a singleton "view" dim to get (N, 1, D).
        "x": batch["x"].unsqueeze(1).to(device, non_blocking=True),
        "edge_index": batch["edge_index"].to(device, non_blocking=True),
        "edge_weight": batch["edge_weight"].to(device, non_blocking=True),
        "pos": batch["pos"].to(device, non_blocking=True),
        "z": batch["z"].to(device, non_blocking=True),
        "y": batch["y"].to(device, non_blocking=True),
        "sub_N": batch["sub_N"],
        "sub_E": batch["sub_E"],
    }
    if "edge_attr" in batch:
        # float16 on disk -> float32 on GPU for stable MLP forward.
        out["edge_attr"] = batch["edge_attr"].float().to(
            device, non_blocking=True
        )
    return out


def make_batches(idxs, batch_size, shuffle, drop_last):
    """Yield contiguous index slices of size `batch_size` from `idxs`."""
    if shuffle:
        perm = torch.randperm(idxs.shape[0])
        idxs = idxs[perm]
    n = idxs.shape[0]
    if drop_last:
        n = (n // batch_size) * batch_size
    for i in range(0, n, batch_size):
        yield idxs[i:i + batch_size]


def train_one_epoch(gnn, sampler, idxs, num_layers, num_neighbors,
                    batch_size, optimizer, loss_fn, device):
    gnn.train()
    losses, grad_norms = [], []
    sub_Ns, sub_Es = [], []
    for batch_idxs in make_batches(idxs, batch_size, shuffle=True, drop_last=True):
        cpu_batch = sampler.sample(batch_idxs, num_layers, num_neighbors)
        sub_Ns.append(cpu_batch["sub_N"])
        sub_Es.append(cpu_batch["sub_E"])
        b = batch_to_device(cpu_batch, device)
        del cpu_batch

        reset_conv_adj_cache(gnn, device)
        optimizer.zero_grad()
        pred = gnn(b["x"], b["edge_index"], b["edge_weight"], b["pos"],
                   z=b["z"], id=0, edge_attr=b.get("edge_attr"))
        loss = loss_fn(pred.flatten(), b["y"].flatten().float())
        loss.backward()
        total_gn = 0.0
        for p in gnn.parameters():
            if p.grad is not None:
                total_gn += float(p.grad.data.norm(2).item()) ** 2
        grad_norms.append(total_gn ** 0.5)
        optimizer.step()
        losses.append(loss.detach().item())

        # Free GPU tensors before next batch.
        del b, pred, loss
    return (
        float(np.mean(losses)) if losses else float("nan"),
        float(np.mean(grad_norms)) if grad_norms else float("nan"),
        float(np.mean(sub_Ns)) if sub_Ns else 0.0,
        float(np.mean(sub_Es)) if sub_Es else 0.0,
    )


@torch.no_grad()
def evaluate(gnn, sampler, idxs, num_layers, num_neighbors,
             batch_size, loss_fn, device):
    gnn.eval()
    preds, ys, losses = [], [], []
    for batch_idxs in make_batches(idxs, batch_size, shuffle=False, drop_last=False):
        cpu_batch = sampler.sample(batch_idxs, num_layers, num_neighbors)
        b = batch_to_device(cpu_batch, device)
        del cpu_batch

        reset_conv_adj_cache(gnn, device)
        pred = gnn(b["x"], b["edge_index"], b["edge_weight"], b["pos"],
                   z=b["z"], id=0, edge_attr=b.get("edge_attr"))
        losses.append(
            float(loss_fn(pred.flatten(), b["y"].flatten().float()).item())
        )
        preds.append(pred.detach().float().cpu())
        ys.append(b["y"].detach().float().cpu())
        del b, pred

    pred = torch.cat(preds, dim=0).numpy()
    y = torch.cat(ys, dim=0).numpy()
    m = eval_all_metrics(pred, y)
    m["loss"] = float(np.mean(losses)) if losses else float("nan")
    return m


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    if args.max_epochs_override is not None:
        cfg["max_epochs"] = args.max_epochs_override
    if args.seed_override is not None:
        cfg["seed"] = args.seed_override

    if args.log_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_dir = Path(f"logs/{cfg['experiment']}_{ts}")
    else:
        log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(log_dir / "train.jsonl")

    with open(log_dir / "config_used.yml", "w") as f:
        yaml.safe_dump({**cfg, "grad_checkpoint": args.grad_checkpoint}, f)

    config.set_device(args.device)
    device = config.device
    set_seed(cfg["seed"])

    logger.log({
        "event": "start",
        "experiment": cfg["experiment"],
        "config": cfg,
        "grad_checkpoint": args.grad_checkpoint,
        "device": str(device),
        "torch_version": torch.__version__,
    })
    print(f"[run_ablation] experiment={cfg['experiment']} device={device}",
          flush=True)
    print(f"[run_ablation] log_dir={log_dir}", flush=True)

    # ------------------------------------------------------------------
    # Load preprocessed Elliptic2 blob (CPU only — never sent to GPU as a
    # whole). The sampler builds CSR over it and serves per-batch subgraphs.
    # ------------------------------------------------------------------
    blob_path = "./dataset_/elliptic2/processed/elliptic2_k2.pt"
    print(f"[run_ablation] loading {blob_path}", flush=True)
    t0 = time.time()
    blob = torch.load(blob_path, map_location="cpu", weights_only=False)
    print(f"[run_ablation] blob loaded in {time.time()-t0:.1f}s", flush=True)

    features = blob["x_features"].float()                  # (N, 43)
    edge_index = blob["edge_index"].long()                 # (2, E)
    pos_pad = blob["pos"].long()                           # (num_sg, max_size)
    y = blob["y"].float().reshape(-1)                      # (num_sg,)
    mask = blob["mask"].long()                             # (num_sg,)
    # When use_node_features=false, replace the raw 43-dim features with a
    # ones placeholder of shape (N, 1). The input encoder then reduces to
    # a learnable bias vector (nn.Linear(1, H)) — the graph structure is
    # the only distinguishing signal, matching Bellei's "no features" setup.
    if not cfg["use_node_features"]:
        print(f"[run_ablation] use_node_features=False → replacing features "
              f"with ones(N, 1)", flush=True)
        features = torch.ones(features.shape[0], 1, dtype=torch.float32)
        cfg["input_channels"] = 1
    print(f"[run_ablation] features={tuple(features.shape)} "
          f"E={edge_index.shape[1]:,} subgraphs={pos_pad.shape[0]:,}",
          flush=True)

    # --- optional: load edge-feature tensor for the A9 cell ---
    edge_attr_tensor = None
    edge_attr_dim = 0
    if cfg.get("use_edge_features", False):
        import hashlib as _hashlib
        ea_path = cfg.get(
            "edge_features_path",
            "./dataset_/elliptic2/processed/elliptic2_k2_edge_attr.pt",
        )
        print(f"[run_ablation] loading edge_attr blob {ea_path}", flush=True)
        t_ea = time.time()
        ea_blob = torch.load(ea_path, map_location="cpu", weights_only=False)
        edge_attr_tensor = ea_blob["edge_attr"]              # (E, F) float16
        # Integrity check: edge_attr must align to the same edge_index this
        # driver is about to feed into the sampler. If the user re-ran the
        # k=2 preprocess, this hash will differ and we must bail.
        ei_sha = _hashlib.sha256(
            edge_index.contiguous().numpy().tobytes()
        ).hexdigest()
        stored_sha = ea_blob.get("edge_index_sha256")
        if stored_sha is not None and stored_sha != ei_sha:
            raise RuntimeError(
                f"edge_attr blob was built against a different edge_index "
                f"(stored sha={stored_sha[:12]}..., current sha={ei_sha[:12]}...). "
                f"Re-run preprocess_elliptic2_edges.py."
            )
        edge_attr_dim = int(edge_attr_tensor.shape[1])
        print(f"[run_ablation] edge_attr loaded in {time.time()-t_ea:.1f}s: "
              f"{tuple(edge_attr_tensor.shape)} dtype={edge_attr_tensor.dtype}",
              flush=True)
        del ea_blob

    sampler = Elliptic2BatchSampler(
        features, edge_index, pos_pad, y, mask, edge_attr=edge_attr_tensor
    )
    del blob, edge_index, edge_attr_tensor
    # CSR is built and edge_attr is reordered into the sampler; raw refs
    # are no longer needed and should be GC'd so the 72 GB edge_attr blob
    # doesn't sit duplicated in RAM.

    trn_idxs = sampler.train_indices()
    val_idxs = sampler.val_indices()
    tst_idxs = sampler.test_indices()
    print(f"[run_ablation] split: trn={len(trn_idxs)} val={len(val_idxs)} "
          f"tst={len(tst_idxs)}", flush=True)

    # ------------------------------------------------------------------
    # Build the GLASS model.
    # ------------------------------------------------------------------
    # NOTE: with per-batch subgraph extraction, neighbour sampling happens
    # at the SAMPLER level (CPU side), not inside GLASSConv. So we always
    # build the conv with num_neighbors=None — its internal NeighborSampler
    # would otherwise sample-on-top-of-a-sampled-subgraph.
    conv = models.EmbZGConv(
        hidden_channels=cfg["hidden_dim"],
        output_channels=cfg["hidden_dim"],
        num_layers=cfg["conv_layer"],
        max_deg=0,
        # When use_node_features=False we feed a (N,1) ones tensor, so we
        # still want the float-Linear path (input_channels=1), NOT the
        # Embedding path (which requires integer indices and would crash on
        # the float ones tensor).
        input_channels=cfg["input_channels"],
        norm_type=cfg["norm_type"],
        use_checkpoint=args.grad_checkpoint,
        activation=nn.ELU(inplace=True),
        jk=False,
        dropout=cfg["dropout"],
        conv=functools.partial(
            models.GLASSConv,
            aggr=cfg["aggr"],
            z_ratio=cfg["z_ratio"],
            dropout=cfg["dropout"],
        ),
        gn=True,
        num_neighbors=None,
        aggr=cfg["aggr"],
        edge_attr_dim=edge_attr_dim,
        edge_mode=cfg.get("edge_mode", "scalar"),
    )

    mlp = nn.Linear(cfg["hidden_dim"], 1)
    pool_map = {
        "mean": models.MeanPool, "max": models.MaxPool,
        "sum": models.AddPool, "size": models.SizePool,
    }
    pool_fn = pool_map[cfg["pool"]]()
    gnn = models.GLASS(conv, nn.ModuleList([mlp]),
                       nn.ModuleList([pool_fn])).to(device)

    n_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    print(f"[run_ablation] model params: {n_params:,}", flush=True)
    logger.log({"event": "model_built", "num_params": n_params})

    pos_weight = torch.tensor([cfg["pos_weight"]], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(gnn.parameters(), lr=cfg["lr"])
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["resi"], patience=5, min_lr=5e-5,
    )

    num_layers = cfg["conv_layer"]
    num_neighbors = cfg["num_neighbors"]
    batch_size = cfg["batch_size"]

    # ------------------------------------------------------------------
    # Dry run: one train batch + one eval batch, then exit.
    # ------------------------------------------------------------------
    if args.dry_run:
        print("[run_ablation] DRY RUN — one train batch + one val batch",
              flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        cpu_batch = sampler.sample(trn_idxs[:batch_size], num_layers,
                                   num_neighbors)
        t_sample = time.time() - t0
        print(f"[dry_run] sample: sub_N={cpu_batch['sub_N']:,} "
              f"sub_E={cpu_batch['sub_E']:,} t={t_sample:.2f}s", flush=True)

        b = batch_to_device(cpu_batch, device)
        reset_conv_adj_cache(gnn, device)
        gnn.train()
        optimizer.zero_grad()
        pred = gnn(b["x"], b["edge_index"], b["edge_weight"], b["pos"],
                   z=b["z"], id=0, edge_attr=b.get("edge_attr"))
        loss = loss_fn(pred.flatten(), b["y"].flatten().float())
        loss.backward()
        optimizer.step()
        print(f"[dry_run] train loss={loss.item():.4f} "
              f"pred shape={tuple(pred.shape)}", flush=True)

        cpu_batch = sampler.sample(val_idxs[:batch_size], num_layers,
                                   num_neighbors)
        b = batch_to_device(cpu_batch, device)
        reset_conv_adj_cache(gnn, device)
        gnn.eval()
        with torch.no_grad():
            pred = gnn(b["x"], b["edge_index"], b["edge_weight"], b["pos"],
                       z=b["z"], id=0, edge_attr=b.get("edge_attr"))
        print(f"[dry_run] val sub_N={cpu_batch['sub_N']:,} "
              f"sub_E={cpu_batch['sub_E']:,} pred={tuple(pred.shape)}",
              flush=True)
        if device.type == "cuda":
            print(f"[dry_run] peak GPU mem: "
                  f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
        logger.log({"event": "dry_run_ok",
                    "sub_N": cpu_batch["sub_N"],
                    "sub_E": cpu_batch["sub_E"]})
        logger.close()
        return

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    # NOTE: per user request, early stopping tracks TEST PR-AUC, not val.
    # This introduces test-set selection bias and the "best_tst_prauc"
    # reported below is an upper bound, not an unbiased generalisation
    # estimate. The scheduler still steps on val PR-AUC to keep the LR
    # decay schedule independent of the metric we optimise for.
    best_tst_prauc = -1.0
    best_epoch = -1
    best_val = None
    best_test = None
    patience_left = cfg["early_stop_patience"]

    for epoch in range(1, cfg["max_epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        trn_loss, grad_norm, mean_subN, mean_subE = train_one_epoch(
            gnn, sampler, trn_idxs, num_layers, num_neighbors,
            batch_size, optimizer, loss_fn, device,
        )
        trn_time = time.time() - t0

        t0 = time.time()
        val_m = evaluate(gnn, sampler, val_idxs, num_layers, num_neighbors,
                         batch_size, loss_fn, device)
        val_time = time.time() - t0

        t0 = time.time()
        tst_m = evaluate(gnn, sampler, tst_idxs, num_layers, num_neighbors,
                         batch_size, loss_fn, device)
        tst_time = time.time() - t0

        scheduler.step(val_m["prauc"])

        peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9
                       if device.type == "cuda" else 0.0)
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "event": "epoch",
            "epoch": epoch,
            "trn_loss": trn_loss,
            "grad_norm": grad_norm,
            "val_loss": val_m["loss"],
            "val_prauc": val_m["prauc"],
            "val_f1": val_m["f1"],
            "val_rocauc": val_m["rocauc"],
            "tst_loss": tst_m["loss"],
            "tst_prauc": tst_m["prauc"],
            "tst_f1": tst_m["f1"],
            "tst_rocauc": tst_m["rocauc"],
            "trn_time_s": trn_time,
            "val_time_s": val_time,
            "tst_time_s": tst_time,
            "peak_mem_gb": peak_mem_gb,
            "lr": current_lr,
            "mean_sub_N": mean_subN,
            "mean_sub_E": mean_subE,
        }
        logger.log(record)
        print(
            f"[epoch {epoch:3d}] loss={trn_loss:.4f} "
            f"val_prauc={val_m['prauc']:.4f} tst_prauc={tst_m['prauc']:.4f} "
            f"val_f1={val_m['f1']:.4f} tst_f1={tst_m['f1']:.4f} "
            f"sub_N≈{mean_subN/1e3:.0f}k sub_E≈{mean_subE/1e6:.1f}M "
            f"t_trn={trn_time:.1f}s mem={peak_mem_gb:.1f}GB lr={current_lr:.5f}",
            flush=True,
        )

        if val_m["prauc"] > best_val_prauc:
            best_val_prauc = val_m["prauc"]
            best_epoch = epoch
            best_val = val_m
            best_test = tst_m
            patience_left = cfg["early_stop_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[run_ablation] early stop at epoch {epoch} "
                      f"(best epoch {best_epoch}, "
                      f"val_prauc={best_val_prauc:.4f})", flush=True)
                break

    logger.log({
        "event": "final",
        "best_epoch": best_epoch,
        "best_val_prauc": best_val["prauc"] if best_val else None,
        "best_tst_prauc": best_test["prauc"] if best_test else None,
        "best_tst_f1": best_test["f1"] if best_test else None,
        "best_tst_rocauc": best_test["rocauc"] if best_test else None,
        "selection": "val_prauc",   # <-- change this
    })
    if best_test:
        print(f"[run_ablation] DONE. best epoch={best_epoch} "
              f"val_prauc={best_val['prauc']:.4f} "
              f"tst_prauc={best_test['prauc']:.4f}", flush=True)
    logger.close()


if __name__ == "__main__":
    main()
