from impl import models, SubGDataset, train, metrics, utils, config
from impl.samplers import NoSampler, NeighborSampler, GraphSAINTSampler, EgoNetSampler

import datasets

import torch
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import argparse
import torch.nn as nn
import functools
import numpy as np
import time
import random
import yaml

parser = argparse.ArgumentParser(description='')

# Dataset settings
parser.add_argument('--dataset', type=str, default='ppi_bp')

# Node feature settings.
# deg means use node degree. one means use homogeneous embeddings.
# nodeid means use pretrained node embeddings in ./Emb
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')

# Node label settings
parser.add_argument('--use_maxzeroone', action='store_true')

# -----------------------------------------------------------------------
# Sampling settings
# --num_neighbors controls neighbor sampling in GLASSConv.
#
# Usage:
#   No sampling (baseline):
#       omit --num_neighbors entirely  →  NoSampler used, full adj
#
#   Uniform k across all layers:
#       --num_neighbors 10             →  NeighborSampler(k=10) for every layer
#
#   Per-layer k (must match conv_layer count from config):
#       --num_neighbors 15 10          →  layer 0 gets k=15, layer 1 gets k=10
#
# During inference (val/test), full adj is always used regardless of this setting.
# -----------------------------------------------------------------------
parser.add_argument(
    '--num_neighbors',
    type=int,
    nargs='+',       # accepts one or more ints: --num_neighbors 10  OR  --num_neighbors 15 10
    default=None,    # None = no sampling, full adj (baseline behaviour)
    help='Number of neighbors to sample per node per layer. '
         'Single int = uniform across layers. '
         'Multiple ints = per-layer values (must match conv_layer count). '
         'Omit for no sampling (baseline).'
)
# -----------------------------------------------------------------------
# Extended sampler selection.
#   --sampler neighbor   --num_neighbors 10  → NeighborSampler (default when --num_neighbors set)
#   --sampler graphsaint --budget 0.5        → GraphSAINTSampler (global edge fraction)
#   --sampler ego        --num_neighbors 10  → EgoNetSampler (degree-biased k neighbors)
#   --sampler none                           → NoSampler (baseline, also default)
# -----------------------------------------------------------------------
parser.add_argument(
    '--sampler',
    type=str,
    default=None,
    choices=['none', 'neighbor', 'graphsaint', 'ego'],
    help='Sampler type. Omit or "none" for baseline (full adj). '
         '"neighbor" uses --num_neighbors k (uniform). '
         '"graphsaint" uses --num_roots + --walk_len (random walk induced subgraph). '
         '"ego" uses --num_seeds (1-hop ego-graph induced subgraph).'
)
parser.add_argument(
    '--num_roots',
    type=int,
    default=200,
    help='Number of random walk roots for GraphSAINTSampler (default: 200).'
)
parser.add_argument(
    '--walk_len',
    type=int,
    default=2,
    help='Random walk length for GraphSAINTSampler (default: 2).'
)
parser.add_argument(
    '--num_seeds',
    type=int,
    default=200,
    help='Number of ego-center seeds for EgoNetSampler (default: 200).'
)

parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--use_seed', action='store_true')

args = parser.parse_args()

config.set_device(0)


def set_seed(seed: int):
    print("seed ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


if args.use_seed:
    set_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

baseG = datasets.load_dataset(args.dataset)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, output_channels = 0, 1
score_fn = None

if baseG.y.unique().shape[0] == 2:
    # binary classification task
    def loss_fn(x, y):
        return BCEWithLogitsLoss()(x.flatten(), y.flatten())
    baseG.y = baseG.y.to(torch.float)
    if baseG.y.ndim > 1:
        output_channels = baseG.y.shape[1]
    else:
        output_channels = 1
    score_fn = metrics.binaryf1
else:
    # multi-class classification task
    baseG.y = baseG.y.to(torch.int64)
    loss_fn = CrossEntropyLoss()
    output_channels = baseG.y.unique().shape[0]
    score_fn = metrics.microf1

loader_fn = SubGDataset.GDataloader
tloader_fn = SubGDataset.GDataloader


def split():
    '''
    load and split dataset.
    '''
    global trn_dataset, val_dataset, tst_dataset, baseG
    global max_deg, output_channels, loader_fn, tloader_fn

    baseG = datasets.load_dataset(args.dataset)

    if baseG.y.unique().shape[0] == 2:
        baseG.y = baseG.y.to(torch.float)
    else:
        baseG.y = baseG.y.to(torch.int64)

    # initialize node features
    if args.use_deg:
        baseG.setDegreeFeature()
    elif args.use_one:
        baseG.setOneFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        raise NotImplementedError

    max_deg = torch.max(baseG.x)
    baseG.to(config.device)

    # split data
    trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))

    # choice of dataloader
    if args.use_maxzeroone:
        def tfunc(ds, bs, shuffle=True, drop_last=True):
            return SubGDataset.ZGDataloader(ds,
                                            bs,
                                            z_fn=utils.MaxZOZ,
                                            shuffle=shuffle,
                                            drop_last=drop_last)

        def loader_fn(ds, bs):
            return tfunc(ds, bs)

        def tloader_fn(ds, bs):
            return tfunc(ds, bs, True, False)
    else:
        def loader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs)

        def tloader_fn(ds, bs):
            return SubGDataset.GDataloader(ds, bs, shuffle=True)


def buildModel(hidden_dim, conv_layer, dropout, jk, pool, z_ratio, aggr,
               batch_size=None):
    '''
    Build a GLASS model.

    Args:
        jk:           whether to use Jumping Knowledge Network.
        conv_layer:   number of GLASSConv layers.
        pool:         pooling function to transfer node embeddings to subgraph embeddings.
        z_ratio:      see GLASSConv in impl/model.py. z_ratio in [0.5, 1].
        aggr:         aggregation method. mean, sum, or gcn.

    Sampling args (from CLI):
        args.num_neighbors:
            None            → no sampling (NoSampler, baseline).
            [k]             → uniform k across all conv_layer layers.
            [k0, k1, ...]   → per-layer k values, must have conv_layer elements.
    '''
    # -----------------------------------------------------------------------
    # Build the sampler list for all conv_layer layers.
    #
    # Priority:
    #   1. --sampler graphsaint → GraphSAINTSampler(budget) for every layer
    #   2. --sampler ego        → EgoNetSampler(k) for every layer
    #   3. --sampler neighbor or --num_neighbors set → NeighborSampler(k)
    #   4. --sampler none / nothing → NoSampler (baseline)
    # -----------------------------------------------------------------------
    sampler_type = args.sampler    # may be None

    # Resolve k for per-node samplers (neighbor / ego)
    num_neighbors = args.num_neighbors   # None, [k], or [k0, k1, ...]
    if num_neighbors is not None:
        if len(num_neighbors) == 1:
            num_neighbors = num_neighbors[0]   # unwrap single-element list
        else:
            if len(num_neighbors) != conv_layer:
                raise ValueError(
                    f"--num_neighbors has {len(num_neighbors)} values but "
                    f"conv_layer={conv_layer}. They must match for per-layer sampling."
                )

    def _make_sampler_list(stype):
        if stype == "graphsaint":
            s = [GraphSAINTSampler(num_roots=args.num_roots,
                                   walk_len=args.walk_len,
                                   aggr=aggr)
                 for _ in range(conv_layer)]
            print(f"[Sampler] GraphSAINTSampler — num_roots={args.num_roots} "
                  f"walk_len={args.walk_len} aggr={aggr}", flush=True)
            return s

        if stype == "ego":
            s = [EgoNetSampler(num_seeds=args.num_seeds, aggr=aggr)
                 for _ in range(conv_layer)]
            print(f"[Sampler] EgoNetSampler — num_seeds={args.num_seeds} "
                  f"aggr={aggr}", flush=True)
            return s

        if stype == "neighbor" or (stype is None and num_neighbors is not None):
            if isinstance(num_neighbors, int):
                s = [NeighborSampler(k=num_neighbors, aggr=aggr)
                     for _ in range(conv_layer)]
                print(f"[Sampler] NeighborSampler — uniform k={num_neighbors} "
                      f"across all {conv_layer} layers, aggr={aggr}", flush=True)
            else:
                s = [NeighborSampler(k=k, aggr=aggr) for k in num_neighbors]
                print(f"[Sampler] NeighborSampler — per-layer k={num_neighbors}, "
                      f"aggr={aggr}", flush=True)
            return s

        # Default: NoSampler (baseline)
        print(f"[Sampler] NoSampler — full adj (baseline)", flush=True)
        return [NoSampler() for _ in range(conv_layer)]

    sampler_list = _make_sampler_list(sampler_type)

    conv = models.EmbZGConv(
        hidden_dim,
        hidden_dim,
        conv_layer,
        max_deg=max_deg,
        activation=nn.ELU(inplace=True),
        jk=jk,
        dropout=dropout,
        conv=functools.partial(models.GLASSConv,
                               aggr=aggr,
                               z_ratio=z_ratio,
                               dropout=dropout),
        gn=True,
        samplers=sampler_list,
        aggr=aggr,
    )

    # use pretrained node embeddings
    if args.use_nodeid:
        print("load ", f"./Emb/{args.dataset}_{hidden_dim}.pt")
        emb = torch.load(f"./Emb/{args.dataset}_{hidden_dim}.pt",
                         map_location=torch.device('cpu')).detach()
        conv.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)

    mlp = nn.Linear(hidden_dim * (conv_layer) if jk else hidden_dim,
                    output_channels)

    pool_fn_fn = {
        "mean": models.MeanPool,
        "max":  models.MaxPool,
        "sum":  models.AddPool,
        "size": models.SizePool
    }
    if pool in pool_fn_fn:
        pool_fn1 = pool_fn_fn[pool]()
    else:
        raise NotImplementedError

    gnn = models.GLASS(conv, torch.nn.ModuleList([mlp]),
                       torch.nn.ModuleList([pool_fn1])).to(config.device)
    return gnn


def test(pool="size",
         aggr="mean",
         hidden_dim=64,
         conv_layer=8,
         dropout=0.3,
         jk=1,
         lr=1e-3,
         z_ratio=0.8,
         batch_size=None,
         resi=0.7):
    '''
    Test a set of hyperparameters in a task.

    Args:
        jk:    whether to use Jumping Knowledge Network.
        z_ratio: see GLASSConv in impl/model.py.
        resi:  the lr reduce factor of ReduceLROnPlateau.
    '''
    outs = []
    t1 = time.time()

    num_div = tst_dataset.y.shape[0] / batch_size

    if args.dataset in ["density", "component", "cut_ratio", "coreness"]:
        num_div /= 5

    outs = []
    for repeat in range(args.repeat):
        set_seed((1 << repeat) - 1)
        print(f"repeat {repeat}")
        split()

        gnn = buildModel(hidden_dim, conv_layer, dropout, jk, pool, z_ratio,
                         aggr, batch_size)

        trn_loader = loader_fn(trn_dataset, batch_size)
        val_loader = tloader_fn(val_dataset, batch_size)
        tst_loader = tloader_fn(tst_dataset, batch_size)

        optimizer = Adam(gnn.parameters(), lr=lr)
        scd = lr_scheduler.ReduceLROnPlateau(optimizer,
                                             factor=resi,
                                             min_lr=5e-5)

        val_score = 0
        tst_score = 0
        early_stop = 0
        trn_time = []

        for i in range(300):
            t1 = time.time()
            loss = train.train(optimizer, gnn, trn_loader, loss_fn)
            trn_time.append(time.time() - t1)
            scd.step(loss)

            if i >= 100 / num_div:
                score, _ = train.test(gnn,
                                      val_loader,
                                      score_fn,
                                      loss_fn=loss_fn)
                if score > val_score:
                    early_stop = 0
                    val_score = score
                    score, _ = train.test(gnn,
                                          tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn)
                    tst_score = score
                    print(
                        f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}",
                        flush=True)
                elif score >= val_score - 1e-5:
                    score, _ = train.test(gnn,
                                          tst_loader,
                                          score_fn,
                                          loss_fn=loss_fn)
                    tst_score = max(score, tst_score)
                    print(
                        f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {score:.4f}",
                        flush=True)
                else:
                    early_stop += 1
                    if i % 10 == 0:
                        print(
                            f"iter {i} loss {loss:.4f} val {score:.4f} tst {train.test(gnn, tst_loader, score_fn, loss_fn=loss_fn)[0]:.4f}",
                            flush=True)

                if val_score >= 1 - 1e-5:
                    early_stop += 1
                if early_stop > 100 / num_div:
                    break

        print(
            f"end: epoch {i+1}, train time {sum(trn_time):.2f} s, "
            f"val {val_score:.3f}, tst {tst_score:.3f}",
            flush=True)
        outs.append(tst_score)

    print(
        f"average {np.average(outs):.3f} error {np.std(outs) / np.sqrt(len(outs)):.3f}"
    )
    print(args)


# read configuration
with open(f"config/{args.dataset}.yml") as f:
    params = yaml.safe_load(f)

print("params", params, flush=True)

split()
test(**(params))