# Practical Subgraph Learning for Anti-Money-Laundering on Blockchain Graphs

This repository accompanies the MTech project report _"Practical Subgraph
Learning for Anti-Money-Laundering on Blockchain Graphs"_ (Karan Agrawal,
IIT Madras, 2026). It builds on the official
[GLASS](https://github.com/Xi-yuanWang/GLASS) implementation by Wang &
Zhang (ICLR 2022) and extends it with:

1. **A per-batch CSR sampler for Elliptic2** that holds the 46.5M-node /
   383M-edge background graph in CPU memory and ships only each
   minibatch's L-hop neighbourhood to the GPU, making the dataset
   trainable on a single 32 GB V100.
2. **A three-sampler characterisation** on the four GLASS-paper synthetic
   benchmarks comparing per-source NeighborSampler, GraphSAINT
   random-walk, and 1-hop EgoNet.
3. **A twelve-cell, seven-seed ablation** disentangling the four factors
   (features, depth, normalisation, fanout cap) that distinguish the
   recipes of Bellei et al. (2024) and Song et al. (2024) on Elliptic2,
   plus four cells exercising edge-feature integrations.

GLASS itself, the synthetic-benchmark loaders, and the training loop
skeleton are taken from the upstream repository (LICENSE inherited from
that repository); the components added or substantially rewritten for
this project are listed in the [What is new](#what-is-new) section.

## Repository layout

```
.
├── config/                    YAML configs
│   ├── component.yml          synthetic — component-classification
│   ├── coreness.yml           synthetic — coreness
│   ├── cut_ratio.yml          synthetic — cut-ratio
│   ├── density.yml            synthetic — density
│   ├── em_user.yml            real-world — EM-User
│   ├── hpo_metab.yml          real-world — HPO-Metab
│   ├── hpo_neuro.yml          real-world — HPO-Neuro
│   ├── ppi_bp.yml             real-world — PPI-BP
│   └── ablation/              twelve Elliptic2 ablation cells (A1..A11)
├── impl/                      core library
│   ├── config.py              CLI/YAML config plumbing
│   ├── models.py              GLASS, EmbZGConv, MyGINConv, GLASSConv
│   ├── samplers.py            NeighborSampler, GraphSAINT-RW, EgoNet (NEW)
│   ├── elliptic2_sampler.py   per-batch CSR L-hop extractor (NEW)
│   ├── train.py               train/eval loops with sampler hooks
│   ├── metrics.py             PR-AUC, ROC-AUC, F1
│   ├── SubGDataset.py         GLASS dataset/dataloader
│   └── utils.py               misc helpers
├── scripts/                   shell + python helpers
│   ├── aggregate_seeds.py     per-cell summary builder
│   ├── compute_edge_attr_norm.py   edge-feature normalisation stats
│   ├── regenerate_summaries.py     re-derive summary JSONs from JSONL
│   ├── run_*.sh                    cell sweeps and resume helpers
│   └── ...
├── datasets.py                synthetic + real-world dataset loaders
├── GLASSTest.py               upstream GLASS entry point (synthetic & real)
├── GNNEmb.py                  upstream — node-embedding pre-training
├── GNNSeg.py                  upstream — GNN-Seg baseline
├── preprocess_elliptic2.py    one-shot 2-hop induction over Elliptic2 (NEW)
├── preprocess_elliptic2_edges.py   edge-feature alignment for Elliptic2 (NEW)
├── inspect_elliptic2.py       sanity-check raw Elliptic2 tensors (NEW)
├── inspect_torch.py           torch / GPU / MPS environment probe (NEW)
├── smoke_test_elliptic2.py    short run to validate the pipeline (NEW)
├── run_ablation_elliptic2.py  driver for the 12-cell ablation (NEW)
├── run_neighbor_sampling_experiments.py  per-source-NeighborSampler sweep (NEW)
├── run_graphsaint_experiments.py         GraphSAINT-RW sweep (NEW)
├── run_ego_experiments.py                1-hop EgoNet sweep (NEW)
├── requirements.txt
├── .gitignore
└── README.md
```

## What is new

Files marked NEW above were authored or substantially rewritten for this
project. The most load-bearing additions are:

- `impl/elliptic2_sampler.py` — CPU-side CSR built once at start-up;
  per-batch `sample()` walks L hops from the seed nodes of the batch's
  target subgraphs and returns a small locally-relabelled subgraph
  ready for the GPU. The full-graph reading of GLASS exhausts a 32 GB
  V100 at any reasonable hidden width; this brings peak GPU memory to
  the hundreds-of-megabytes range on representative cells.
- `impl/samplers.py` — three drop-in samplers for `GLASSConv` /
  `EmbZGConv` (per-source NeighborSampler, GraphSAINT random-walk,
  1-hop EgoNet), each preserving the GLASS labelling channel and
  `aggr`-correct edge re-weighting.
- `preprocess_elliptic2.py` / `preprocess_elliptic2_edges.py` — the
  2-hop induction around the 90,745 labelled nodes and the edge-feature
  alignment / normalisation that the ablation needs.
- `run_ablation_elliptic2.py` — driver for the twelve-cell ablation with
  per-epoch JSONL logs, peak-GPU tracking, and validation-PR-AUC early
  stopping.
- `run_*_experiments.py` — three drivers that sweep `k` (and analogous
  knobs) for the sampler characterisation on the synthetic benchmarks.

## Setup

Tested with Python 3.9, PyTorch 1.9+, and PyTorch Geometric 2.0+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch-scatter` and `torch-sparse` need wheels matching your PyTorch /
CUDA combination — see the
[PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Datasets

This repository **does not ship the datasets**. Place them under
`dataset/` and `dataset_/` as expected by the upstream loaders:

- **Synthetic (GLASS paper, ICLR 2022).** `density`, `coreness`,
  `component`, `cut_ratio`. Available from the
  [official GLASS Dropbox / PKU mirror](https://github.com/Xi-yuanWang/GLASS).
  Place the unzipped folders under `dataset_/`.
- **Real-world (SubGNN).** `em_user`, `hpo_metab`, `hpo_neuro`, `ppi_bp`.
  See the same upstream link. Place under `dataset/`.
- **Elliptic2 (Bellei et al. 2024).** Available from the
  [Elliptic2 release](https://github.com/MITIBMxGraph/Elliptic2). The
  preprocessing step below builds the 2-hop induction expected by
  `run_ablation_elliptic2.py`.

Final layout once datasets are placed:

```
.
├── dataset/                            # real-world (SubGNN)
│   ├── em_user/  hpo_metab/  hpo_neuro/  ppi_bp/
└── dataset_/                           # synthetic + Elliptic2
    ├── component/  coreness/  cut_ratio/  density/
    └── elliptic2/
        ├── raw/                        # Bellei et al. release files
        └── processed/                  # built by preprocess_elliptic2*.py
```

## Reproducing the synthetic experiments

The four GLASS-paper synthetic benchmarks at the upstream defaults:

```bash
python GLASSTest.py --use_one --use_seed --use_maxzeroone \
    --repeat 10 --device 0 --dataset density
# repeat for: component, coreness, cut_ratio
```

To run the **sampler characterisation** that Chapter 5 of the report
discusses (per-source NeighborSampler, GraphSAINT-RW, EgoNet):

```bash
python run_neighbor_sampling_experiments.py --dataset density
python run_graphsaint_experiments.py        --dataset density
python run_ego_experiments.py               --dataset density
```

Each driver writes structured logs to `logs/Sampling/`,
`logs/GraphSAINT/`, `logs/EgoNet/` respectively, plus a
`{dataset}_summary.txt` table per run.

## Reproducing the Elliptic2 ablation

Step 1 — preprocess. This produces the 2-hop induction `.pt` and the
edge-feature stats:

```bash
python preprocess_elliptic2.py
python preprocess_elliptic2_edges.py
```

Step 2 — run a single cell:

```bash
CUDA_VISIBLE_DEVICES=0 python run_ablation_elliptic2.py \
    --config config/ablation/A8.yml \
    --log_dir logs/ablations/A8 \
    --seed 0
```

Step 3 — sweep all twelve cells over seven seeds (a representative
example sweep script lives at `scripts/run_all_experiments.sh`).

The driver writes one JSONL line per epoch with training and
validation metrics, plus a final `{"event": "final", ...}` row. Use
`scripts/aggregate_seeds.py` to roll those up into per-cell mean ± SE
summaries.

## Logs and outputs

`logs/` is `.gitignored`. The repository ships configs and code only;
generated artefacts (per-epoch JSONLs, per-cell summary JSONs, run
directories) live outside version control.

## Citing

If you use the per-batch sampler, the sampler characterisation, or the
ablation design, please cite this report alongside the original GLASS
paper:

```bibtex
@mastersthesis{agrawal2026practical,
  title  = {Practical Subgraph Learning for Anti-Money-Laundering on Blockchain Graphs},
  author = {Karan Agrawal},
  school = {Indian Institute of Technology Madras},
  year   = {2026},
  type   = {{Master of Technology Project Report}}
}

@inproceedings{wang2022glass,
  title     = {{GLASS}: {GNN} with Labeling Tricks for Subgraph Representation Learning},
  author    = {Xiyuan Wang and Muhan Zhang},
  booktitle = {International Conference on Learning Representations},
  year      = {2022},
  url       = {https://openreview.net/forum?id=XLxhEjKNbXj}
}
```

## Acknowledgements

- Wang & Zhang (ICLR 2022) for the GLASS implementation this project
  builds on.
- Bellei et al. (KDD 2024) for releasing Elliptic2.
- Song et al. (ICAIF 2024) for the per-batch / one-layer recipe that
  motivated the engineering of the per-batch sampler.
- Prof. John Augustine (CSE, IIT Madras) for guiding this project.
