"""
elliptic2_sampler.py — CPU-side CSR + per-batch L-hop subgraph extractor.

Why this file exists
--------------------
The vanilla GLASS pipeline materialises the entire 46.5M-node Elliptic2
background graph on the GPU and runs message passing over the full
(46.5M, H) activation tensor. That OOMs at H=8 on a 49 GB RTX 6000 — even
after every memory optimisation we could think of.

Song et al. (ICAIF 2024, Table 1) report GLASS on the same dataset with
PR-AUC=0.816 using only 6.23 GB on a single 16 GB V100. Their §5.1.3
recipe: "we use only one MPNN layer with neighborhood sampling, which
significantly reduces the training time and the memory requirement"
followed by "we implement data preprocessing to enable faster data
loading". The implementation detail they don't spell out: their sampler
keeps edges on CPU and only puts the per-batch L-hop neighbourhood on
the GPU. The (N, H) activation is then (~1M, H), not (46.5M, H).

This module reproduces that approach. CSR is built once in __init__ and
held in CPU RAM (~3 GB for Elliptic2). Each call to sample() walks the
CSR for L hops from the batch's seed nodes and returns a small,
locally-relabelled subgraph ready to be moved to the GPU.

Public API
----------
    Elliptic2BatchSampler(features, edge_index, pos_pad, y, mask)
        .train_indices() / .val_indices() / .test_indices() -> LongTensor
        .sample(subgraph_idxs, num_layers, num_neighbors) -> dict with:
            x:           (sub_N, D)        float
            edge_index:  (2, sub_E)        long, locally relabelled
            edge_weight: (sub_E,)          float
            pos:         (B, max_sg_size)  long, locally relabelled, -1 pad
            z:           (sub_N,)          long, MaxZOZ on local node space
            y:           (B,)              float
            sub_N, sub_E:                  int (for logging)

Notes on correctness
--------------------
* GLASS uses `adj @ x`, no self-loop. We collect ALL edges walked during
  L-hop expansion. After L hops, we have the full L-hop induced subgraph
  on the seed nodes — i.e. every edge needed to compute messages out to
  depth L.
* `pos` (the per-subgraph node id matrix) is remapped from global to
  local node ids using a cached lookup buffer. Padding -1 is preserved.
* When num_neighbors > 0 we sample neighbours WITH REPLACEMENT for
  vectorisation speed. For Elliptic2's avg degree ≈ 8.2 vs typical k=25,
  this is essentially "take them all" and the bias is negligible.
"""
import torch


class Elliptic2BatchSampler:
    def __init__(self, features, edge_index, pos_pad, y, mask, edge_attr=None):
        """
        features:   (N, D)              float CPU tensor
        edge_index: (2, E)              long  CPU tensor — directed (src, dst)
        pos_pad:    (num_subgraphs, S)  long  CPU tensor, -1 padding
        y:          (num_subgraphs,)    float CPU tensor
        mask:       (num_subgraphs,)    long  CPU tensor (0=trn, 1=val, 2=tst)
        edge_attr:  (E, F)              optional CPU tensor aligned with
                                        edge_index. When provided, sample()
                                        returns a per-batch edge_attr_local
                                        tensor aligned with edge_index_local.
                                        Used by the A9 edge-features cell.
        """
        assert not features.is_cuda and not edge_index.is_cuda
        self.features = features.contiguous()
        self.pos_pad = pos_pad.contiguous()
        self.y = y.contiguous()
        self.mask = mask.contiguous()
        self.N = int(features.shape[0])
        self.D = int(features.shape[1])

        # --- build CSR ---
        # Sort edges by source so each node's out-neighbours are contiguous in
        # `col`. rowptr[u:u+2] then indexes u's slice.
        print(f"[sampler] building CSR: N={self.N:,} E={edge_index.shape[1]:,}",
              flush=True)
        src = edge_index[0]
        order = torch.argsort(src)
        self.col = edge_index[1][order].contiguous()
        deg = torch.bincount(src, minlength=self.N)
        rowptr = torch.zeros(self.N + 1, dtype=torch.long)
        rowptr[1:] = deg.cumsum(0)
        self.rowptr = rowptr.contiguous()
        if edge_attr is not None:
            assert edge_attr.shape[0] == edge_index.shape[1], (
                f"edge_attr rows {edge_attr.shape[0]} != "
                f"edge_index cols {edge_index.shape[1]}"
            )
            # Reorder edge_attr with the same `order` that produced `col`
            # so edge_attr_sorted[i] aligns with the edge (src_i, col[i]).
            self.edge_attr_sorted = edge_attr[order].contiguous()
            print(f"[sampler] edge_attr aligned: "
                  f"{tuple(self.edge_attr_sorted.shape)} "
                  f"dtype={self.edge_attr_sorted.dtype}", flush=True)
        else:
            self.edge_attr_sorted = None
        del order, deg
        print(f"[sampler] CSR ready: rowptr {tuple(self.rowptr.shape)} "
              f"col {tuple(self.col.shape)}", flush=True)

        # Reusable global->local id lookup buffer. We touch only sub_N slots
        # per call and reset them at the end, so we never re-allocate the
        # (N,) buffer (~360 MB) per batch.
        self._local_id_buf = torch.full((self.N,), -1, dtype=torch.long)

    def train_indices(self):
        return torch.nonzero(self.mask == 0, as_tuple=False).flatten()

    def val_indices(self):
        return torch.nonzero(self.mask == 1, as_tuple=False).flatten()

    def test_indices(self):
        return torch.nonzero(self.mask == 2, as_tuple=False).flatten()

    def _expand_one_hop(self, seeds, num_neighbors, return_gather_idx=False):
        """
        seeds: (S,) long
        num_neighbors: int — k cap, 0 means take all out-neighbours
        return_gather_idx: if True, also return the CSR positions of each
            kept edge so the caller can look up the corresponding edge_attr
            from self.edge_attr_sorted. Defaults to False for backward
            compatibility with cells that don't use edge features.

        Returns:
            edge_src: (M,) long — global ids of edge sources (in seeds, repeats)
            edge_dst: (M,) long — global ids of edge destinations
            [gather_idx: (M,) long — only when return_gather_idx=True]
        """
        starts = self.rowptr[seeds]
        ends = self.rowptr[seeds + 1]
        degs = (ends - starts).long()
        total_full = int(degs.sum().item())
        if total_full == 0:
            empty = torch.empty(0, dtype=torch.long)
            if return_gather_idx:
                return empty, empty, empty
            return empty, empty

        if num_neighbors and num_neighbors > 0:
            # Per-seed cap with WITH-REPLACEMENT sampling (vectorised, O(M)).
            # For Elliptic2 (avg deg ≈ 8.2) and k=25, keep_per_seed = degs in
            # almost all cases anyway, so the bias is negligible.
            keep_per_seed = torch.minimum(
                degs, torch.full_like(degs, num_neighbors)
            )
            total_kept = int(keep_per_seed.sum().item())
            if total_kept == 0:
                empty = torch.empty(0, dtype=torch.long)
                if return_gather_idx:
                    return empty, empty, empty
                return empty, empty
            edge_src = torch.repeat_interleave(seeds, keep_per_seed)
            # Random offset in [0, deg[i]) for each kept slot of seed i.
            row_starts = torch.repeat_interleave(starts, keep_per_seed)
            row_degs = torch.repeat_interleave(degs.float(), keep_per_seed)
            rand_offsets = (torch.rand(total_kept) * row_degs).long()
            gather_idx = row_starts + rand_offsets
            edge_dst = self.col[gather_idx]
        else:
            # No cap — take every out-edge of every seed (vectorised).
            edge_src = torch.repeat_interleave(seeds, degs)
            row_starts = torch.repeat_interleave(starts, degs)
            within_row = torch.arange(total_full, dtype=torch.long) - \
                torch.repeat_interleave(degs.cumsum(0) - degs, degs)
            gather_idx = row_starts + within_row
            edge_dst = self.col[gather_idx]

        if return_gather_idx:
            return edge_src, edge_dst, gather_idx
        return edge_src, edge_dst

    def sample(self, subgraph_idxs, num_layers, num_neighbors):
        """
        subgraph_idxs: (B,) long  — indices into pos_pad for this batch
        num_layers:    int        — receptive-field depth (1 or 2)
        num_neighbors: int        — k cap per node per hop (0 = no cap)
        """
        # --- 1. seed nodes = the union of all subgraph nodes in the batch ---
        batch_pos = self.pos_pad[subgraph_idxs]            # (B, S)
        valid_pos = batch_pos[batch_pos >= 0]
        seed_nodes = torch.unique(valid_pos)               # (S0,)

        # --- 2. L-hop expansion. Collect every edge walked. ---
        want_attr = self.edge_attr_sorted is not None
        all_src = []
        all_dst = []
        all_gather = []   # CSR positions; only populated when want_attr
        frontier = seed_nodes
        seen = seed_nodes
        for _ in range(num_layers):
            if want_attr:
                e_src, e_dst, e_gi = self._expand_one_hop(
                    frontier, num_neighbors, return_gather_idx=True
                )
            else:
                e_src, e_dst = self._expand_one_hop(frontier, num_neighbors)
            if e_src.numel() > 0:
                all_src.append(e_src)
                all_dst.append(e_dst)
                if want_attr:
                    all_gather.append(e_gi)
                # Next hop expands from the cumulative receptive field. This
                # over-collects (we re-expand from prev seeds) but stays correct.
                seen = torch.cat([seen, e_dst]).unique()
            frontier = seen

        sub_nodes = seen                                   # (sub_N,)
        sub_N = int(sub_nodes.shape[0])

        # --- 3. global -> local id remap (use cached buffer) ---
        local_id = self._local_id_buf
        local_id[sub_nodes] = torch.arange(sub_N, dtype=torch.long)

        # --- 4. relabel edges, then dedupe (u, v) pairs ---
        # For num_layers >= 2 the expansion loop re-walks out-edges of
        # already-visited nodes, producing duplicate edges that would
        # double-count in buildAdj's degree sum. torch.unique on (E, 2)
        # removes them. For L=1 there are no dupes and this is a fast no-op.
        edge_attr_local = None
        if all_src:
            global_src = torch.cat(all_src)
            global_dst = torch.cat(all_dst)
            local_src = local_id[global_src]
            local_dst = local_id[global_dst]
            valid = (local_src >= 0) & (local_dst >= 0)
            local_src = local_src[valid]
            local_dst = local_dst[valid]
            ei_t = torch.stack([local_src, local_dst], dim=1)  # (E, 2)
            if want_attr:
                gi = torch.cat(all_gather)[valid]
                ea_per_edge = self.edge_attr_sorted[gi]         # (E, F)
                # Dedup with return_inverse so we can collapse edge_attr too.
                unique_ei, inverse = torch.unique(
                    ei_t, dim=0, return_inverse=True
                )
                U = unique_ei.size(0)
                ea_unique = torch.empty(
                    U, self.edge_attr_sorted.shape[1],
                    dtype=self.edge_attr_sorted.dtype,
                )
                # Scatter: for duplicate edges (same underlying undirected
                # edge walked at multiple hops) edge_attr values are
                # identical, so last-write-wins is safe.
                ea_unique[inverse] = ea_per_edge
                edge_index_local = unique_ei.t().contiguous()
                edge_attr_local = ea_unique
            else:
                ei_t = torch.unique(ei_t, dim=0)
                edge_index_local = ei_t.t().contiguous()
        else:
            edge_index_local = torch.empty(2, 0, dtype=torch.long)
            if want_attr:
                edge_attr_local = torch.empty(
                    0, self.edge_attr_sorted.shape[1],
                    dtype=self.edge_attr_sorted.dtype,
                )

        edge_weight_local = torch.ones(
            edge_index_local.shape[1], dtype=torch.float
        )

        # --- 5. local features and pos ---
        x_local = self.features[sub_nodes]                 # (sub_N, D)
        pos_local = local_id[batch_pos.clamp(min=0)]
        pos_local[batch_pos < 0] = -1

        # --- 6. local MaxZOZ mask ---
        z_local = torch.zeros(sub_N, dtype=torch.long)
        valid_local = pos_local[pos_local >= 0]
        z_local[valid_local] = 1

        # --- 7. reset local_id buffer (only the slots we touched) ---
        local_id[sub_nodes] = -1

        out = {
            "x": x_local,
            "edge_index": edge_index_local,
            "edge_weight": edge_weight_local,
            "pos": pos_local,
            "z": z_local,
            "y": self.y[subgraph_idxs],
            "sub_N": sub_N,
            "sub_E": int(edge_index_local.shape[1]),
        }
        if edge_attr_local is not None:
            out["edge_attr"] = edge_attr_local
        return out
