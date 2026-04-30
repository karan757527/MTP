import torch


class BaseSampler:
    """
    Base class for all graph samplers used in GLASSConv.

    Contract:
        - Input adj is a torch.sparse_coo_tensor, already normalized by buildAdj.
        - Output must also be a torch.sparse_coo_tensor on the same device.
        - self.adj must NEVER be modified — always work on a copy.
        - During inference (training=False), return full adj for stable predictions.
    """

    def sample(self, adj, training=True):
        raise NotImplementedError

    @staticmethod
    def _renorm(aggr, full_row, new_row, new_val, n_node, device):
        """
        Recompute edge weights after sampling.

        sum:  importance sampling correction — multiply by full_deg/sampled_deg.
        mean: recompute as 1/sampled_deg from scratch.
        gcn:  approximate — rescale source side only.
        """
        ones_full = torch.ones(full_row.shape[0], device=device)
        ones_new  = torch.ones(new_row.shape[0],  device=device)

        if aggr == "sum":
            full_deg = torch.zeros(n_node, device=device)
            full_deg.scatter_add_(0, full_row, ones_full)

            sampled_deg = torch.zeros(n_node, device=device)
            sampled_deg.scatter_add_(0, new_row, ones_new)
            sampled_deg.clamp_(min=1.0)

            return new_val * full_deg[new_row] / sampled_deg[new_row]

        elif aggr == "mean":
            sampled_deg = torch.zeros(n_node, device=device)
            sampled_deg.scatter_add_(0, new_row, ones_new)
            sampled_deg[sampled_deg < 0.5] = 1.0
            return (1.0 / sampled_deg)[new_row]

        elif aggr == "gcn":
            full_deg = torch.zeros(n_node, device=device)
            full_deg.scatter_add_(0, full_row, ones_full)
            full_deg[full_deg < 0.5] = 1.0

            sampled_deg = torch.zeros(n_node, device=device)
            sampled_deg.scatter_add_(0, new_row, ones_new)
            sampled_deg[sampled_deg < 0.5] = 1.0

            return new_val * torch.sqrt(full_deg[new_row] / sampled_deg[new_row])

        else:
            raise NotImplementedError(
                f"Renormalization not implemented for aggr='{aggr}'. "
                f"Supported: 'mean', 'sum', 'gcn'."
            )


class NoSampler(BaseSampler):
    """
    Baseline sampler — no sampling at all.
    Returns the full adjacency matrix unchanged.
    """

    def sample(self, adj, training=True):
        return adj


class NeighborSampler(BaseSampler):
    """
    For each node, randomly keep only k neighbors during message passing.
    Fully vectorized — single argsort, no double argsort, no scatter_reduce_.

    Core idea:
        Build a float sort key that encodes both node and score in one value:

            key[i] = row[i] + (1.0 - score[i])

        Since row[i] is an integer in [0, n_node) and score[i] is in [0, 1),
        (1.0 - score[i]) occupies the fractional part (0, 1].

        A single argsort on key sorts primarily by node (integer part),
        and within each node by score descending (lower fractional part =
        higher score = better neighbor comes first).

        After sorting, use diff-based group detection to assign within-group
        ranks and keep only the top-k edges per node.

    Args:
        k:    int, number of neighbors to keep per node.
        aggr: str, must match the aggr used in buildAdj ("mean", "sum", "gcn").
    """

    def __init__(self, k, aggr="mean"):
        self.k = k
        self.aggr = aggr

    def sample(self, adj, training=True):
        if not training:
            return adj
        return self._sample_and_renorm(adj)

    def _sample_and_renorm(self, adj):
        adj = adj.coalesce()
        row, col = adj.indices()   # [2, num_edges]
        val = adj.values()         # [num_edges]
        n_node = adj.shape[0]
        num_edges = row.shape[0]
        device = adj.device

        # --- Step 1: single sort by float key ---
        scores = torch.rand(num_edges, device=device)
        key    = row.float() + (1.0 - scores)
        order  = torch.argsort(key)

        row_final = row[order]
        col_final = col[order]
        val_final = val[order]

        # --- Step 2: within-group rank via diff ---
        positions = torch.arange(num_edges, device=device)

        start_flag = torch.zeros(num_edges, dtype=torch.long, device=device)
        start_flag[0] = 1
        if num_edges > 1:
            start_flag[1:] = (row_final[1:] != row_final[:-1]).long()

        group_id  = torch.cumsum(start_flag, dim=0) - 1
        first_pos = positions[start_flag == 1]
        ranks     = positions - first_pos[group_id]

        # --- Step 3: keep top-k per node ---
        keep_mask = ranks < self.k
        if not hasattr(self, '_logged_coverage'):
            self._logged_coverage = True
            kept = keep_mask.sum().item()
            total = num_edges
            print(f"[EdgeCoverage] kept={kept}/{total} "
                f"({kept/total*100:.1f}% of edges) "
                f"reduction={1-kept/total:.3f}")

        new_row = row_final[keep_mask]
        new_col = col_final[keep_mask]
        new_val = val_final[keep_mask]

        # --- Step 4: renormalize ---
        new_val = BaseSampler._renorm(self.aggr, row, new_row, new_val, n_node, device)

        return torch.sparse_coo_tensor(
            torch.stack([new_row, new_col]),
            new_val,
            adj.shape,
            device=device
        ).coalesce()


class GraphSAINTSampler(BaseSampler):
    """
    GraphSAINT random walk sampler (Zeng et al., 2020).

    Algorithm:
        1. Sample num_roots root nodes uniformly at random.
        2. From each root, take a random walk of walk_len steps, following
           outgoing edges chosen uniformly at random.
        3. Keep ALL edges in the induced subgraph of the visited node set.

    Key difference from NeighborSampler:
        NeighborSampler: per-source-node, keeps exactly k outgoing edges.
        GraphSAINTSampler: samples a connected node set via random walks,
                           then keeps ALL edges within that node set
                           (including inter-neighbor edges / triangles).

    Args:
        num_roots: int — number of random walk starting nodes per forward pass.
        walk_len:  int — number of steps per random walk (default 2).
        aggr:      str — aggregation type for renormalization ('mean','sum','gcn').
    """

    def __init__(self, num_roots, walk_len=2, aggr="mean"):
        self.num_roots = num_roots
        self.walk_len  = walk_len
        self.aggr      = aggr

    def sample(self, adj, training=True):
        if not training:
            return adj
        return self._sample_and_renorm(adj)

    def _sample_and_renorm(self, adj):
        adj = adj.coalesce()          # guarantees row is sorted
        row, col = adj.indices()
        val  = adj.values()
        n_node    = adj.shape[0]
        num_edges = row.shape[0]
        device    = adj.device

        # --- Build CSR row_ptr for O(1) neighbor lookup ---
        # After coalesce(), row is sorted so cumsum gives correct row_ptr.
        ones    = torch.ones(num_edges, dtype=torch.long, device=device)
        deg     = torch.zeros(n_node, dtype=torch.long, device=device)
        deg.scatter_add_(0, row, ones)
        row_ptr = torch.zeros(n_node + 1, dtype=torch.long, device=device)
        row_ptr[1:] = deg.cumsum(0)

        # --- Sample root nodes ---
        num_roots = min(self.num_roots, n_node)
        current   = torch.randperm(n_node, device=device)[:num_roots]

        visited = torch.zeros(n_node, dtype=torch.bool, device=device)
        visited[current] = True

        # --- Random walks ---
        for _ in range(self.walk_len):
            start   = row_ptr[current]            # start offset in col[] for each walker
            end     = row_ptr[current + 1]        # exclusive end
            deg_cur = end - start                 # out-degree of each walker

            has_nbr = deg_cur > 0
            # Random offset in [0, deg_cur): clamp handles isolated nodes safely
            rand_off = (torch.rand(num_roots, device=device) * deg_cur.float()).long()
            rand_off = rand_off.clamp(max=(deg_cur - 1).clamp(min=0))

            edge_idx        = start + rand_off    # position in col[]
            next_nodes      = current.clone()
            if has_nbr.any():
                next_nodes[has_nbr] = col[edge_idx[has_nbr]]
            current = next_nodes
            visited[current] = True

        # --- Induced subgraph: keep edges where BOTH endpoints were visited ---
        keep_mask = visited[row] & visited[col]

        if not hasattr(self, '_logged_coverage'):
            self._logged_coverage = True
            kept  = keep_mask.sum().item()
            total = num_edges
            print(f"[EdgeCoverage] kept={kept}/{total} "
                  f"({kept/total*100:.1f}% of edges) "
                  f"reduction={1-kept/total:.3f}")

        new_row = row[keep_mask]
        new_col = col[keep_mask]
        new_val = val[keep_mask]

        new_val = BaseSampler._renorm(self.aggr, row, new_row, new_val, n_node, device)

        return torch.sparse_coo_tensor(
            torch.stack([new_row, new_col]),
            new_val,
            adj.shape,
            device=device
        ).coalesce()


class EgoNetSampler(BaseSampler):
    """
    Ego-graph (ego-network) sampler.

    Algorithm:
        1. Sample num_seeds seed nodes uniformly at random.
        2. Expand each seed by exactly 1 hop: collect all direct neighbors.
        3. Keep ALL edges in the induced subgraph of
           {seeds} ∪ {all neighbors of seeds}.

    Key difference from GraphSAINTSampler:
        GraphSAINTSampler: follows random walk paths — can drift far from roots,
                           samples a path-shaped connected component.
        EgoNetSampler: always expands exactly 1 hop around each seed — produces
                       dense star-shaped ego neighborhoods, preserves local
                       clustering (triangles) within each ego graph.

    Args:
        num_seeds: int — number of ego-center nodes sampled per forward pass.
        aggr:      str — aggregation type for renormalization ('mean','sum','gcn').
    """

    def __init__(self, num_seeds, aggr="mean"):
        self.num_seeds = num_seeds
        self.aggr      = aggr

    def sample(self, adj, training=True):
        if not training:
            return adj
        return self._sample_and_renorm(adj)

    def _sample_and_renorm(self, adj):
        adj = adj.coalesce()
        row, col = adj.indices()
        val  = adj.values()
        n_node    = adj.shape[0]
        num_edges = row.shape[0]
        device    = adj.device

        # --- Sample seed nodes ---
        num_seeds = min(self.num_seeds, n_node)
        seeds     = torch.randperm(n_node, device=device)[:num_seeds]

        visited = torch.zeros(n_node, dtype=torch.bool, device=device)
        visited[seeds] = True

        # --- 1-hop expansion: mark all neighbors of seeds as visited ---
        # Edge (u→v): if u is a seed, v joins the ego neighborhood.
        seed_edge_mask    = visited[row]          # edges whose source is a seed
        visited[col[seed_edge_mask]] = True       # destinations become members

        # --- Induced subgraph: keep edges where BOTH endpoints are members ---
        keep_mask = visited[row] & visited[col]

        if not hasattr(self, '_logged_coverage'):
            self._logged_coverage = True
            kept  = keep_mask.sum().item()
            total = num_edges
            print(f"[EdgeCoverage] kept={kept}/{total} "
                  f"({kept/total*100:.1f}% of edges) "
                  f"reduction={1-kept/total:.3f}")

        new_row = row[keep_mask]
        new_col = col[keep_mask]
        new_val = val[keep_mask]

        new_val = BaseSampler._renorm(self.aggr, row, new_row, new_val, n_node, device)

        return torch.sparse_coo_tensor(
            torch.stack([new_row, new_col]),
            new_val,
            adj.shape,
            device=device
        ).coalesce()
