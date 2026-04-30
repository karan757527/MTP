import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from .utils import pad2batch
from .samplers import BaseSampler, NoSampler, NeighborSampler, GraphSAINTSampler, EgoNetSampler


def _build_norm(norm_type: str, channels: int):
    '''
    Factory for normalization layers used by EmbZGConv.
    norm_type: 'graph' -> GraphNorm, 'layer' -> LayerNorm, 'batch' -> BatchNorm1d.
    '''
    nt = (norm_type or "graph").lower()
    if nt == "graph":
        return GraphNorm(channels)
    if nt == "layer":
        return nn.LayerNorm(channels)
    if nt == "batch":
        return nn.BatchNorm1d(channels)
    raise ValueError(f"unknown norm_type: {norm_type!r}")


class Seq(nn.Module):
    ''' 
    An extension of nn.Sequential. 
    Args: 
        modlist an iterable of modules to add.
    '''
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


class MLP(nn.Module):
    '''
    Multi-Layer Perception.
    Args:
        tail_activation: whether to use activation function at the last layer.
        activation: activation function.
        gn: whether to use GraphNorm layer.
    '''
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 dropout=0,
                 tail_activation=False,
                 activation=nn.ReLU(inplace=True),
                 gn=False):
        super().__init__()
        modlist = []
        self.seq = None
        if num_layers == 1:
            modlist.append(nn.Linear(input_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)
        else:
            modlist.append(nn.Linear(input_channels, hidden_channels))
            for _ in range(num_layers - 2):
                if gn:
                    modlist.append(GraphNorm(hidden_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
                modlist.append(nn.Linear(hidden_channels, hidden_channels))
            if gn:
                modlist.append(GraphNorm(hidden_channels))
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout, inplace=True))
            modlist.append(activation)
            modlist.append(nn.Linear(hidden_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)

    def forward(self, x):
        return self.seq(x)


def buildAdj(edge_index, edge_weight, n_node: int, aggr: str):
    '''
        Calculating the normalized adjacency matrix.
        Args:
            n_node: number of nodes in graph.
            aggr: the aggregation method, can be "mean", "sum" or "gcn".
        '''
    adj = torch.sparse_coo_tensor(edge_index,
                                  edge_weight,
                                  size=(n_node, n_node))
    deg = torch.sparse.sum(adj, dim=(1, )).to_dense().flatten()
    deg[deg < 0.5] += 1.0
    if aggr == "mean":
        deg = 1.0 / deg
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "sum":
        return torch.sparse_coo_tensor(edge_index,
                                       edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "gcn":
        deg = torch.pow(deg, -0.5)
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight *
                                       deg[edge_index[1]],
                                       size=(n_node, n_node))
    else:
        raise NotImplementedError


class GLASSConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for GLASS.
    We use different parameters to transform the features of node with different labels individually, and mix them.
    Args:
        aggr: the aggregation method.
        z_ratio: the ratio to mix the transformed features.
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean",
                 z_ratio=0.8,
                 dropout=0.2,
                 sampler: BaseSampler = None,
                 norm_type: str = "graph",
                 edge_attr_dim: int = 0,
                 edge_mlp_hidden: int = 32,
                 edge_mode: str = "scalar"):
        super().__init__()
        self.trans_fns = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels)
        ])
        self.comb_fns = nn.ModuleList([
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Linear(in_channels + out_channels, out_channels)
        ])
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.norm_type = norm_type
        self.gn = _build_norm(norm_type, out_channels)
        self.z_ratio = z_ratio
        self.dropout = dropout
        # If no sampler provided, default to NoSampler (full adj, baseline behaviour).
        self.sampler = sampler if sampler is not None else NoSampler()
        # Edge-feature pathway. Two variants, both no-op when edge_attr_dim=0:
        #   "scalar"  — MLP(95→h→1) + softplus; multiplies the default edge_weight
        #               before buildAdj. Cheap gate. (Approach A, A9 cell.)
        #   "additive" — Linear(95→out_channels); per-edge projected feature is
        #               scatter_add'd into the destination node, then added to
        #               the adj@x message. Richer signal per edge. (Approach B,
        #               A9b cell.)
        self.edge_attr_dim = edge_attr_dim
        self.edge_mode = edge_mode if edge_attr_dim > 0 else "scalar"
        if edge_attr_dim > 0 and edge_mode == "scalar":
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_attr_dim, edge_mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(edge_mlp_hidden, 1),
            )
            self.edge_proj = None
        elif edge_attr_dim > 0 and edge_mode == "additive":
            self.edge_mlp = None
            self.edge_proj = nn.Linear(edge_attr_dim, out_channels)
        else:
            self.edge_mlp = None
            self.edge_proj = None
        self.reset_parameters()

    def reset_parameters(self):
        for _ in self.trans_fns:
            _.reset_parameters()
        for _ in self.comb_fns:
            _.reset_parameters()
        if hasattr(self.gn, "reset_parameters"):
            self.gn.reset_parameters()
        if self.edge_mlp is not None:
            for m in self.edge_mlp:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        if self.edge_proj is not None:
            self.edge_proj.reset_parameters()

    def forward(self, x_, edge_index, edge_weight, mask, edge_attr=None):
        # Approach A (edge_mode="scalar"): replace edge_weight with a learned
        # positive scalar gate from the edge_mlp before the adj is built.
        # Softplus keeps weights positive so buildAdj's degree normalisation
        # stays well-defined.
        if edge_attr is not None and self.edge_mlp is not None:
            w = self.edge_mlp(edge_attr.float())        # (E, 1)
            edge_weight = F.softplus(w).squeeze(-1)     # (E,)
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
        # Get adj for this forward pass — sampled during training, full during inference.
        # self.adj is never modified; adj is a temporary variable for this pass only.
        adj = self.sampler.sample(self.adj, training=self.training)
        # transform node features with different parameters individually.
        x1 = self.activation(self.trans_fns[1](x_))
        x0 = self.activation(self.trans_fns[0](x_))
        # mix transformed feature.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        # pass messages using (possibly sampled) adj.
        x = adj @ x
        # Approach B (edge_mode="additive"): inject per-destination-node edge
        # messages. Each edge contributes W_e·edge_attr to its dst node's
        # pre-norm representation. Complements, does not replace, adj@x — so
        # edges carry signal independent of what's flowing through nodes.
        if edge_attr is not None and self.edge_proj is not None:
            ef = self.edge_proj(edge_attr.float())      # (E, out_channels)
            dst = edge_index[1]
            em = x.new_zeros(x.shape[0], ef.shape[1])
            idx = dst.unsqueeze(1).expand(-1, ef.shape[1])
            em.scatter_add_(0, idx, ef)
            x = x + em
        x = self.gn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat((x, x_), dim=-1)
        # transform node features with different parameters individually.
        x1 = self.comb_fns[1](x)
        x0 = self.comb_fns[0](x)
        # mix transformed feature.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        return x


class EmbZGConv(nn.Module):
    '''
    combination of some GLASSConv layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features. Ignored if input_channels is set.
        input_channels: dimensionality of real-valued input features. When set,
            the input embedding is a Linear(input_channels, hidden) instead of
            the default Embedding(max_deg+1, hidden). Use this for datasets that
            provide real-valued node features (e.g. Elliptic2 raw_emb, 43-dim).
        norm_type: 'graph' (GraphNorm, default), 'layer' (LayerNorm),
            or 'batch' (BatchNorm1d). Applied at both input and per-layer norms.
        conv: the message passing layer we use.
        gn: whether to apply normalization at all.
        jk: whether to use Jumping Knowledge Network.
        num_neighbors: int or list of ints controlling neighbor sampling per layer.
            - None or 0: no sampling (uses NoSampler, full adj).
            - int:       same k for all layers (uniform sampling).
            - list:      per-layer k values, length must equal num_layers.
                         e.g. [15, 10] for a 2-layer model.
            The aggr type is read from kwargs to initialize NeighborSampler correctly.
    '''
    def __init__(self,
                 hidden_channels,
                 output_channels,
                 num_layers,
                 max_deg,
                 dropout=0,
                 activation=nn.ReLU(),
                 conv=GLASSConv,
                 gn=True,
                 jk=False,
                 num_neighbors=None,
                 samplers=None,
                 input_channels=None,
                 norm_type="graph",
                 use_checkpoint=False,
                 edge_attr_dim=0,
                 edge_mlp_hidden=32,
                 edge_mode="scalar",
                 **kwargs):
        super().__init__()
        self.input_channels = input_channels
        self.use_features = input_channels is not None
        self.norm_type = norm_type
        # Gradient checkpointing — enable for large graphs (e.g. Elliptic2)
        # where per-layer activations would otherwise OOM. Each layer recomputes
        # its forward during backward, trading ~1.3x compute for a large memory
        # reduction.
        self.use_checkpoint = use_checkpoint
        self.edge_attr_dim = edge_attr_dim
        self.edge_mode = edge_mode
        if self.use_features:
            self.input_emb = nn.Linear(input_channels, hidden_channels)
        else:
            self.input_emb = nn.Embedding(max_deg + 1,
                                          hidden_channels,
                                          scale_grad_by_freq=False)
        self.emb_gn = _build_norm(norm_type, hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk

        # --- Build per-layer sampler list ---
        # aggr is in kwargs (passed via functools.partial in GLASSTest.py).
        # We read it here to initialise NeighborSampler with the correct aggr.
        aggr = kwargs.get("aggr", "mean")
        # Propagate norm_type + edge_attr_dim to inner GLASSConv layers so
        # all normalisation follows the ablation setting, and each conv
        # builds its own edge_mlp when edge features are in use (A9 cell).
        kwargs["norm_type"] = norm_type
        kwargs["edge_attr_dim"] = edge_attr_dim
        kwargs["edge_mlp_hidden"] = edge_mlp_hidden
        kwargs["edge_mode"] = edge_mode

        

        def make_sampler(k):
            if k is None or k == 0:
                return NoSampler()
            return NeighborSampler(k=k, aggr=aggr)

        if samplers is not None:
            # Pre-built sampler list passed in — use directly (e.g. GraphSAINT, EgoNet).
            assert len(samplers) == num_layers, (
                f"samplers list length ({len(samplers)}) must equal num_layers ({num_layers})."
            )
        elif num_neighbors is None or num_neighbors == 0:
            # No sampling — all layers use NoSampler (baseline behaviour).
            samplers = [NoSampler() for _ in range(num_layers)]
        elif isinstance(num_neighbors, int):
            # Uniform k across all layers.
            samplers = [make_sampler(num_neighbors) for _ in range(num_layers)]
        elif isinstance(num_neighbors, list):
            # Per-layer k values.
            assert len(num_neighbors) == num_layers, (
                "num_neighbors list length must equal num_layers."
            )
            samplers = [make_sampler(k) for k in num_neighbors]
        else:
            raise ValueError(
                f"num_neighbors must be None, int, or list. Got {type(num_neighbors)}."
            )

        # Build conv layers, passing the right sampler to each.
        # sampler is NOT passed via **kwargs — assigned explicitly per layer
        # so each layer gets its own sampler instance with the right k.
        for layer_idx in range(num_layers - 1):
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=hidden_channels,
                     activation=activation,
                     sampler=samplers[layer_idx],
                     **kwargs))
        self.convs.append(
            conv(in_channels=hidden_channels,
                 out_channels=output_channels,
                 activation=activation,
                 sampler=samplers[num_layers - 1],
                 **kwargs))
        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(_build_norm(norm_type, hidden_channels))
            if self.jk:
                self.gns.append(
                    _build_norm(
                        norm_type,
                        output_channels + (num_layers - 1) * hidden_channels))
            else:
                self.gns.append(_build_norm(norm_type, output_channels))
        else:
            self.gns = None
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.input_emb, "reset_parameters"):
            self.input_emb.reset_parameters()
        if hasattr(self.emb_gn, "reset_parameters"):
            self.emb_gn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                if hasattr(gn, "reset_parameters"):
                    gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None, edge_attr=None):
        # z is the node label.
        if z is None:
            mask = (torch.zeros(
                (x.shape[0]), device=x.device) < 0.5).reshape(-1, 1)
        else:
            mask = (z > 0.5).reshape(-1, 1)
        # Input transform: Linear for real features, Embedding for integer.
        if self.use_features:
            x = self.input_emb(x.float())
        else:
            x = self.input_emb(x).reshape(x.shape[0], -1)
        x = self.emb_gn(x)
        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)

        def _run_conv(conv_layer, xi, ei, ew, mk, ea):
            return conv_layer(xi, ei, ew, mk, edge_attr=ea)

        # pass messages at each layer.
        for layer, conv in enumerate(self.convs[:-1]):
            if self.use_checkpoint and self.training:
                x = checkpoint(_run_conv, conv, x, edge_index, edge_weight,
                               mask, edge_attr, use_reentrant=False)
            else:
                x = conv(x, edge_index, edge_weight, mask, edge_attr=edge_attr)
            xs.append(x)
            if not (self.gns is None):
                x = self.gns[layer](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.use_checkpoint and self.training:
            x = checkpoint(_run_conv, self.convs[-1], x, edge_index,
                           edge_weight, mask, edge_attr, use_reentrant=False)
        else:
            x = self.convs[-1](x, edge_index, edge_weight, mask,
                               edge_attr=edge_attr)
        xs.append(x)

        if self.jk:
            x = torch.cat(xs, dim=-1)
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x
        else:
            x = xs[-1]
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x


class PoolModule(nn.Module):
    '''
    Modules used for pooling node embeddings to produce subgraph embeddings.
    Args: 
        trans_fn: module to transfer node embeddings.
        pool_fn: module to pool node embeddings like global_add_pool.
    '''
    def __init__(self, pool_fn, trans_fn=None):
        super().__init__()
        self.pool_fn = pool_fn
        self.trans_fn = trans_fn

    def forward(self, x, batch):
        # The j-th element in batch vector is i if node j is in the i-th subgraph.
        # for example [0,1,0,0,1,1,2,2] means nodes 0,2,3 in subgraph 0, nodes 1,4,5 in subgraph 1, and nodes 6,7 in subgraph 2.
        if self.trans_fn is not None:
            x = self.trans_fn(x)
        return self.pool_fn(x, batch)


class AddPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_add_pool, trans_fn)


class MaxPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_max_pool, trans_fn)


class MeanPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_mean_pool, trans_fn)


class SizePool(AddPool):
    def __init__(self, trans_fn=None):
        super().__init__(trans_fn)

    def forward(self, x, batch):
        if x is not None:
            if self.trans_fn is not None:
                x = self.trans_fn(x)
        x = GraphSizeNorm()(x, batch)
        return self.pool_fn(x, batch)


class GLASS(nn.Module):
    '''
    GLASS model: combine message passing layers and mlps and pooling layers.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''
    def __init__(self, conv: EmbZGConv, preds: nn.ModuleList,
                 pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None, edge_attr=None):
        embs = []
        for _ in range(x.shape[1]):
            emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                            edge_index, edge_weight, z, edge_attr=edge_attr)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        return emb

    def Pool(self, emb, subG_node, pool):
        batch, pos = pad2batch(subG_node)
        emb = emb[pos]
        emb = pool(emb, batch)
        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0,
                edge_attr=None):
        emb = self.NodeEmb(x, edge_index, edge_weight, z, edge_attr=edge_attr)
        emb = self.Pool(emb, subG_node, self.pools[id])
        return self.preds[id](emb)


# models used for producing node embeddings.


class MyGCNConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for pretrained GNNs.
    Args:
        aggr: the aggregation method.
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean"):
        super().__init__()
        self.trans_fn = nn.Linear(in_channels, out_channels)
        self.comb_fn = nn.Linear(in_channels + out_channels, out_channels)
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)

    def reset_parameters(self):
        self.trans_fn.reset_parameters()
        self.comb_fn.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
        x = self.trans_fn(x_)
        x = self.activation(x)
        x = self.adj @ x
        x = self.gn(x)
        x = torch.cat((x, x_), dim=-1)
        x = self.comb_fn(x)
        return x


class EmbGConv(torch.nn.Module):
    '''
    combination of some message passing layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features.
        conv: the message passing layer we use.
        gn: whether to use GraphNorm.
        jk: whether to use Jumping Knowledge Network.
    '''
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 max_deg: int,
                 dropout=0,
                 activation=nn.ReLU(inplace=True),
                 conv=GCNConv,
                 gn=True,
                 jk=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1, hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
        if num_layers > 1:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=hidden_channels,
                     **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(
                    conv(in_channels=hidden_channels,
                         out_channels=hidden_channels,
                         **kwargs))
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=output_channels,
                     **kwargs))
        else:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=output_channels,
                     **kwargs))
        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
        else:
            self.gns = None
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None):
        xs = []
        x = F.dropout(self.input_emb(x.reshape(-1)),
                      p=self.dropout,
                      training=self.training)
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if not (self.gns is None):
                x = self.gns[layer](x)
            xs.append(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(self.convs[-1](x, edge_index, edge_weight))
        if self.jk:
            return torch.cat(xs, dim=-1)
        else:
            return xs[-1]


class EdgeGNN(nn.Module):
    '''
    EdgeGNN model: combine message passing layers and mlps and pooling layers to do link prediction task.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''
    def __init__(self, conv, preds: nn.ModuleList, pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None):
        embs = []
        for _ in range(x.shape[1]):
            emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                            edge_index, edge_weight, z)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        return emb

    def Pool(self, emb, subG_node, pool):
        emb = emb[subG_node]
        emb = torch.mean(emb, dim=1)
        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        emb = self.NodeEmb(x, edge_index, edge_weight, z)
        emb = self.Pool(emb, subG_node, self.pools[id])
        return self.preds[id](emb)