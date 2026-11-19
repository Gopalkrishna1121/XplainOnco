"""Graph branch (paper Section 3.3, Eq. 1): a GAT with one learnable gate per STRING edge.

h_v^(l+1) = ELU( sum_{u in N(v) U {v}} m_uv * alpha_vu * W h_u ),   m_uv = sigmoid(omega_uv), m_vv = 1

Written in plain PyTorch on an edge list so it has no extra dependencies and so the
gates can be overridden at call time (needed for edge-level Integrated Gradients).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def undirected_to_directed(edge_und: torch.Tensor, n_nodes: int):
    """Build a directed edge list (both directions + self loops) from undirected edges [2, E].

    Returns src, dst, und_idx. For self loops und_idx = -1 (their gate is fixed at 1).
    """
    u, v = edge_und
    e = edge_und.shape[1]
    ar = torch.arange(e, dtype=torch.long)
    loops = torch.arange(n_nodes, dtype=torch.long)
    src = torch.cat([u, v, loops])
    dst = torch.cat([v, u, loops])
    und_idx = torch.cat([ar, ar, torch.full((n_nodes,), -1, dtype=torch.long)])
    return src, dst, und_idx


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.lin = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, head_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(heads * head_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, h, src, dst, m_edge):
        """h: [B, n, in]; m_edge: [E'] or [B, E'] gate value for each directed edge (self loops = 1)."""
        bsz, n, _ = h.shape
        wh = self.lin(h).view(bsz, n, self.heads, self.head_dim)
        a_s = (wh * self.att_src).sum(-1)  # [B, n, H]
        a_d = (wh * self.att_dst).sum(-1)
        e = F.leaky_relu(a_s[:, src] + a_d[:, dst], 0.2)  # [B, E', H]
        e_max = torch.full((bsz, n, self.heads), -1e30, dtype=e.dtype, device=e.device)
        e_max = e_max.scatter_reduce(1, dst.view(1, -1, 1).expand_as(e), e.detach(), "amax", include_self=True)
        ex = torch.exp(e - e_max[:, dst])
        denom = torch.zeros(bsz, n, self.heads, dtype=e.dtype, device=e.device).index_add(1, dst, ex)
        alpha = self.drop(ex / (denom[:, dst] + 1e-16))  # [B, E', H]
        m = m_edge if m_edge.dim() == 2 else m_edge.unsqueeze(0)
        msg = (alpha * m.unsqueeze(-1)).unsqueeze(-1) * wh[:, src]  # [B, E', H, C]
        out = torch.zeros(bsz, n, self.heads, self.head_dim, dtype=h.dtype, device=h.device).index_add(1, dst, msg)
        return F.elu(out.reshape(bsz, n, -1) + self.bias)


class GraphBranch(nn.Module):
    """Three GAT layers (4 heads x 16 = 64 units), shared edge gates, attention pooling."""

    def __init__(self, edge_und: torch.Tensor, n_nodes: int, in_dim: int = 3, hidden: int = 64,
                 heads: int = 4, layers: int = 3, gate_init: float = 0.9, dropout: float = 0.1):
        super().__init__()
        assert hidden % heads == 0
        self.n_nodes, self.n_edges = n_nodes, edge_und.shape[1]
        src, dst, und_idx = undirected_to_directed(edge_und, n_nodes)
        self.register_buffer("src", src, persistent=False)
        self.register_buffer("dst", dst, persistent=False)
        self.register_buffer("und_idx", und_idx, persistent=False)
        self.register_buffer("edge_und", edge_und.clone(), persistent=False)
        # gate logits omega_uv, initialised so that sigmoid(omega) = gate_init (paper: 0.9)
        omega0 = math.log(gate_init / (1.0 - gate_init))
        self.omega = nn.Parameter(torch.full((self.n_edges,), omega0))
        dims = [in_dim] + [hidden] * layers
        self.layers = nn.ModuleList(GATLayer(dims[i], heads, hidden // heads, dropout) for i in range(layers))
        self.pool_proj = nn.Linear(hidden, hidden)
        self.pool_score = nn.Linear(hidden, 1)
        self.out_dim = hidden

    def gates(self) -> torch.Tensor:
        """m_uv = sigmoid(omega_uv), one value per undirected STRING edge."""
        return torch.sigmoid(self.omega)

    def _directed_gates(self, gates: torch.Tensor) -> torch.Tensor:
        """Expand undirected gates ([E] or [B, E]) to the directed edge list, appending 1s for self loops."""
        loops = self.und_idx < 0
        idx = self.und_idx.clamp(min=0)
        g = gates[..., idx]
        ones = torch.ones_like(g)
        return torch.where(loops, ones, g)

    def forward(self, x: torch.Tensor, gates: torch.Tensor | None = None):
        """x: [B, n, 3]. gates: optional override ([E] or [B, E]); defaults to the learned gates."""
        if gates is None:
            gates = self.gates()
        m_edge = self._directed_gates(gates)
        h = x
        for layer in self.layers:
            h = layer(h, self.src, self.dst, m_edge)
        w = torch.softmax(self.pool_score(torch.tanh(self.pool_proj(h))), dim=1)  # [B, n, 1]
        h_g = (w * h).sum(1)
        return h, h_g
