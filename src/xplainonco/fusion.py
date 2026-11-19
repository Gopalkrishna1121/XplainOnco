"""Bidirectional cross-attention between gene tokens H and tree tokens L (paper Section 3.5, Eq. 6)."""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Gene tokens and tree tokens read each other; both are mean-pooled, concatenated and classified.

    mode = "cross_attention": the full model.
    mode = "concat":          ablation A2, pooled gene and tree vectors are simply concatenated.
    The two attention maps are kept on ``self.attn`` after every forward pass (with their autograd
    graph) so the attribution layer can weight them by their gradients (Eq. 11).
    """

    def __init__(self, d: int, tree_dim: int, n_trees: int, n_classes: int, heads: int = 4,
                 dropout: float = 0.3, mode: str = "cross_attention"):
        super().__init__()
        assert d % heads == 0 and mode in ("cross_attention", "concat")
        self.d, self.heads, self.dk, self.mode = d, heads, d // heads, mode
        self.tree_proj = nn.Linear(tree_dim, d)  # W_l
        self.tree_embed = nn.Parameter(torch.randn(n_trees, d) * 0.02)  # e_t
        # gene -> tree direction (A^GT): queries from genes, keys / values from trees
        self.q_g, self.k_t, self.v_t = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        # tree -> gene direction (A^TG): queries from trees, keys / values from genes
        self.q_t, self.k_g, self.v_g = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * d, n_classes)
        self.attn: tuple[torch.Tensor, torch.Tensor] | None = None

    def tokens(self, y: torch.Tensor) -> torch.Tensor:
        return self.tree_proj(y) + self.tree_embed  # [B, T, d]

    def _split(self, x):  # [B, N, d] -> [B, h, N, dk]
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.dk).transpose(1, 2)

    def forward(self, H: torch.Tensor, y_trees: torch.Tensor) -> torch.Tensor:
        L = self.tokens(y_trees)
        if self.mode == "concat":
            self.attn = None
            pooled = torch.cat([H.mean(1), L.mean(1)], dim=-1)
            return self.classifier(self.drop(pooled))
        scale = self.dk ** 0.5
        a_gt = torch.softmax(self._split(self.q_g(H)) @ self._split(self.k_t(L)).transpose(-1, -2) / scale, dim=-1)
        a_tg = torch.softmax(self._split(self.q_t(L)) @ self._split(self.k_g(H)).transpose(-1, -2) / scale, dim=-1)
        self.attn = (a_gt, a_tg)  # [B, h, n, T] and [B, h, T, n]
        b = H.shape[0]
        h_new = H + (a_gt @ self._split(self.v_t(L))).transpose(1, 2).reshape(b, -1, self.d)
        l_new = L + (a_tg @ self._split(self.v_g(H))).transpose(1, 2).reshape(b, -1, self.d)
        pooled = torch.cat([h_new.mean(1), l_new.mean(1)], dim=-1)
        return self.classifier(self.drop(pooled))

    def entropy_term(self) -> torch.Tensor:
        """Mean entropy of each gene's attention distribution over trees (Eq. 7, lambda_1 term)."""
        if self.attn is None:
            return torch.zeros(())
        a = self.attn[0]
        return -(a * torch.log(a + 1e-12)).sum(-1).mean()
