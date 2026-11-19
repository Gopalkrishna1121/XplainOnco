"""XplainOnco: graph branch + node-conditioned tree branch + cross-attention fusion (one loss, end to end)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .fusion import CrossAttentionFusion
from .graph_branch import GraphBranch
from .tree_branch import TreeBranch


@dataclass
class XOConfig:
    n_genes: int
    n_classes: int = 5
    n_clin: int = 0
    n_feat: int = 3  # expression, mutation flag, copy number
    hidden: int = 64
    heads: int = 4
    gat_layers: int = 3
    gate_init: float = 0.9
    tree_layers: int = 2
    n_trees: int = 128
    tree_depth: int = 6
    tree_dim: int = 8
    cross_heads: int = 4
    dropout: float = 0.3
    gat_dropout: float = 0.1
    # --- switches used by the baselines and ablations (paper Section 4.3)
    mode: str = "full"  # "full" | "graph_only" (B2)
    conditioning: bool = True  # False -> A1 / A3
    fusion: str = "cross_attention"  # "concat" -> A2 / A3
    xi_init_std: float = 1.0  # spread of the initial entmax feature logits
    kappa_max: float = 0.5  # soft bound on the log-temperature conditioning (implementation choice)
    gamma_max: float = 0.5  # soft bound on the threshold shift, in standardised units (implementation choice)


class XplainOnco(nn.Module):
    """forward(X, z) -> logits.

    X: [B, n, 3]  gene graph view (log-CPM expression, mutation flag, GISTIC level)
    z: [B, 3n + n_clin] gene table view, same gene order, clinical covariates last
    """

    def __init__(self, cfg: XOConfig, edge_und: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.graph = GraphBranch(edge_und, cfg.n_genes, cfg.n_feat, cfg.hidden, cfg.heads, cfg.gat_layers,
                                 cfg.gate_init, cfg.gat_dropout)
        if cfg.mode == "graph_only":
            self.trees = None
            self.fusion = None
            self.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, cfg.n_classes))
        else:
            cond = cfg.conditioning
            self.trees = TreeBranch(cfg.n_genes, cfg.n_feat, cfg.n_clin, cfg.hidden, cfg.tree_layers, cfg.n_trees,
                                    cfg.tree_depth, cfg.tree_dim, conditioning=cond,
                                    xi_init_std=cfg.xi_init_std, kappa_max=cfg.kappa_max, gamma_max=cfg.gamma_max)
            self.fusion = CrossAttentionFusion(cfg.hidden, cfg.tree_dim, self.trees.total_trees, cfg.n_classes,
                                               cfg.cross_heads, cfg.dropout, cfg.fusion)
            self.classifier = None

    # ------------------------------------------------------------------ forward
    def forward(self, X: torch.Tensor, z: torch.Tensor, gates: torch.Tensor | None = None) -> torch.Tensor:
        H, h_g = self.graph(X, gates)
        if self.cfg.mode == "graph_only":
            return self.classifier(h_g)
        y = self.trees(z, H, h_g)
        return self.fusion(H, y)

    # ------------------------------------------------------------------ helpers
    def gates(self) -> torch.Tensor:
        return self.graph.gates()

    def regularisers(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(attention entropy, L1 of gates): the lambda_1 and lambda_2 terms of Eq. 7."""
        ent = self.fusion.entropy_term() if self.fusion is not None else torch.zeros((), device=self.graph.omega.device)
        return ent, self.gates().sum()

    @torch.no_grad()
    def conditioning_magnitude(self, X: torch.Tensor, z: torch.Tensor) -> dict:
        """Diagnostics for H2: how far the conditioning nets moved from their zero initialisation."""
        if self.trees is None or not self.cfg.conditioning:
            return {"gamma": 0.0, "kappa": 0.0, "gamma0": 0.0}
        H, h_g = self.graph(X)
        out = {"gamma": 0.0, "kappa": 0.0, "gamma0": 0.0}
        for layer in self.trees.layers:
            nl = len(self.trees.layers)
            out["gamma"] += (layer.gamma_max * torch.tanh(layer.gamma(H) / layer.gamma_max)).abs().mean().item() / nl
            out["kappa"] += (layer.kappa_max * torch.tanh(layer.kappa(H) / layer.kappa_max)).abs().mean().item() / nl
            out["gamma0"] += (layer.gamma_max * torch.tanh(layer.gamma0(h_g) / layer.gamma_max)).abs().mean().item() / nl
        return out
