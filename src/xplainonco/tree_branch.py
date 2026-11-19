"""Node-conditioned soft decision trees (paper Section 3.4, Eqs. 2-5).

A NODE ensemble (Popov et al., 2020) of oblivious trees whose split thresholds and
temperatures are shifted / scaled by the graph embeddings of the genes the split reads:

    f_tk(z)   = sum_j a_tkj z_j,                a_tk = entmax15(xi_tk)                 (2)
    pi_tk(z)  = entmoid15((f_tk - b~_tk) / tau~_tk)                                    (3)
    b~_tk     = b_tk + sum_j a_tkj gamma(h_g(j)) + gamma0(h_G)                         (4)
    tau~_tk   = tau_tk * exp( sum_j a_tkj kappa(h_g(j)) )                              (5)

The output layers of gamma, gamma0 and kappa are zero-initialised, so training starts
from an ordinary NODE ensemble (this is also the ablation A1: keep them at zero).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .entmax import entmax15, entmoid15


def _mlp(d_in: int, hidden: int = 32) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class ConditionedObliviousLayer(nn.Module):
    """One layer of ``n_trees`` oblivious trees of depth ``depth``.

    gene_of_col[j] is the gene index behind input column j, or -1 for columns with no gene
    (clinical covariates and the outputs of earlier layers). Those columns receive zero conditioning.
    """

    def __init__(self, gene_of_col: torch.Tensor, n_genes: int, emb_dim: int, n_trees: int = 128,
                 depth: int = 6, tree_dim: int = 8, conditioning: bool = True, cond_hidden: int = 32,
                 xi_init_std: float = 1.0, kappa_max: float = 0.5, gamma_max: float = 0.5):
        super().__init__()
        self.kappa_max, self.gamma_max = kappa_max, gamma_max
        self.n_trees, self.depth, self.tree_dim = n_trees, depth, tree_dim
        self.n_genes, self.conditioning = n_genes, conditioning
        n_cols = gene_of_col.numel()
        self.n_cols = n_cols
        pad = torch.where(gene_of_col < 0, torch.full_like(gene_of_col, n_genes), gene_of_col)
        self.register_buffer("col_gene", pad, persistent=False)  # n_genes == "no gene" slot
        self.register_buffer("gene_of_col", gene_of_col.clone(), persistent=False)

        self.xi = nn.Parameter(torch.randn(n_trees, depth, n_cols) * xi_init_std)  # feature logits
        self.b = nn.Parameter(torch.randn(n_trees, depth) * 0.5)  # base thresholds
        self.log_tau = nn.Parameter(torch.zeros(n_trees, depth))  # base temperatures (log scale)
        self.leaf = nn.Parameter(torch.randn(n_trees, 2 ** depth, tree_dim) * 0.1)
        if conditioning:
            self.gamma = _mlp(emb_dim, cond_hidden)  # threshold shift per gene
            self.kappa = _mlp(emb_dim, cond_hidden)  # log temperature scale per gene
            self.gamma0 = _mlp(emb_dim, cond_hidden)  # patient-level shift from pooled embedding

    def entmax_weights(self) -> torch.Tensor:
        """a_tkj, shape [T, D, J]: which columns each split reads."""
        return entmax15(self.xi, dim=-1)

    def _to_cols(self, per_gene: torch.Tensor) -> torch.Tensor:
        """[B, n] per-gene signal -> [B, J] per-column signal (zero for gene-less columns)."""
        padded = torch.cat([per_gene, per_gene.new_zeros(per_gene.shape[0], 1)], dim=1)
        return padded[:, self.col_gene]

    def forward(self, z: torch.Tensor, H: torch.Tensor | None, h_g: torch.Tensor | None):
        bsz = z.shape[0]
        T, D = self.n_trees, self.depth
        a = self.entmax_weights().reshape(T * D, self.n_cols)
        f = z @ a.t()  # [B, T*D]
        b = self.b.reshape(1, T * D)
        log_tau = self.log_tau.reshape(1, T * D)
        if self.conditioning and H is not None:
            g_raw = self.gamma(H).squeeze(-1)
            g_sig = self._to_cols(self.gamma_max * torch.tanh(g_raw / self.gamma_max))  # [B, J]
            # gamma and kappa are soft-bounded with tanh: a threshold moves by at most gamma_max (in standardised
            # units) and a temperature by at most e^{+-kappa_max}. Implementation choice for stability; the
            # paper's Eqs. 4-5 do not bound them, and the unbounded version overfits on small cohorts.
            k_raw = self.kappa(H).squeeze(-1)
            k_sig = self._to_cols(self.kappa_max * torch.tanh(k_raw / self.kappa_max))
            g0 = self.gamma0(h_g)
            b = b + g_sig @ a.t() + self.gamma_max * torch.tanh(g0 / self.gamma_max)  # Eq. 4  (gamma0 is [B, 1])
            log_tau = log_tau + k_sig @ a.t()  # Eq. 5
        pi = entmoid15((f - b) / torch.exp(log_tau)).reshape(bsz, T, D)  # Eq. 3
        # leaf probabilities: product of left/right probabilities along the path
        p = pi.new_ones(bsz, T, 1)
        for k in range(D):
            pk = pi[:, :, k : k + 1]
            p = torch.cat([p * (1.0 - pk), p * pk], dim=-1)
        y = torch.einsum("btl,tld->btd", p, self.leaf)  # [B, T, tree_dim]
        return y


class TreeBranch(nn.Module):
    """Stack of conditioned oblivious layers with NODE-style dense connections."""

    def __init__(self, n_genes: int, n_feat: int, n_clin: int, emb_dim: int, layers: int = 2,
                 n_trees: int = 128, depth: int = 6, tree_dim: int = 8, conditioning: bool = True,
                 xi_init_std: float = 1.0, kappa_max: float = 0.5, gamma_max: float = 0.5):
        super().__init__()
        self.n_genes, self.n_feat, self.n_clin = n_genes, n_feat, n_clin
        self.n_z = n_feat * n_genes + n_clin
        z_map = torch.cat([torch.arange(n_genes).repeat(n_feat), torch.full((n_clin,), -1, dtype=torch.long)])
        self.layers = nn.ModuleList()
        in_cols = self.n_z
        extra = 0
        for _ in range(layers):
            gene_of_col = torch.cat([z_map, torch.full((extra,), -1, dtype=torch.long)])
            self.layers.append(ConditionedObliviousLayer(gene_of_col, n_genes, emb_dim, n_trees, depth,
                                                         tree_dim, conditioning, xi_init_std=xi_init_std,
                                                         kappa_max=kappa_max, gamma_max=gamma_max))
            extra += n_trees * tree_dim
        self.total_trees = layers * n_trees
        self.tree_dim = tree_dim

    def forward(self, z, H, h_g):
        outs = []
        inp = z
        for layer in self.layers:
            y = layer(inp, H, h_g)
            outs.append(y)
            inp = torch.cat([inp, y.flatten(1)], dim=1)
        return torch.cat(outs, dim=1)  # [B, total_trees, tree_dim]

    @torch.no_grad()
    def usage_matrix(self) -> torch.Tensor:
        """U[t, u] = (1/D) sum_k sum_{j: g(j)=u} a_tkj  (paper Eq. 11), over all trees of all layers."""
        mats = []
        for layer in self.layers:
            a = layer.entmax_weights()  # [T, D, J]
            col_gene = layer.col_gene
            u = a.new_zeros(layer.n_trees, self.n_genes + 1)
            u.index_add_(1, col_gene, a.sum(1))  # sum over depth -> [T, J] -> scatter to genes
            mats.append(u[:, : self.n_genes] / layer.depth)
        return torch.cat(mats, dim=0)  # [total_trees, n]
