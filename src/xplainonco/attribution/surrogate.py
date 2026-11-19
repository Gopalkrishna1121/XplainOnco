"""Graph-structured local surrogate (paper Section 3.6.3, Eq. 12).

Each perturbation picks 1-3 seed genes and replaces the features of every seed and its one-hop STRING
neighbours with baseline values, in both views. HSIC Lasso then finds which masked genes the class
logit depends on, allowing non-linear and combined (non-additive) effects.

Scalability note. Eq. 12 as written uses N_p x N_p Gram matrices. We use the Block HSIC Lasso estimator
(Climente-Gonzalez et al., 2019): perturbations are split into blocks of size B and the Gram matrices are
built inside blocks, which keeps the design matrix at N_p * B rows. For binary masks the centred delta
kernel has the closed form 0.5 * s s^T with s = centred +-1 mask, which is what is used below.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import lars_path

from .ig import gene_columns


def ego_net_perturbations(model, X, z, X0, z0, adj_list, target: int, n_perturb: int = 1000,
                          seeds_range=(1, 3), batch: int = 250, rng=None):
    """Returns masks [N_p, n] (1 = gene replaced by baseline) and the class logit F^(q) for each."""
    rng = np.random.default_rng() if rng is None else rng
    n = X.shape[1]
    g = gene_columns(model)
    col_of_gene = [torch.where(g == v)[0] for v in range(n)]  # columns of z for each gene
    masks = np.zeros((n_perturb, n), dtype=np.float32)
    for q in range(n_perturb):
        k = int(rng.integers(seeds_range[0], seeds_range[1] + 1))
        for s in rng.choice(n, size=k, replace=False):
            masks[q, s] = 1.0
            masks[q, adj_list[s]] = 1.0
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n_perturb, batch):
            m = torch.from_numpy(masks[i : i + batch])  # [b, n]
            xb = torch.where(m.unsqueeze(-1) > 0, X0.expand(len(m), -1, -1), X.expand(len(m), -1, -1))
            zb = z.expand(len(m), -1).clone()
            col_mask = torch.zeros(len(m), z.shape[1])
            col_mask[:, : g.numel()][:, g >= 0] = m[:, g[g >= 0]]
            zb = torch.where(col_mask > 0, z0.expand(len(m), -1), zb)
            outs.append(model(xb, zb)[:, target])
    return masks, torch.cat(outs).numpy()


def block_hsic_lasso(masks: np.ndarray, f: np.ndarray, num_feat: int = 60, block: int = 20, rng=None) -> np.ndarray:
    """Non-negative HSIC Lasso weights beta_v >= 0 (length n). At most ``num_feat`` genes are non-zero."""
    rng = np.random.default_rng() if rng is None else rng
    n_p, n = masks.shape
    perm = rng.permutation(n_p)
    n_blocks = n_p // block
    rows, ys = [], []
    for b in range(n_blocks):
        idx = perm[b * block : (b + 1) * block]
        s = 2.0 * masks[idx] - 1.0  # +-1
        s = s - s.mean(0, keepdims=True)  # centred
        norm = np.linalg.norm(s, axis=0, keepdims=True)
        u = np.where(norm > 1e-9, s / np.maximum(norm, 1e-9), 0.0)  # [B, n], zero for constant columns
        rows.append(np.einsum("iv,jv->ijv", u, u).reshape(block * block, n))
        fb = f[idx]
        d = np.abs(fb[:, None] - fb[None, :])
        sigma = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        K = np.exp(-(d ** 2) / (2 * sigma ** 2 + 1e-12))
        H = np.eye(block) - 1.0 / block
        Kc = H @ K @ H
        ys.append((Kc / (np.linalg.norm(Kc) + 1e-12)).reshape(-1))
    A = np.concatenate(rows).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    active = np.where(np.abs(A).sum(0) > 0)[0]
    if len(active) == 0:
        return np.zeros(n)
    _, _, coefs = lars_path(A[:, active], y, method="lasso", positive=True, max_iter=num_feat)
    beta = np.zeros(n)
    beta[active] = coefs[:, -1]
    return np.maximum(beta, 0.0)


def surrogate_beta(model, X, z, X0, z0, adj_list, target: int, n_perturb: int = 1000, num_feat: int = 60,
                   rng=None) -> np.ndarray:
    masks, f = ego_net_perturbations(model, X, z, X0, z0, adj_list, target, n_perturb, rng=rng)
    return block_hsic_lasso(masks, f, num_feat=num_feat, rng=rng)
