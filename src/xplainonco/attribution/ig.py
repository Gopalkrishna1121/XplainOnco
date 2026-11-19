"""Joint Integrated Gradients over both input views (paper Section 3.6.1, Eqs. 8-9)."""
from __future__ import annotations

import numpy as np
import torch


def gauss_legendre(steps: int):
    """Nodes and weights of Gauss-Legendre quadrature mapped to [0, 1] (weights sum to 1)."""
    x, w = np.polynomial.legendre.leggauss(steps)
    return torch.tensor((x + 1) / 2, dtype=torch.float32), torch.tensor(w / 2, dtype=torch.float32)


def gene_columns(model) -> torch.Tensor:
    """g(j): gene index behind each tabular column j (n_feat * n columns), -1 for clinical columns."""
    if model.trees is None:
        n = model.cfg.n_genes
        return torch.cat([torch.arange(n).repeat(model.cfg.n_feat), torch.full((model.cfg.n_clin,), -1)])
    n_z = model.trees.n_z
    return model.trees.layers[0].gene_of_col[:n_z]


def joint_ig(model, X, z, X0, z0, target: int, steps: int = 64, chunk: int = 32):
    """One path integral over both views.

    X: [1, n, 3], z: [1, J] for the patient; X0, z0 the baseline in the same shapes.
    Returns dict with phi_graph [n, 3], phi_tab [J], per-gene phi_G, phi_T, phi (=phi_G + phi_T) and the
    completeness gap |sum(phi) - (F(x) - F(x'))|, which is the check on the quadrature.
    """
    model.eval()
    s_nodes, s_w = gauss_legendre(steps)
    acc_x = torch.zeros_like(X[0])
    acc_z = torch.zeros_like(z[0])
    for i in range(0, steps, chunk):
        s = s_nodes[i : i + chunk].view(-1, 1, 1)
        w = s_w[i : i + chunk]
        xs = (X0 + s * (X - X0)).detach().requires_grad_(True)
        zs = (z0 + s[:, :, 0] * (z - z0)).detach().requires_grad_(True)
        out = model(xs, zs)[:, target].sum()
        gx, gz = torch.autograd.grad(out, [xs, zs], allow_unused=True)
        if gx is not None:
            acc_x += (w.view(-1, 1, 1) * gx).sum(0)
        if gz is not None:
            acc_z += (w.view(-1, 1) * gz).sum(0)
    phi_graph = (X[0] - X0[0]) * acc_x  # [n, 3]
    phi_tab = (z[0] - z0[0]) * acc_z  # [J]
    n = X.shape[1]
    g = gene_columns(model)
    phi_t = torch.zeros(n)
    sel = g >= 0
    phi_t.index_add_(0, g[sel], phi_tab[: g.numel()][sel])
    phi_g = phi_graph.sum(-1)
    with torch.no_grad():
        f1 = model(X, z)[0, target]
        f0 = model(X0, z0)[0, target]
    total = phi_graph.sum() + phi_tab.sum()
    return {
        "phi_graph": phi_graph.detach(), "phi_tab": phi_tab.detach(),
        "phi_G": phi_g.detach(), "phi_T": phi_t.detach(), "phi": (phi_g + phi_t).detach(),
        "completeness_gap": float((total - (f1 - f0)).abs()), "delta_logit": float(f1 - f0),
    }
