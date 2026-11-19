"""Edge scores: edge-gate Integrated Gradients (Eq. 10) and gradient-weighted cross-attention coupling (Eq. 11)."""
from __future__ import annotations

import torch

from .ig import gauss_legendre


def edge_ig(model, X, z, target: int, steps: int = 64, chunk: int = 32):
    """psi_uv: IG along the path s * m, s from 0 (no STRING messages, self-loops kept) to 1 (trained gates).

    Completeness: sum(psi) = F(x; m) - F(x; 0).
    """
    model.eval()
    m = model.gates().detach()
    s_nodes, s_w = gauss_legendre(steps)
    acc = torch.zeros_like(m)
    for i in range(0, steps, chunk):
        s = s_nodes[i : i + chunk]
        w = s_w[i : i + chunk]
        g = (s.view(-1, 1) * m.view(1, -1)).detach().requires_grad_(True)  # [S, E]
        out = model(X.expand(len(s), -1, -1), z.expand(len(s), -1), gates=g)[:, target].sum()
        (grad,) = torch.autograd.grad(out, g)
        acc += (w.view(-1, 1) * grad).sum(0)
    psi = m * acc
    with torch.no_grad():
        f1 = model(X, z)[0, target]
        f0 = model(X, z, gates=torch.zeros_like(m))[0, target]
    return psi.detach(), float(((psi.sum()) - (f1 - f0)).abs())


def attention_coupling(model, X, z, target: int):
    """S = C + C^T with C = (R_GT + R_TG^T) U  (Eq. 11). Returns a dense [n, n] matrix.

    R are the cross-attention maps multiplied by their gradient w.r.t. the class logit, positive part,
    averaged over heads (Chefer et al., 2021). U[t, u] is how strongly tree t splits on gene u.
    Returns zeros for models without cross-attention (baselines, ablation A2).
    """
    n = model.cfg.n_genes
    if model.fusion is None or model.fusion.mode != "cross_attention":
        return torch.zeros(n, n)
    model.eval()
    with torch.enable_grad():
        logit = model(X, z)[0, target]
        a_gt, a_tg = model.fusion.attn
        g_gt, g_tg = torch.autograd.grad(logit, [a_gt, a_tg])
    r_gt = (g_gt * a_gt).clamp(min=0).mean(1)[0]  # [n, T]
    r_tg = (g_tg * a_tg).clamp(min=0).mean(1)[0]  # [T, n]
    U = model.trees.usage_matrix()  # [T, n]
    C = (r_gt + r_tg.t()) @ U  # [n, n]
    return (C + C.t()).detach()
