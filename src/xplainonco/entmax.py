"""1.5-entmax and its two-class form (entmoid), as used by NODE.

Peters, Niculae & Martins (2019), "Sparse sequence-to-sequence models".
Popov, Morozov & Babenko (2020), "Neural Oblivious Decision Ensembles".

Both functions are written with plain differentiable torch ops (sort, cumsum,
gather), so autograd gives the exact Jacobian almost everywhere and Integrated
Gradients can flow through them.
"""
from __future__ import annotations

import torch


def entmax15(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact sort-based 1.5-entmax along ``dim``. Output sums to 1 and is sparse."""
    x = x - x.max(dim=dim, keepdim=True).values
    x = x / 2.0
    x_sorted, _ = torch.sort(x, dim=dim, descending=True)
    d = x.shape[dim]
    shape = [1] * x.dim()
    shape[dim] = d
    k = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).view(shape)

    mean = x_sorted.cumsum(dim) / k
    mean_sq = (x_sorted ** 2).cumsum(dim) / k
    ss = k * (mean_sq - mean ** 2)
    delta = (1.0 - ss) / k
    tau = mean - torch.sqrt(torch.clamp(delta, min=0.0))
    support = (tau <= x_sorted).sum(dim=dim, keepdim=True)
    tau_star = tau.gather(dim, support - 1)
    return torch.clamp(x - tau_star, min=0.0) ** 2


def entmoid15(x: torch.Tensor) -> torch.Tensor:
    """Two-class 1.5-entmax, entmax15([x, 0])[0]: a smooth step that is exactly 0 / 1 for |x| >= 2.

    Closed form for |x| < 2:  p = (x + sqrt(8 - x^2))^2 / 16  (p = 0.5 at x = 0).
    """
    ax = x.abs()
    root = torch.sqrt(torch.clamp(8.0 - ax ** 2, min=0.0))
    y_neg = torch.where(ax < 2.0, (root - ax) ** 2 / 16.0, torch.zeros_like(ax))
    return torch.where(x >= 0, 1.0 - y_neg, y_neg)
