"""Training objective (paper Eq. 7).

L = L_CB + lambda_1 * mean_v H(A^GT_v) + lambda_2 * sum_{uv} m_uv
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def class_balanced_weights(y: np.ndarray, n_classes: int, beta: float = 0.999) -> torch.Tensor:
    """Cui et al. (2019): w_c proportional to (1 - beta) / (1 - beta^{n_c}), rescaled to sum to n_classes."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


def xplainonco_loss(logits, y, class_weights, entropy, gate_l1, lam1: float, lam2: float):
    ce = F.cross_entropy(logits, y, weight=class_weights.to(logits.device))
    return ce + lam1 * entropy + lam2 * gate_l1, ce
