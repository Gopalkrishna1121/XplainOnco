"""Explanation tests: deletion / insertion faithfulness and the weight-randomisation sanity check."""
from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.stats import spearmanr

from ..attribution.ig import gene_columns


def _mask_genes(model_cols, X, z, X0, z0, genes: np.ndarray):
    """Set the given genes to baseline in BOTH views."""
    X, z = X.clone(), z.clone()
    if len(genes) == 0:
        return X, z
    gi = torch.as_tensor(genes, dtype=torch.long)
    X[:, gi, :] = X0[:, gi, :]
    cols = torch.isin(model_cols, gi)
    z[:, : model_cols.numel()][:, cols] = z0[:, : model_cols.numel()][:, cols]
    return X, z


@torch.no_grad()
def deletion_insertion(predict_fn, X, z, X0, z0, ranking: np.ndarray, target: int, model_cols: torch.Tensor,
                       steps: int = 20, max_frac: float = 0.25):
    """Track the predicted-class probability while genes are removed (deletion) or restored (insertion)
    in ranked order. Returns the curves and their areas (deletion should be LOW, insertion HIGH).

    predict_fn(X, z) -> probabilities [N, C]. Averaged over the patients in X / z.
    """
    n = X.shape[1]
    k_max = max(1, int(n * max_frac))
    ks = np.unique(np.linspace(0, k_max, steps + 1).astype(int))
    dele, ins = [], []
    for k in ks:
        top = ranking[:k]
        Xd, zd = _mask_genes(model_cols, X, z, X0, z0, top)
        dele.append(float(predict_fn(Xd, zd)[:, target].mean()))
        Xb, zb = X0.expand_as(X).clone(), z0.expand_as(z).clone()  # start from the all-baseline input
        if k:
            gi = torch.as_tensor(top, dtype=torch.long)
            Xb[:, gi, :] = X[:, gi, :]
            cols = torch.isin(model_cols, gi)
            zb[:, : model_cols.numel()][:, cols] = z[:, : model_cols.numel()][:, cols]
        ins.append(float(predict_fn(Xb, zb)[:, target].mean()))
    xs = ks / k_max
    return {"k": ks.tolist(), "deletion": dele, "insertion": ins,
            "deletion_auc": float(np.trapezoid(dele, xs)), "insertion_auc": float(np.trapezoid(ins, xs))}


def reference_rankings(n: int, degree: np.ndarray, rng) -> dict:
    return {"random": rng.permutation(n), "degree": np.argsort(-degree, kind="stable")}


def randomisation_sanity(attr_fn, model, layer_order: list[str], rng_seed: int = 0):
    """Adebayo et al. (2018) cascading randomisation.

    attr_fn(model) -> gene score vector. Weights are re-initialised layer by layer starting from the output;
    returns the Spearman correlation between the original scores and the scores after each step. A faithful
    explanation should decorrelate towards zero as more of the model is randomised.
    """
    base = attr_fn(model)
    if getattr(model, "fusion", None) is not None:
        model.fusion.attn = None  # drop the stored attention graph, which cannot be deep-copied
    work = copy.deepcopy(model)
    torch.manual_seed(rng_seed)
    out = []
    for name in layer_order:
        mod = dict(work.named_modules())[name]
        for p in mod.parameters():
            torch.nn.init.normal_(p, 0.0, 0.1) if p.dim() > 1 else torch.nn.init.zeros_(p)
        s = attr_fn(work)
        rho = spearmanr(base, s).statistic if np.std(s) > 0 and np.std(base) > 0 else 0.0
        out.append({"randomised_through": name, "spearman": float(rho)})
    return out
