#!/usr/bin/env python
"""Integrated Gradients completeness check on the demo model saved by export_site_data.py.

For a few correctly classified test patients per class, compares sum(phi) with F(x) - F(baseline) at 64 and 256
Gauss-Legendre steps. Synthetic cohort only.

    python scripts/export_site_data.py && python scripts/completeness_check.py
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xplainonco.attribution.ig import joint_ig  # noqa: E402
from xplainonco.data import fit_transform_fold, make_synthetic_cohort  # noqa: E402
from xplainonco.model import XOConfig, XplainOnco  # noqa: E402
from xplainonco.train import predict_proba  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "results", "demo_model.pt"))
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(2)
    cohort = make_synthetic_cohort(n_patients=600, n_genes=400, seed=args.seed)
    tr, te = next(StratifiedKFold(5, shuffle=True, random_state=args.seed).split(np.zeros(600), cohort.y))
    fold = fit_transform_fold(cohort, tr, max_genes=260, var_quantile=0.5)
    cfg = XOConfig(n_genes=fold.n_genes, n_classes=len(cohort.classes), n_clin=fold.n_clin, n_trees=64, tree_depth=5)
    model = XplainOnco(cfg, fold.edge_und)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    pred = predict_proba(model, fold.X[te], fold.z[te]).argmax(1)
    worst = {64: 0.0, 256: 0.0}
    for c, name in enumerate(cohort.classes):
        idx = [int(i) for i, p, t in zip(te, pred, cohort.y[te]) if t == c and p == c][: args.per_class]
        for steps in (64, 256):
            rel = []
            for i in idx:
                r = joint_ig(model, fold.X[i : i + 1], fold.z[i : i + 1], fold.X0, fold.z0, c, steps=steps)
                rel.append(100 * r["completeness_gap"] / abs(r["delta_logit"]))
            worst[steps] = max(worst[steps], max(rel))
            print(f"{name}  {steps:>3} steps  gap as % of logit change: " + "  ".join(f"{v:.2f}" for v in rel), flush=True)
    print(f"worst case: {worst[64]:.2f}% at 64 steps, {worst[256]:.2f}% at 256 steps")


if __name__ == "__main__":
    main()
