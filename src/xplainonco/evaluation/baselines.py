"""Baselines of paper Section 4.3.

B1  XGBoost on the retained gene table (+ TreeSHAP as the comparison explanation)
B2  the graph attention branch alone with a linear classifier   -> XplainOnco(mode="graph_only")
B3  MOGONET                                                      -> NOT included, see README
B4  late fusion: B1 and B2 trained separately, probabilities averaged (the design in the original abstract)
"""
from __future__ import annotations

import numpy as np
import torch
import xgboost as xgb

from ..attribution.ig import gene_columns


class XGBBaseline:
    def __init__(self, n_classes: int, n_genes: int, n_feat: int = 3, seed: int = 0, n_estimators: int = 300,
                 max_depth: int = 4, learning_rate: float = 0.08):
        self.n_classes, self.n_genes, self.n_feat = n_classes, n_genes, n_feat
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                           subsample=0.8, colsample_bytree=0.5, random_state=seed, n_jobs=2, verbosity=0,
                           objective="multi:softprob", num_class=n_classes)
        self.model = None

    def fit(self, z: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        # class-balanced sample weights, the same idea as the class-balanced loss of the main model
        counts = np.bincount(y, minlength=self.n_classes).clip(min=1)
        w = (1.0 / counts[y]) * len(y) / self.n_classes if sample_weight is None else sample_weight
        self.model = xgb.XGBClassifier(**self.params).fit(z, y, sample_weight=w)
        return self

    def predict_proba(self, z) -> np.ndarray:
        z = z.numpy() if isinstance(z, torch.Tensor) else z
        p = self.model.predict_proba(z)
        if p.shape[1] < self.n_classes:  # a class missing from the training fold
            full = np.zeros((p.shape[0], self.n_classes))
            full[:, self.model.classes_] = p
            p = full
        return p

    def treeshap_gene_scores(self, z: np.ndarray, target: int) -> np.ndarray:
        """Mean |SHAP| per gene for class ``target`` (the three columns of a gene are summed)."""
        z = z.numpy() if isinstance(z, torch.Tensor) else z
        contrib = self.model.get_booster().predict(xgb.DMatrix(z), pred_contribs=True)  # [N, C, J+1]
        c = contrib[:, target, :-1] if contrib.ndim == 3 else contrib[:, :-1]
        absc = np.abs(c).mean(0)
        n = self.n_genes
        scores = np.zeros(n)
        for r in range(self.n_feat):
            scores += absc[r * n : (r + 1) * n]
        return scores


def late_fusion(p_xgb: np.ndarray, p_gat: np.ndarray) -> np.ndarray:
    """B4: average the predicted probabilities of B1 and B2."""
    return 0.5 * (p_xgb + p_gat)
