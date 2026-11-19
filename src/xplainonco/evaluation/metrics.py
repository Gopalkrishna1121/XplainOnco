"""Classification and explanation metrics (paper Section 4.5)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(1)
    n_classes = proba.shape[1]
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=range(n_classes), zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, pred)),
    }
    out["per_class_f1"] = f1_score(y_true, pred, average=None, labels=range(n_classes), zero_division=0).tolist()
    try:
        present = np.unique(y_true)
        aucs = [roc_auc_score((y_true == c).astype(int), proba[:, c]) for c in present]
        out["macro_auc"] = float(np.mean(aucs))
    except ValueError:
        out["macro_auc"] = float("nan")
    return out


def corrected_resampled_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired scores from repeated k-fold CV.

    a, b: paired scores over all outer test folds (e.g. 15 = 5 folds x 3 repeats).
    The variance is inflated by (1/k + n_test/n_train) to allow for overlapping training sets.
    """
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float("nan"), 1.0
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def rank_biased_overlap(r1, r2, p: float = 0.9, depth: int | None = None) -> float:
    """Extrapolated rank-biased overlap (Webber et al., 2010) between two ranked lists of items."""
    r1, r2 = list(r1), list(r2)
    depth = depth or min(len(r1), len(r2))
    s1, s2 = set(), set()
    agree_sum = 0.0
    for d in range(1, depth + 1):
        s1.add(r1[d - 1])
        s2.add(r2[d - 1])
        agree_sum += (len(s1 & s2) / d) * (p ** d)
    a_d = len(s1 & s2) / depth
    return float((1 - p) / p * agree_sum + a_d * p ** depth)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)
