"""Nested, stratified cross-validation with baselines, ablations and explanation tests (paper Section 4.2-4.5).

Outer folds report; an inner split of each outer training set is used for early stopping / model selection.
Every data-dependent step runs inside the outer training fold (see data.preprocess).

This module is a faithful but *scaled* implementation: the number of repeats, folds, epochs, attribution
patients and perturbations are arguments, so the same code runs the full protocol (5 folds x 3 repeats) on
a GPU and a small smoke test on a laptop.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..attribution import AttributionConfig, CrossModalAttribution, gene_ranking
from ..attribution.ig import gene_columns
from ..data.preprocess import fit_transform_fold
from ..model import XOConfig, XplainOnco
from ..train import TrainConfig, fit, predict_proba
from .baselines import XGBBaseline, late_fusion
from .explain_metrics import deletion_insertion, randomisation_sanity, reference_rankings
from .metrics import classification_metrics, jaccard, rank_biased_overlap


@dataclass
class CVConfig:
    outer_folds: int = 5
    repeats: int = 3
    inner_folds: int = 4
    seed: int = 0
    models: tuple = ("xplainonco", "A1", "A2", "A3", "B1", "B2", "B4")
    max_genes: int = 1500
    var_quantile: float = 0.9
    n_trees: int = 128
    tree_depth: int = 6
    tree_layers: int = 2
    train: TrainConfig = field(default_factory=TrainConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    attr_patients: int = 20  # correctly classified outer-test patients per subtype used for attribution
    explain: bool = True
    sanity: bool = True
    verbose: bool = True


def _xo_config(name: str, fold, ccfg: CVConfig, n_classes: int) -> XOConfig:
    base = dict(n_genes=fold.n_genes, n_classes=n_classes, n_clin=fold.n_clin, n_trees=ccfg.n_trees,
                tree_depth=ccfg.tree_depth, tree_layers=ccfg.tree_layers)
    if name == "xplainonco":
        return XOConfig(**base)
    if name == "A1":  # conditioning removed, cross-attention kept
        return XOConfig(**base, conditioning=False)
    if name == "A2":  # cross-attention replaced by concatenation, conditioning kept
        return XOConfig(**base, fusion="concat")
    if name == "A3":  # both removed
        return XOConfig(**base, conditioning=False, fusion="concat")
    if name == "B2":
        return XOConfig(**base, mode="graph_only")
    raise KeyError(name)


def _log(ccfg, *a):
    if ccfg.verbose:
        print(*a, flush=True)


def run_cv(cohort, ccfg: CVConfig) -> dict:
    n_classes = len(cohort.classes)
    results = {"config": {k: v for k, v in asdict(ccfg).items() if k not in ("train", "attribution")},
               "classes": cohort.classes, "folds": []}
    rng = np.random.default_rng(ccfg.seed)
    for rep in range(ccfg.repeats):
        skf = StratifiedKFold(ccfg.outer_folds, shuffle=True, random_state=ccfg.seed + rep)
        for k, (tr, te) in enumerate(skf.split(np.zeros(cohort.n_patients), cohort.y)):
            t0 = time.time()
            _log(ccfg, f"[repeat {rep} fold {k}] preprocessing")
            fold = fit_transform_fold(cohort, tr, ccfg.max_genes, var_quantile=ccfg.var_quantile)
            inner = StratifiedKFold(ccfg.inner_folds, shuffle=True, random_state=ccfg.seed + 100 * rep + k)
            tr_in, va_in = next(inner.split(np.zeros(len(tr)), cohort.y[tr]))
            tr2, va = tr[tr_in], tr[va_in]
            y = cohort.y
            rec = {"repeat": rep, "fold": k, "n_genes": fold.n_genes, "n_edges": int(fold.edge_und.shape[1]),
                   "metrics": {}, "gene_names": fold.genes}
            probs, trained = {}, {}
            if "B1" in ccfg.models or "B4" in ccfg.models:
                xg = XGBBaseline(n_classes, fold.n_genes, seed=ccfg.seed + k).fit(fold.z.numpy()[tr2], y[tr2])
                probs["B1"] = xg.predict_proba(fold.z[te])
                trained["B1"] = xg
            for name in ("xplainonco", "A1", "A2", "A3", "B2"):
                if name not in ccfg.models and not (name == "B2" and "B4" in ccfg.models):
                    continue
                torch.manual_seed(ccfg.seed + k)
                model = XplainOnco(_xo_config(name, fold, ccfg, n_classes), fold.edge_und)
                model, hist = fit(model, fold.X, fold.z, fold.y, tr2, va, ccfg.train)
                probs[name] = predict_proba(model, fold.X[te], fold.z[te])
                trained[name] = model
                rec.setdefault("epochs", {})[name] = len(hist)
                _log(ccfg, f"  {name:10s} macro-F1 {classification_metrics(y[te], probs[name])['macro_f1']:.3f} "
                           f"({len(hist)} epochs, {time.time() - t0:.0f}s elapsed)")
            if "B4" in ccfg.models:
                probs["B4"] = late_fusion(probs["B1"], probs["B2"])
            for name, p in probs.items():
                if name in ccfg.models:
                    rec["metrics"][name] = classification_metrics(y[te], p)
            if "xplainonco" in trained:
                rec["conditioning_magnitude"] = trained["xplainonco"].conditioning_magnitude(fold.X[te], fold.z[te])
            if ccfg.explain and "xplainonco" in trained:
                rec["explanations"] = _explain_fold(trained, probs, fold, cohort, te, ccfg, rng)
            results["folds"].append(rec)
            _log(ccfg, f"[repeat {rep} fold {k}] done in {time.time() - t0:.0f}s")
    results["summary"] = summarise(results, ccfg)
    return results


def _explain_fold(trained, probs, fold, cohort, te, ccfg: CVConfig, rng) -> dict:
    model = trained["xplainonco"]
    att = CrossModalAttribution(model, fold, ccfg.attribution)
    pred = probs["xplainonco"].argmax(1)
    y_te = cohort.y[te]
    cols = gene_columns(model)
    xg = trained.get("B1")
    out = {"subtypes": {}}
    predict_xo = lambda X, z: torch.softmax(model(X, z), -1).detach().numpy()  # noqa: E731
    for c, cname in enumerate(cohort.classes):
        correct = [int(i) for i, p, t in zip(te, pred, y_te) if t == c and p == c]
        if len(correct) < 2:
            continue
        correct = correct[: ccfg.attr_patients]
        half = max(1, len(correct) // 2)
        attr_set, faith_set = correct[:half], correct[half:] or correct[:half]
        res = att.explain(attr_set, c)
        ranking = gene_ranking(res["prizes"], res["phi_abs"])
        sub = res["subnetwork"]
        Xf, zf = fold.X[faith_set], fold.z[faith_set]
        refs = reference_rankings(fold.n_genes, fold.degree, rng)
        curves = {"xplainonco": deletion_insertion(predict_xo, Xf, zf, fold.X0, fold.z0, ranking, c, cols)}
        for rname, r in refs.items():
            curves[rname] = deletion_insertion(predict_xo, Xf, zf, fold.X0, fold.z0, r, c, cols)
        entry = {
            "n_attr_patients": len(attr_set), "n_faith_patients": len(faith_set),
            "ranking_top50": [fold.genes[i] for i in ranking[:50]],
            "subnetwork_genes": [fold.genes[i] for i in sub["nodes"]],
            "n_connectors": int(sub["connector"].sum()),
            "completeness_gap": res["completeness_gap_max"],
            "edge_completeness_gap": res["edge_completeness_gap_max"],
            "faithfulness": {k: {"deletion_auc": v["deletion_auc"], "insertion_auc": v["insertion_auc"]}
                             for k, v in curves.items()},
        }
        if xg is not None:  # TreeSHAP ranking, evaluated on its own model
            shap_scores = xg.treeshap_gene_scores(fold.z[attr_set], c)
            shap_rank = np.argsort(-shap_scores)
            entry["treeshap_top50"] = [fold.genes[i] for i in shap_rank[:50]]
            entry["faithfulness"]["treeshap_on_B1"] = {
                k: v for k, v in deletion_insertion(lambda X, z: xg.predict_proba(z), Xf, zf, fold.X0, fold.z0,
                                                    shap_rank, c, cols).items() if k.endswith("_auc")}
        if ccfg.sanity and c == int(np.bincount(y_te, minlength=len(cohort.classes)).argmax()):  # once, on the largest class
            order = _randomisation_order(model)
            idx0 = attr_set[:1]

            def score_fn(m):
                a = CrossModalAttribution(m, fold, AttributionConfig(ig_steps=32, n_perturb=0))
                s = a.patient_sources(idx0[0], c, with_surrogate=False)
                return np.abs(s["phi"].numpy())

            entry["sanity"] = randomisation_sanity(score_fn, model, order, rng_seed=ccfg.seed)
        out["subtypes"][cname] = entry
    return out


def _randomisation_order(model) -> list[str]:
    """Layer names from the output towards the input, for cascading randomisation."""
    names = []
    if model.fusion is not None:
        names += ["fusion.classifier", "fusion.v_g", "fusion.k_g", "fusion.q_t", "fusion.v_t", "fusion.k_t",
                  "fusion.q_g", "fusion.tree_proj"]
        names += [f"trees.layers.{i}" for i in reversed(range(len(model.trees.layers)))]
    else:
        names += ["classifier"]
    names += [f"graph.layers.{i}" for i in reversed(range(len(model.graph.layers)))]
    return names


def summarise(results: dict, ccfg: CVConfig) -> dict:
    """Mean and std of every metric over outer test folds, plus cross-fold stability of the explanations."""
    folds = results["folds"]
    summ = {"classification": {}, "stability": {}, "faithfulness": {}}
    models = sorted({m for f in folds for m in f["metrics"]})
    for m in models:
        for key in ("accuracy", "macro_f1", "macro_auc", "kappa"):
            vals = np.array([f["metrics"][m][key] for f in folds if m in f["metrics"]])
            summ["classification"].setdefault(m, {})[key] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                                                            "n_folds": int(len(vals))}
        pc = np.array([f["metrics"][m]["per_class_f1"] for f in folds if m in f["metrics"]])
        summ["classification"][m]["per_class_f1_mean"] = pc.mean(0).tolist()
    expl = [f.get("explanations") for f in folds if f.get("explanations")]
    for cname in results["classes"]:
        ents = [e["subtypes"][cname] for e in expl if cname in e["subtypes"]]
        if len(ents) < 2:
            continue
        for src, key in (("xplainonco", "ranking_top50"), ("treeshap_on_B1", "treeshap_top50")):
            if key not in ents[0]:
                continue
            rbos = [rank_biased_overlap(a[key], b[key], 0.9)
                    for i, a in enumerate(ents) for b in ents[i + 1:]]
            summ["stability"].setdefault(cname, {})[f"rbo_{src}"] = float(np.mean(rbos))
        jac = [jaccard(a["subnetwork_genes"], b["subnetwork_genes"]) for i, a in enumerate(ents) for b in ents[i + 1:]]
        summ["stability"].setdefault(cname, {})["subnetwork_jaccard"] = float(np.mean(jac))
        for src in ents[0]["faithfulness"]:
            summ["faithfulness"].setdefault(cname, {})[src] = {
                "deletion_auc": float(np.mean([e["faithfulness"][src]["deletion_auc"] for e in ents if src in e["faithfulness"]])),
                "insertion_auc": float(np.mean([e["faithfulness"][src]["insertion_auc"] for e in ents if src in e["faithfulness"]])),
            }
    return summ


def save(results: dict, path: str):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as fh:
        json.dump(results, fh, indent=1, default=conv)
