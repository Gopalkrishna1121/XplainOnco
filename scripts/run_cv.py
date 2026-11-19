#!/usr/bin/env python
"""Nested stratified cross-validation: XplainOnco, ablations A1-A3 and baselines B1, B2, B4, plus explanation tests.

    python scripts/run_cv.py --quick                       # synthetic cohort, small settings, a few minutes
    python scripts/run_cv.py --data path/to/tcga_folder    # TCGA-BRCA tables (see docs/DATA.md), paper settings
    python scripts/run_cv.py --data path/to/tcga_folder --withhold-label-genes

Numbers printed for the synthetic cohort say nothing about real tumours. B3 (MOGONET) and GNNExplainer are not
implemented in this repository.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xplainonco.attribution import AttributionConfig  # noqa: E402
from xplainonco.data import make_synthetic_cohort  # noqa: E402
from xplainonco.evaluation import CVConfig, corrected_resampled_ttest, run_cv, save  # noqa: E402
from xplainonco.train import TrainConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder with the TCGA tables (docs/DATA.md); omit to use the synthetic cohort")
    ap.add_argument("--quick", action="store_true", help="1 repeat x 3 folds, few trees and epochs; for checking the pipeline")
    ap.add_argument("--withhold-label-genes", action="store_true", help="drop the PAM50 genes before everything else")
    ap.add_argument("--no-explain", action="store_true", help="classification only")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "cv.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--counts", action="store_true", help="expression.tsv holds raw counts (default: already log-scale)")
    args = ap.parse_args()
    t0 = time.time()

    if args.data:
        from xplainonco.data.tcga import load_tcga, withhold_label_genes

        cohort = load_tcga(args.data, expr_is_counts=args.counts)
        if args.withhold_label_genes:
            cohort = withhold_label_genes(cohort)
        print(f"TCGA tables: {cohort.n_patients} patients, {cohort.n_genes} genes, classes {cohort.classes}")
    else:
        cohort = make_synthetic_cohort(n_patients=600, n_genes=400, seed=args.seed)
        if args.withhold_label_genes:
            from xplainonco.data.tcga import withhold_label_genes

            cohort = withhold_label_genes(cohort)
        print(f"SYNTHETIC cohort: {cohort.n_patients} patients, {cohort.n_genes} genes. Results say nothing about real tumours.")

    if args.quick:
        cfg = CVConfig(outer_folds=3, repeats=1, inner_folds=4, seed=args.seed, max_genes=260, var_quantile=0.5,
                       n_trees=48, tree_depth=5, train=TrainConfig(epochs=40, lr=3e-3, patience=10),
                       attribution=AttributionConfig(ig_steps=24, n_perturb=300, surrogate_patients=2, surrogate_features=30),
                       attr_patients=4, explain=not args.no_explain, sanity=False)
    else:
        cfg = CVConfig(seed=args.seed, explain=not args.no_explain)

    results = run_cv(cohort, cfg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save(results, args.out)

    summ = results["summary"]["classification"]
    print("\nmacro-F1 over outer test folds (mean +- std):")
    for m, v in sorted(summ.items(), key=lambda kv: -kv[1]["macro_f1"]["mean"]):
        print(f"  {m:11s} {v['macro_f1']['mean']:.3f} +- {v['macro_f1']['std']:.3f}   (n_folds={v['macro_f1']['n_folds']})")

    folds = results["folds"]
    if "xplainonco" in summ and len(folds) > 2:
        n = cohort.n_patients
        k = cfg.outer_folds
        n_test, n_train = n // k, n - n // k
        print("\ncorrected resampled t-test on macro-F1, XplainOnco vs each model (Nadeau & Bengio):")
        a = np.array([f["metrics"]["xplainonco"]["macro_f1"] for f in folds])
        for m in summ:
            if m == "xplainonco":
                continue
            b = np.array([f["metrics"][m]["macro_f1"] for f in folds if m in f["metrics"]])
            if len(b) == len(a):
                t, p = corrected_resampled_ttest(a, b, n_train, n_test)
                print(f"  vs {m:11s} mean diff {a.mean() - b.mean():+.3f}  t={t:+.2f}  p={p:.3f}")
    stab = results["summary"].get("stability", {})
    if stab:
        print("\nexplanation stability across folds:")
        for c, v in stab.items():
            print(f"  {c}: " + ", ".join(f"{k}={x:.2f}" for k, x in v.items()))
    print(f"\nwrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
