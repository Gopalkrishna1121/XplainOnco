#!/usr/bin/env python
"""Train XplainOnco on the synthetic cohort, run the Cross-Modal Attribution Layer, and export everything the
website's interactive demos need to docs/data/demo.js.

EVERYTHING EXPORTED HERE COMES FROM SYNTHETIC DATA with planted ground truth. It demonstrates the mechanics of
the method (and lets the code be checked against a known answer); it is not a result on TCGA-BRCA.

    python scripts/export_site_data.py            # about 10 minutes on 2 CPU cores
"""
import argparse
import json
import os
import sys
import time

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xplainonco.attribution import AttributionConfig, CrossModalAttribution, gene_ranking  # noqa: E402
from xplainonco.attribution.ig import gene_columns  # noqa: E402
from xplainonco.data import fit_transform_fold, make_synthetic_cohort  # noqa: E402
from xplainonco.evaluation.baselines import XGBBaseline  # noqa: E402
from xplainonco.evaluation.cv import _randomisation_order  # noqa: E402
from xplainonco.evaluation.explain_metrics import deletion_insertion, randomisation_sanity, reference_rankings  # noqa: E402
from xplainonco.evaluation.metrics import classification_metrics  # noqa: E402
from xplainonco.model import XOConfig, XplainOnco  # noqa: E402
from xplainonco.train import TrainConfig, fit, predict_proba  # noqa: E402

ZETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
MUS = [0.0, 0.2, 0.4, 0.6, 0.8]


def layout_graph(g, n, seed=7):
    """Spring layout of the largest component; small components are parked on a ring just outside it, because a
    plain spring layout throws them far away and the min-max scaling then squashes the real network into a corner."""
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    big = g.subgraph(comps[0])
    pos = nx.spring_layout(big, seed=seed, k=1.6 / np.sqrt(len(big)), iterations=300)
    xy = np.zeros((n, 2))
    for i, p in pos.items():
        xy[i] = p
    c0 = xy[list(comps[0])].mean(0)
    radius = np.abs(xy[list(comps[0])] - c0).max() * 1.08
    rest = comps[1:]
    for j, comp in enumerate(rest):
        ang = 2 * np.pi * (j + 0.5) / max(len(rest), 1)
        centre = c0 + 1.12 * radius * np.array([np.cos(ang), np.sin(ang)])
        nodes = sorted(comp)
        for k, v in enumerate(nodes):  # a short chain, perpendicular to the ring direction
            off = (k - (len(nodes) - 1) / 2) * 0.07 * radius
            xy[v] = centre + off * np.array([-np.sin(ang), np.cos(ang)])
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    return xy


def relayout(path):
    """Re-compute only the node positions inside an existing demo.js (no retraining)."""
    txt = open(path).read()
    head, body = txt.split("=", 1)
    d = json.loads(body.strip().rstrip(";"))
    g = nx.Graph()
    g.add_nodes_from(range(len(d["genes"])))
    g.add_edges_from(d["edges"])
    xy = layout_graph(g, len(d["genes"]))
    for i, gene in enumerate(d["genes"]):
        gene["x"], gene["y"] = round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4)
    with open(path, "w") as fh:
        fh.write(head + "= " + json.dumps(d, separators=(",", ":")) + ";\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "data", "demo.js"))
    ap.add_argument("--patients-per-subtype", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relayout-only", action="store_true", help="only recompute node positions in an existing demo.js")
    args = ap.parse_args()
    if args.relayout_only:
        relayout(args.out)
        return
    torch.set_num_threads(2)
    t0 = time.time()

    cohort = make_synthetic_cohort(n_patients=600, n_genes=400, seed=args.seed)
    skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    tr, te = next(skf.split(np.zeros(cohort.n_patients), cohort.y))
    fold = fit_transform_fold(cohort, tr, max_genes=260, var_quantile=0.5)
    tr2, va = train_test_split(tr, test_size=0.2, stratify=cohort.y[tr], random_state=args.seed)
    print(f"cohort {cohort.n_patients} patients, retained {fold.n_genes} genes, {fold.edge_und.shape[1]} STRING-like edges")

    cfg = XOConfig(n_genes=fold.n_genes, n_classes=len(cohort.classes), n_clin=fold.n_clin, n_trees=64, tree_depth=5)
    torch.manual_seed(args.seed)
    model = XplainOnco(cfg, fold.edge_und)
    model, hist = fit(model, fold.X, fold.z, fold.y, tr2, va, TrainConfig(epochs=args.epochs, lr=3e-3, patience=20))
    proba = predict_proba(model, fold.X[te], fold.z[te])
    m = classification_metrics(cohort.y[te], proba)
    xg = XGBBaseline(len(cohort.classes), fold.n_genes).fit(fold.z.numpy()[tr2], cohort.y[tr2])
    mx = classification_metrics(cohort.y[te], xg.predict_proba(fold.z[te]))
    print(f"trained in {time.time() - t0:.0f}s  test macro-F1 {m['macro_f1']:.3f}  (XGBoost {mx['macro_f1']:.3f})")
    torch.save(model.state_dict(), os.path.join(os.path.dirname(args.out), "..", "..", "results", "demo_model.pt"))

    # ------------------------------------------------------------------ layout and static graph
    g = nx.Graph()
    g.add_nodes_from(range(fold.n_genes))
    g.add_edges_from(fold.edge_und.numpy().T.tolist())
    xy = layout_graph(g, fold.n_genes)
    truth = cohort.truth
    orig = fold.gene_idx
    module_of = -np.ones(fold.n_genes, dtype=int)
    for c, cname in enumerate(cohort.classes):
        module_of[np.isin(orig, truth["modules"][cname])] = c
    is_hub = np.isin(orig, truth["hubs"])
    is_label = np.isin(orig, truth["label_genes"])
    deg = fold.degree

    # ------------------------------------------------------------------ attribution per subtype
    att_cfg = AttributionConfig(ig_steps=64, n_perturb=1000, surrogate_patients=4, zeta=0.25, mu=0.4)
    att = CrossModalAttribution(model, fold, att_cfg)
    pred = proba.argmax(1)
    cols = gene_columns(model)
    predict_xo = lambda X, z: torch.softmax(model(X, z), -1).detach().numpy()  # noqa: E731
    rng = np.random.default_rng(args.seed)
    out_sub = []
    for c, cname in enumerate(cohort.classes):
        correct = [int(i) for i, p, t in zip(te, pred, cohort.y[te]) if t == c and p == c][: args.patients_per_subtype]
        print(f"[{cname}] {len(correct)} correctly classified test patients", flush=True)
        if len(correct) < 2:
            continue
        srcs = [att.patient_sources(i, c, with_surrogate=k < att_cfg.surrogate_patients) for k, i in enumerate(correct)]
        base = att.combine(srcs, correct, att_cfg.zeta, att_cfg.mu)
        grid = {}
        for z_ in ZETAS:
            for m_ in MUS:
                r = att.combine(srcs, correct, z_, m_)
                s = r["subnetwork"]
                grid[f"{z_}|{m_}"] = {
                    "nodes": s["nodes"].tolist(), "connector": s["connector"].astype(int).tolist(),
                    "edges": s["edges"].tolist(),
                    "in_module": int(np.isin(s["nodes"], np.where(module_of == c)[0]).sum()),
                    "hubs": int(is_hub[s["nodes"]].sum()), "label_genes": int(is_label[s["nodes"]].sum()),
                }
        # patient-level view for one example patient
        single = att.combine(srcs[:1], correct[:1], att_cfg.zeta, att_cfg.mu)
        ex = single["subnetwork"]
        # faithfulness curves (ranking from the first half of patients, deletion on the second half)
        half = max(1, len(correct) // 2)
        res_a = att.combine(srcs[:half], correct[:half], att_cfg.zeta, att_cfg.mu)
        ranking = gene_ranking(res_a["prizes"], res_a["phi_abs"])
        faith = correct[half:] or correct[:half]
        refs = reference_rankings(fold.n_genes, fold.degree, rng)
        curves = {"XplainOnco": deletion_insertion(predict_xo, fold.X[faith], fold.z[faith], fold.X0, fold.z0, ranking, c, cols),
                  "random": deletion_insertion(predict_xo, fold.X[faith], fold.z[faith], fold.X0, fold.z0, refs["random"], c, cols),
                  "degree": deletion_insertion(predict_xo, fold.X[faith], fold.z[faith], fold.X0, fold.z0, refs["degree"], c, cols)}
        entry = {
            "name": cname, "n_patients": len(correct),
            "planted_module": np.where(module_of == c)[0].tolist(),
            "phi_G": base["phi_G"].round(4).tolist(), "phi_T": base["phi_T"].round(4).tolist(),
            "phi_abs": base["phi_abs"].round(5).tolist(), "beta": base["beta"].round(5).tolist(),
            "expr_z": base["expr_z"].round(3).tolist(),
            "mut_freq": fold.X[correct][:, :, 1].mean(0).numpy().round(3).tolist(),
            "edge_r": base["r_uv"].round(4).tolist(),
            "grid": grid,
            "default": f"{att_cfg.zeta}|{att_cfg.mu}",
            "patient": {"index": int(correct[0]), "nodes": ex["nodes"].tolist(), "edges": ex["edges"].tolist(),
                        "connector": ex["connector"].astype(int).tolist(),
                        "expr_z": fold.X[correct[0], :, 0].numpy().round(3).tolist(),
                        "mut": fold.X[correct[0], :, 1].numpy().astype(int).tolist(),
                        "prizes": single["prizes"].round(4).tolist(), "phi": single["phi"].round(4).tolist(),
                        "phi_G": single["phi_G"].round(4).tolist(), "phi_T": single["phi_T"].round(4).tolist()},
            "faithfulness": {k: {kk: (np.round(vv, 4).tolist() if isinstance(vv, list) else round(vv, 4))
                                 for kk, vv in v.items()} for k, v in curves.items()},
            "completeness_gap": round(base["completeness_gap_max"], 4),
            "edge_completeness_gap": round(base["edge_completeness_gap_max"], 4),
        }
        if c == 1:  # the class with the planted AND-like pair: where does each source put credit on it?
            pair = np.where(np.isin(orig, truth["and_pair"]))[0]
            entry["and_pair"] = pair.tolist()
            ig_rank = np.argsort(np.argsort(-base["phi_abs"]))
            b_rank = np.argsort(np.argsort(-base["beta"]))
            entry["and_pair_ranks"] = {"ig": [int(ig_rank[i]) + 1 for i in pair], "surrogate": [int(b_rank[i]) + 1 for i in pair]}
        out_sub.append(entry)
        print(f"   default subnetwork {len(base['subnetwork']['nodes'])} genes; "
              f"module hits {grid[entry['default']]['in_module']}/{len(entry['planted_module'])}, hubs {grid[entry['default']]['hubs']}", flush=True)

    # ------------------------------------------------------------------ sanity check (cascading randomisation)
    c_big = int(np.bincount(cohort.y[te], minlength=len(cohort.classes)).argmax())
    idx0 = [int(i) for i, p, t in zip(te, pred, cohort.y[te]) if t == c_big and p == c_big][0]

    def score_fn(mm):
        a = CrossModalAttribution(mm, fold, AttributionConfig(ig_steps=32, n_perturb=0))
        return np.abs(a.patient_sources(idx0, c_big, with_surrogate=False)["phi"].numpy())

    try:
        sanity = randomisation_sanity(score_fn, model, _randomisation_order(model), rng_seed=args.seed)
    except Exception as exc:  # never lose the (long) attribution run to a failure in the optional sanity check
        print("sanity check failed:", exc)
        sanity = []

    payload = {
        "meta": {
            "synthetic": True, "seed": args.seed, "n_patients": cohort.n_patients,
            "n_genes_raw": cohort.n_genes, "n_genes": fold.n_genes, "n_edges": int(fold.edge_und.shape[1]),
            "classes": cohort.classes, "class_counts": np.bincount(cohort.y).tolist(),
            "zetas": ZETAS, "mus": MUS,
            "model": {"trees_per_layer": 64, "depth": 5, "tree_layers": 2, "gat_layers": 3, "heads": 4, "hidden": 64},
            "smoke_test_macro_f1": {"xplainonco": round(m["macro_f1"], 3), "xgboost": round(mx["macro_f1"], 3)},
            "conditioning_magnitude": {k: round(v, 3) for k, v in model.conditioning_magnitude(fold.X[te], fold.z[te]).items()},
        },
        "genes": [{"id": fold.genes[i], "x": round(float(xy[i, 0]), 4), "y": round(float(xy[i, 1]), 4), "deg": int(deg[i]),
                   "module": int(module_of[i]), "hub": bool(is_hub[i]), "label": bool(is_label[i])} for i in range(fold.n_genes)],
        "edges": fold.edge_und.numpy().T.tolist(),
        "subtypes": out_sub,
        "sanity": sanity,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write("// Generated by scripts/export_site_data.py from SYNTHETIC data. Not a TCGA result.\n")
        fh.write("window.XO_DEMO = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
