"""Cross-Modal Attribution Layer: ties the four evidence sources to the Steiner subnetwork (Figure 3).

Runs after training and adds no trainable part to the classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .edges import attention_coupling, edge_ig
from .ig import joint_ig
from .subnetwork import edge_relevance, extract_subnetwork, gene_prizes
from .surrogate import surrogate_beta


@dataclass
class AttributionConfig:
    ig_steps: int = 64
    n_perturb: int = 1000
    surrogate_features: int = 60
    surrogate_patients: int = 8  # patients (per subtype) the surrogate is run on; IG / coupling use all
    zeta: float = 0.3  # share of the surrogate in the gene prize (tuned on inner folds in the full protocol)
    mu: float = 0.3  # degree penalty, as a fraction of the largest unpenalised prize
    target_size: tuple = (10, 25)
    seed: int = 0


class CrossModalAttribution:
    def __init__(self, model, fold, cfg: AttributionConfig | None = None):
        """model: trained XplainOnco; fold: FoldData (provides baseline, STRING subgraph, degrees)."""
        self.model, self.fold = model.eval(), fold
        self.cfg = cfg or AttributionConfig()
        self.edge_und = fold.edge_und.numpy()
        n = fold.n_genes
        self.adj = [[] for _ in range(n)]
        for u, v in self.edge_und.T:
            self.adj[u].append(int(v))
            self.adj[v].append(int(u))
        self.adj = [np.array(a, dtype=np.int64) for a in self.adj]
        self.rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------ per-patient sources
    def patient_sources(self, idx: int, target: int, with_surrogate: bool = True) -> dict:
        f = self.fold
        X, z = f.X[idx : idx + 1], f.z[idx : idx + 1]
        out = joint_ig(self.model, X, z, f.X0, f.z0, target, steps=self.cfg.ig_steps)
        psi, psi_gap = edge_ig(self.model, X, z, target, steps=self.cfg.ig_steps)
        S = attention_coupling(self.model, X, z, target)
        u, v = f.edge_und
        out.update({"psi": psi.numpy(), "psi_gap": psi_gap, "S_edges": S[u, v].numpy(), "S": S})
        out["beta"] = (surrogate_beta(self.model, X, z, f.X0, f.z0, self.adj, target, self.cfg.n_perturb,
                                      self.cfg.surrogate_features, self.rng) if with_surrogate
                       else np.zeros(f.n_genes))
        return out

    # ------------------------------------------------------------------ one patient or one subtype
    def explain(self, patient_idx: list[int], target: int, zeta: float | None = None, mu: float | None = None):
        """Subnetwork for one patient (list of length 1) or averaged over a list of patients of a subtype."""
        zeta = self.cfg.zeta if zeta is None else zeta
        mu = self.cfg.mu if mu is None else mu
        srcs = []
        for k, i in enumerate(patient_idx):
            srcs.append(self.patient_sources(int(i), target, with_surrogate=k < self.cfg.surrogate_patients))
        return self.combine(srcs, patient_idx, zeta, mu)

    def combine(self, srcs: list[dict], patient_idx, zeta: float, mu: float) -> dict:
        f = self.fold
        phi = np.mean([s["phi"].numpy() for s in srcs], 0)
        phi_G = np.mean([s["phi_G"].numpy() for s in srcs], 0)
        phi_T = np.mean([s["phi_T"].numpy() for s in srcs], 0)
        # |phi| is averaged per patient first, so patients pushing a gene in opposite directions do not cancel
        phi_abs = np.mean([np.abs(s["phi"].numpy()) for s in srcs], 0)
        betas = [s["beta"] for s in srcs if s["beta"].sum() > 0]
        beta = np.mean(betas, 0) if betas else np.zeros(f.n_genes)
        psi = np.mean([np.abs(s["psi"]) for s in srcs], 0)
        S_edges = np.mean([s["S_edges"] for s in srcs], 0)
        prizes = gene_prizes(phi_abs, beta, f.degree.astype(float), zeta, mu)
        r_uv = edge_relevance(psi, S_edges)
        sub = extract_subnetwork(self.edge_und, prizes, r_uv, self.cfg.target_size)
        expr_z = f.X[np.asarray(patient_idx), :, 0].mean(0).numpy()
        return {
            "phi": phi, "phi_G": phi_G, "phi_T": phi_T, "phi_abs": phi_abs, "beta": beta, "psi": psi,
            "S_edges": S_edges, "r_uv": r_uv, "prizes": prizes, "subnetwork": sub, "expr_z": expr_z,
            "completeness_gap_max": max(s["completeness_gap"] for s in srcs),
            "edge_completeness_gap_max": max(s["psi_gap"] for s in srcs),
            "n_patients": len(patient_idx), "sources": srcs, "zeta": zeta, "mu": mu,
        }

    def regrid(self, srcs: list[dict], patient_idx, zetas, mus) -> dict:
        """Re-fuse stored sources over a (zeta, mu) grid without recomputing IG / coupling / surrogate."""
        return {(z, m): self.combine(srcs, patient_idx, z, m) for z in zetas for m in mus}


def gene_ranking(prizes: np.ndarray, phi_abs: np.ndarray | None = None) -> np.ndarray:
    """Gene indices from most to least important: by prize, ties (including all zero prizes) broken by |phi|."""
    tie = np.zeros_like(prizes) if phi_abs is None else phi_abs / max(phi_abs.max(), 1e-12)
    return np.lexsort((-tie, -prizes))
