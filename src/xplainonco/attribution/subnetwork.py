"""Fusion and subnetwork extraction (paper Section 3.6.4, Eqs. 13-14).

Gene prizes mix the normalised IG and surrogate scores and subtract a degree penalty (Eq. 13); edge costs
are eta * (1 - r_uv); a prize-collecting Steiner tree (PCST) returns the Oncogenic Pathway Subnetwork.

Parametrisation. The paper normalises phi and beta to unit sum, so a typical prize is about 1/n and
"mu and eta are set to give subnetworks of 10 to 25 genes". To make those two knobs interpretable we
(i) express the degree penalty mu as a fraction of the largest unpenalised prize, and (ii) rescale prizes
so that the largest is 1 and search the cost scale eta (log-bisection) until the subnetwork has the
requested size. Both are re-parametrisations of the same objective, not changes to it.
"""
from __future__ import annotations

import numpy as np


def gene_prizes(phi: np.ndarray, beta: np.ndarray, degree: np.ndarray, zeta: float, mu: float) -> np.ndarray:
    """Eq. 13:  p_v = max(0, (1-zeta) * phi_hat + zeta * beta_hat - mu * d_hat), then scaled so max(p) = 1.

    mu is relative to the largest unpenalised prize (see module docstring).
    """
    phi_hat = np.abs(phi) / max(np.abs(phi).sum(), 1e-12)
    b_sum = beta.sum()
    beta_hat = beta / b_sum if b_sum > 0 else np.zeros_like(beta)
    if b_sum <= 0:
        zeta = 0.0  # no surrogate signal: fall back to IG alone
    base = (1.0 - zeta) * phi_hat + zeta * beta_hat
    d_hat = degree / max(degree.max(), 1)
    p = np.maximum(0.0, base - mu * base.max() * d_hat)
    return p / max(p.max(), 1e-12)


def edge_relevance(psi: np.ndarray, S_edges: np.ndarray | None) -> np.ndarray:
    """r_uv = mean of max-normalised |psi_uv| and S_uv (only psi when no coupling is available)."""
    a = np.abs(psi) / max(np.abs(psi).max(), 1e-12)
    if S_edges is None or np.max(S_edges) <= 0:
        return a
    b = S_edges / max(S_edges.max(), 1e-12)
    return 0.5 * (a + b)


def _pcst(edges: np.ndarray, prizes: np.ndarray, costs: np.ndarray):
    """Hegde-Indyk-Schmidt near-linear PCST via pcst_fast, with a networkx fallback."""
    try:
        import pcst_fast

        nodes, eidx = pcst_fast.pcst_fast(edges.astype(np.int64), prizes.astype(np.float64),
                                          costs.astype(np.float64), -1, 1, "strong", 0)
        return np.asarray(nodes), np.asarray(eidx)
    except ImportError:  # pragma: no cover - exercised only without pcst_fast
        return _pcst_fallback(edges, prizes, costs)


def _pcst_fallback(edges, prizes, costs):  # pragma: no cover
    """Greedy approximation: Steiner tree over prized nodes, then prune leaves that cost more than they earn."""
    import networkx as nx
    from networkx.algorithms.approximation import steiner_tree

    g = nx.Graph()
    for k, (u, v) in enumerate(edges):
        g.add_edge(int(u), int(v), weight=float(costs[k]), idx=k)
    terminals = [int(i) for i in np.where(prizes > 0)[0] if i in g]
    if len(terminals) < 2:
        return np.array(terminals), np.array([], dtype=int)
    best_nodes = max(nx.connected_components(g.subgraph(terminals).copy()), key=len, default=set())
    comp = max(nx.connected_components(g), key=lambda c: len(set(terminals) & c))
    terms = [t for t in terminals if t in comp]
    t = steiner_tree(g.subgraph(comp), terms, weight="weight")
    changed = True
    while changed:
        changed = False
        for v in [v for v in t.nodes if t.degree[v] == 1]:
            (u,) = list(t.neighbors(v))
            if prizes[v] < t[u][v]["weight"] and t.number_of_nodes() > 2:
                t.remove_node(v)
                changed = True
    return np.array(sorted(t.nodes)), np.array([g[u][v]["idx"] for u, v in t.edges])


def extract_subnetwork(edge_und: np.ndarray, prizes: np.ndarray, r_uv: np.ndarray,
                       target_size: tuple[int, int] = (10, 25), eta: float | None = None, iters: int = 40):
    """Solve Eq. 14 and return the connected subgraph.

    edge_und: [2, E] undirected STRING edges among the retained genes; r_uv: [E] edge relevance.
    If ``eta`` is None it is searched so that the subnetwork has between target_size[0] and target_size[1] genes.
    Returns dict(nodes, edges (pairs of gene indices), edge_idx, connector (bool per node), eta).
    """
    edges = edge_und.T
    lo_t, hi_t = target_size

    def solve(e):
        costs = np.maximum(e * (1.0 - r_uv), 1e-4)
        nodes, eidx = _pcst(edges, prizes, costs)
        return nodes, eidx

    if eta is None:
        lo, hi = 1e-4, 1e2  # eta bounds; larger eta -> costlier edges -> smaller subnetwork
        best = None
        for _ in range(iters):
            mid = np.sqrt(lo * hi)
            nodes, eidx = solve(mid)
            size = len(nodes)
            best = (mid, nodes, eidx) if best is None or abs(size - (lo_t + hi_t) / 2) < abs(len(best[1]) - (lo_t + hi_t) / 2) else best
            if lo_t <= size <= hi_t:
                best = (mid, nodes, eidx)
                break
            if size > hi_t:
                lo = mid
            else:
                hi = mid
        eta, nodes, eidx = best
    else:
        nodes, eidx = solve(eta)
    nodes = np.asarray(nodes, dtype=np.int64)
    pe = edges[np.asarray(eidx, dtype=np.int64)] if len(eidx) else np.zeros((0, 2), dtype=np.int64)
    return {"nodes": nodes, "edges": pe, "edge_idx": np.asarray(eidx, dtype=np.int64),
            "connector": prizes[nodes] <= 0, "eta": float(eta)}
