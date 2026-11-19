"""Over-representation test for subnetworks (paper Section 3.6.4 / 4.5).

The paper runs gseapy against MSigDB Hallmark and Reactome. gseapy downloads gene sets from the network,
so this module offers the same test offline: a hypergeometric over-representation test with
Benjamini-Hochberg correction against a user-supplied GMT file, with the *retained genes* as background
(a whole-genome background would inflate enrichment, because those genes were already selected).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom

from ..data.preprocess import benjamini_hochberg


def read_gmt(path: str) -> dict[str, set[str]]:
    sets = {}
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 2:
                sets[parts[0]] = set(parts[2:])
    return sets


def ora(query: set[str], background: set[str], gene_sets: dict[str, set[str]], min_overlap: int = 2):
    """Hypergeometric over-representation. Returns a list of dicts sorted by FDR."""
    query = query & background
    rows = []
    for name, members in gene_sets.items():
        members = members & background
        k = len(query & members)
        if len(members) == 0 or k < min_overlap:
            continue
        p = hypergeom.sf(k - 1, len(background), len(members), len(query))
        rows.append({"set": name, "overlap": k, "set_size": len(members), "p": float(p)})
    if not rows:
        return []
    q = benjamini_hochberg(np.array([r["p"] for r in rows]))
    for r, qq in zip(rows, q):
        r["fdr"] = float(qq)
    return sorted(rows, key=lambda r: r["fdr"])
