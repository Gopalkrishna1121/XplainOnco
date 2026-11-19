import numpy as np
import torch

from xplainonco.attribution import block_hsic_lasso, extract_subnetwork, gene_prizes
from xplainonco.attribution.edges import attention_coupling, edge_ig
from xplainonco.attribution.ig import joint_ig
from xplainonco.attribution.subnetwork import edge_relevance
from xplainonco.evaluation.metrics import corrected_resampled_ttest, jaccard, rank_biased_overlap
from test_core import _inputs, _tiny


def test_joint_ig_satisfies_completeness_and_splits_by_view():
    model, _ = _tiny()
    X, z = _inputs(b=1)
    X0, z0 = torch.zeros_like(X), torch.zeros_like(z)
    out = joint_ig(model, X, z, X0, z0, target=2, steps=64)
    assert out["completeness_gap"] < 5e-3 * max(1.0, abs(out["delta_logit"])) + 5e-3
    assert torch.allclose(out["phi"], out["phi_G"] + out["phi_T"])
    assert out["phi_graph"].shape == (24, 3) and out["phi_tab"].shape == (3 * 24 + 2,)


def test_edge_ig_satisfies_completeness():
    model, eu = _tiny()
    X, z = _inputs(b=1)
    psi, gap = edge_ig(model, X, z, target=1, steps=64)
    assert psi.shape == (eu.shape[1],) and gap < 1e-2


def test_attention_coupling_is_symmetric_and_nonnegative():
    model, _ = _tiny()
    X, z = _inputs(b=1)
    S = attention_coupling(model, X, z, target=0)
    assert S.shape == (24, 24) and torch.allclose(S, S.t()) and (S >= 0).all()


def test_block_hsic_lasso_finds_the_gene_the_output_depends_on():
    rng = np.random.default_rng(0)
    masks = (rng.random((1000, 30)) < 0.3).astype(np.float32)
    f = -2.0 * masks[:, 7] + 0.05 * rng.normal(size=1000)  # output only depends on gene 7
    beta = block_hsic_lasso(masks, f, num_feat=5, rng=rng)
    assert beta.argmax() == 7 and (beta >= 0).all()


def test_block_hsic_lasso_sees_an_and_interaction():
    rng = np.random.default_rng(1)
    masks = (rng.random((1000, 30)) < 0.4).astype(np.float32)
    f = -(masks[:, 3] * masks[:, 11]) + 0.02 * rng.normal(size=1000)  # drops only when BOTH are masked
    beta = block_hsic_lasso(masks, f, num_feat=6, rng=rng)
    assert set(np.argsort(-beta)[:2]) == {3, 11}


def test_degree_penalty_discounts_hubs():
    phi = np.ones(6); beta = np.zeros(6)
    degree = np.array([1, 1, 1, 1, 1, 20.0])  # gene 5 is a hub with exactly the same raw score
    flat = gene_prizes(phi, beta, degree, zeta=0.0, mu=0.0)
    assert np.allclose(flat, flat[0])  # no penalty: the hub ties with everyone
    half = gene_prizes(phi, beta, degree, zeta=0.0, mu=0.5)
    assert half[5] < half[:5].min()  # penalised
    full = gene_prizes(phi, beta, degree, zeta=0.0, mu=1.0)
    assert full[5] == 0.0 and full[:5].min() > 0  # mu = 1: a maximal-degree gene with an average score disappears


def test_pcst_returns_connected_subnetwork_with_a_connector():
    # path 0-1-2-3-4; prizes only on the ends and one side branch, gene 2 is a zero-prize connector
    edge_und = np.array([[0, 1, 2, 3, 2], [1, 2, 3, 4, 5]])
    prizes = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    r = np.ones(5)
    sub = extract_subnetwork(edge_und, prizes, r, eta=0.05)
    nodes = set(sub["nodes"].tolist())
    assert {0, 4}.issubset(nodes) and {1, 2, 3}.issubset(nodes)  # connectors bridge the two prized ends
    assert sub["connector"][np.isin(sub["nodes"], [1, 2, 3])].all()
    # connected: edges form a spanning tree over the nodes
    assert len(sub["edges"]) == len(nodes) - 1


def test_edge_relevance_is_bounded():
    r = edge_relevance(np.array([-2.0, 1.0, 0.5]), np.array([0.0, 4.0, 1.0]))
    assert r.min() >= 0 and r.max() <= 1.0


def test_metrics():
    a = list("abcdefghij")
    assert rank_biased_overlap(a, a) > 0.99
    assert rank_biased_overlap(a, list("klmnopqrst")) < 0.01
    assert jaccard("abc", "bcd") == 0.5
    t, p = corrected_resampled_ttest(np.array([.9, .92, .91, .93, .9]), np.array([.8, .81, .82, .8, .79]), 800, 200)
    assert t > 0 and p < 0.05
