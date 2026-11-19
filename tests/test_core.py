"""Unit tests for the maths of the paper: entmax, the conditioned split, the gated GAT, IG completeness."""
import numpy as np
import pytest
import torch

from xplainonco.entmax import entmax15, entmoid15
from xplainonco.model import XOConfig, XplainOnco
from xplainonco.tree_branch import ConditionedObliviousLayer


def _tiny(n=24, e=60, seed=0, **kw):
    torch.manual_seed(seed)
    eu = torch.randint(0, n, (2, e))
    eu = eu[:, eu[0] != eu[1]]
    cfg = XOConfig(n_genes=n, n_clin=2, n_trees=8, tree_depth=3, tree_layers=2, hidden=16, heads=2, **kw)
    return XplainOnco(cfg, eu).eval(), eu


def _inputs(n=24, b=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(b, n, 3, generator=g)
    z = torch.cat([X.permute(0, 2, 1).reshape(b, -1), torch.randn(b, 2, generator=g)], 1)
    return X, z


def test_entmax_is_a_sparse_distribution():
    x = torch.randn(6, 50, dtype=torch.float64) * 3
    y = entmax15(x)
    assert torch.allclose(y.sum(-1), torch.ones(6, dtype=torch.float64))
    assert (y == 0).any() and (y >= 0).all()


def test_entmoid_matches_two_class_entmax_and_saturates():
    z = torch.linspace(-4, 4, 41, dtype=torch.float64)
    ref = entmax15(torch.stack([z, torch.zeros_like(z)], -1), -1)[..., 0]
    assert torch.allclose(entmoid15(z), ref, atol=1e-12)
    assert entmoid15(torch.tensor(0.0)).item() == pytest.approx(0.5)
    assert entmoid15(torch.tensor(2.5)).item() == 1.0 and entmoid15(torch.tensor(-2.5)).item() == 0.0


def test_entmax_gradients():
    x = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: entmax15(t, -1), (x,))


def test_zero_initialised_conditioning_equals_plain_node():
    model, _ = _tiny()
    X, z = _inputs()
    with_cond = model(X, z)
    for layer in model.trees.layers:
        layer.conditioning = False
    assert torch.allclose(with_cond, model(X, z), atol=1e-6)


def test_conditioning_shifts_the_threshold_as_in_eq4():
    """With one tree, one level and one gene column, b~ = b + a * gamma(h) (a = 1 because entmax has one input)."""
    n, d = 3, 4
    gene_of_col = torch.tensor([1])  # a single column that belongs to gene 1
    layer = ConditionedObliviousLayer(gene_of_col, n, d, n_trees=1, depth=1, tree_dim=1, gamma_max=10.0, kappa_max=10.0)
    with torch.no_grad():
        layer.b.zero_(); layer.log_tau.zero_(); layer.xi.zero_()
        layer.leaf[0, 0, 0], layer.leaf[0, 1, 0] = 0.0, 1.0  # output = P(go right)
        for net in (layer.gamma, layer.kappa, layer.gamma0):
            net[-1].weight.zero_(); net[-1].bias.zero_()
        layer.gamma[-1].bias.fill_(0.5)  # gamma(h) = 0.5 for every gene
    z = torch.tensor([[0.5]])
    H, hg = torch.zeros(1, n, d), torch.zeros(1, d)
    layer.conditioning = False
    base = layer(z, H, hg).item()
    layer.conditioning = True
    shifted = layer(z, H, hg).item()
    shift = 10.0 * torch.tanh(torch.tensor(0.5 / 10.0)).item()  # gamma is soft-bounded by gamma_max
    expect = entmoid15(torch.tensor(0.5 - shift)).item()  # f - (b + shift)
    assert base == pytest.approx(entmoid15(torch.tensor(0.5)).item())
    assert shifted == pytest.approx(expect, abs=1e-5) and shifted < base


def test_gate_scales_the_message_exactly_as_in_eq1():
    """h_v = ELU( sum_u m_uv * alpha_vu * W h_u ): checked against a hand computation for one layer."""
    from xplainonco.graph_branch import GATLayer, undirected_to_directed

    torch.manual_seed(0)
    n = 4
    eu = torch.tensor([[0, 1, 2], [1, 2, 3]])  # path 0-1-2-3
    src, dst, und = undirected_to_directed(eu, n)
    layer = GATLayer(3, heads=1, head_dim=5).eval()
    h = torch.randn(1, n, 3)
    m_und = torch.tensor([0.0, 0.7, 1.0])
    m_edge = torch.where(und < 0, torch.ones(len(und)), m_und[und.clamp(min=0)])
    out = layer(h, src, dst, m_edge)[0]
    wh = layer.lin(h)[0]
    for v in range(n):
        inc = [k for k in range(len(src)) if dst[k] == v]
        e = torch.stack([torch.nn.functional.leaky_relu((wh[src[k]] * layer.att_src[0]).sum() + (wh[v] * layer.att_dst[0]).sum(), 0.2) for k in inc])
        alpha = torch.softmax(e, 0)
        msg = sum(m_edge[k] * a * wh[src[k]] for k, a in zip(inc, alpha))
        assert torch.allclose(out[v], torch.nn.functional.elu(msg + layer.bias), atol=1e-5)


def test_zero_gates_remove_every_message_between_genes():
    """With all gates at 0 a node's output is its own transformed feature scaled by a scalar self-attention weight."""
    model, eu = _tiny()
    X, _ = _inputs()
    zero = torch.zeros(eu.shape[1])
    layer = model.graph.layers[0]
    m_edge = model.graph._directed_gates(zero)
    out = layer(X, model.graph.src, model.graph.dst, m_edge)
    wh = layer.lin(X)  # [B, n, H*C]
    heads, c = layer.heads, layer.head_dim
    # undo the bias and the ELU, then every head's vector must be parallel to W h_v
    pre = torch.where(out > 0, out, torch.log1p(out.clamp(min=-0.999))) - layer.bias
    pre = pre.view(*pre.shape[:2], heads, c)
    wh = wh.view(*wh.shape[:2], heads, c)
    cos = torch.nn.functional.cosine_similarity(pre, wh, dim=-1)
    assert (cos > 0.999).all()


def test_per_sample_gates_equal_shared_gates():
    model, eu = _tiny()
    X, _ = _inputs()
    g = torch.rand(eu.shape[1])
    h_shared, _ = model.graph(X, g)
    h_per, _ = model.graph(X, g.expand(X.shape[0], -1))
    assert torch.allclose(h_shared, h_per, atol=1e-6)


def test_usage_matrix_rows_sum_to_gene_mass():
    model, _ = _tiny()
    U = model.trees.usage_matrix()
    assert U.shape == (16, 24) and (U >= 0).all()
    assert (U.sum(1) <= 1.0 + 1e-5).all()  # clinical / earlier-layer columns take the rest


def test_baselines_and_ablation_configs_run():
    X, z = _inputs()
    for kw in (dict(mode="graph_only"), dict(conditioning=False), dict(fusion="concat"),
               dict(conditioning=False, fusion="concat")):
        m, _ = _tiny(**kw)
        assert m(X, z).shape == (4, 5)
