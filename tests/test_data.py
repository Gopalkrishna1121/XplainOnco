import numpy as np

from xplainonco.data import fit_transform_fold, make_synthetic_cohort
from xplainonco.data.preprocess import benjamini_hochberg


def test_synthetic_cohort_has_the_properties_the_paper_cares_about():
    c = make_synthetic_cohort(n_patients=500, n_genes=200, seed=3)
    counts = np.bincount(c.y)
    assert counts.max() / counts.min() > 10 - 2  # strongly imbalanced classes
    import networkx as nx

    g = nx.Graph([(u, v) for u, v, s in zip(c.edges_u, c.edges_v, c.edges_score) if s >= 700])
    for mod in c.truth["modules"].values():
        assert nx.is_connected(g.subgraph(mod.tolist()))  # every planted module is a connected pathway
    assert len(c.truth["hubs"]) == 6


def test_fold_preprocessing_uses_training_statistics_only():
    c = make_synthetic_cohort(n_patients=400, n_genes=150, seed=1)
    tr = np.arange(0, 300)
    f1 = fit_transform_fold(c, tr, max_genes=100, var_quantile=0.5)
    c2 = make_synthetic_cohort(n_patients=400, n_genes=150, seed=1)
    c2.expr[300:] += 50.0  # wildly change the TEST patients, and their labels
    c2.y[300:] = (c2.y[300:] + 1) % 5
    f2 = fit_transform_fold(c2, tr, max_genes=100, var_quantile=0.5)
    assert f1.genes == f2.genes  # gene selection ignored the test fold
    assert np.allclose(f1.X[:300].numpy(), f2.X[:300].numpy())  # so did the standardisation
    assert np.allclose(f1.X0.numpy(), f2.X0.numpy())  # and the IG baseline


def test_string_threshold_and_baseline_conventions():
    c = make_synthetic_cohort(n_patients=300, n_genes=120, seed=2)
    f = fit_transform_fold(c, np.arange(200), max_genes=80, var_quantile=0.5)
    assert f.X0[0, :, 1].abs().sum() == 0  # mutation flags are zero in the baseline
    assert f.edge_und.shape[0] == 2 and (f.edge_und[0] < f.edge_und[1]).all()  # unique undirected edges
    assert f.z.shape[1] == 3 * f.n_genes + f.n_clin and f.degree.sum() == 2 * f.edge_und.shape[1]


def test_benjamini_hochberg_is_monotone():
    q = benjamini_hochberg(np.array([0.001, 0.01, 0.03, 0.5, 0.9]))
    assert (np.diff(q[np.argsort([0.001, 0.01, 0.03, 0.5, 0.9])]) >= 0).all() and q.max() <= 1


def test_tcga_loader_reads_the_documented_layout(tmp_path):
    import pandas as pd

    from xplainonco.data.tcga import load_tcga, withhold_label_genes

    rng = np.random.default_rng(0)
    genes = [f"GENE{i}" for i in range(12)]
    samples = [f"TCGA-AA-{i:04d}-01A-11R-A00Z-07" for i in range(10)]
    subtypes = ["LumA"] * 4 + ["Basal"] * 3 + ["Her2"] * 3

    def write(name, arr, index, columns):
        pd.DataFrame(arr, index=index, columns=columns).to_csv(tmp_path / name, sep="\t")

    write("expression.tsv", rng.normal(5, 1, (12, 10)), genes, samples)
    write("cna.tsv", rng.integers(-2, 3, (12, 10)), genes, samples)
    write("mutation.tsv", rng.integers(0, 2, (8, 10)), genes[:8], samples)  # genes without mutation rows default to 0
    pd.DataFrame({"subtype": subtypes, "age": rng.integers(40, 80, 10)},
                 index=[s[:12] for s in samples]).to_csv(tmp_path / "labels.tsv", sep="\t")
    pd.DataFrame({"a": ["GENE0", "GENE1", "GENE2", "GENE0"], "b": ["GENE1", "GENE2", "NOPE", "GENE0"],
                  "score": [900, 650, 999, 800]}).to_csv(tmp_path / "string_links.tsv", sep="\t", index=False)
    (tmp_path / "pam50.txt").write_text("GENE3\nGENE4\nNOT_A_GENE\n")

    c = load_tcga(str(tmp_path), expr_is_counts=False)
    assert c.n_patients == 10 and c.n_genes == 12 and c.classes == ["LumA", "Her2", "Basal"]
    assert c.y.tolist() == [0] * 4 + [2] * 3 + [1] * 3
    assert c.mut[:, 8:].sum() == 0 and c.clin.shape == (10, 1)
    assert len(c.edges_u) == 2  # unknown gene and self-link dropped
    assert c.label_genes == ["GENE3", "GENE4"]
    w = withhold_label_genes(c)
    assert w.n_genes == 10 and "GENE3" not in w.genes and w.label_genes == []
