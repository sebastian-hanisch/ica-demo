"""Orakel (anderer Rechenweg als der eigene Code): scikit-learn (FastICA mit identischem Startwert und bereits weißen Daten,
PCA-Weißung), SciPy (Zuordnung, Kurtosis), np.corrcoef, Definitionsschleifen (Amari-Index, Mischung, Verzögerung) und die
Fixpunktbedingung der Deflation. Nur kleine Instanzen, wenige Sekunden."""

import math
import warnings

import numpy as np
import pytest

import ica_algorithm as alg
import ica_constants as C
import ica_evaluation as ev
import ica_scenario as sc

sk_decomp = pytest.importorskip("sklearn.decomposition")
sp_opt = pytest.importorskip("scipy.optimize")
sp_stats = pytest.importorskip("scipy.stats")


def _mixture(rng, nc, T, kind="laplace"):
    S = rng.laplace(size=(nc, T)) if kind == "laplace" else rng.uniform(-1, 1, size=(nc, T))
    return rng.standard_normal((nc, nc)) @ S


@pytest.mark.parametrize("contrast", C.CONTRASTS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_symmetric_fastica_equals_sklearn_with_the_same_start(contrast, seed):
    """Dieselbe Fixpunktiteration, dieselbe Startmatrix, bereits weiße Daten -> W stimmt auf Rundungsniveau überein."""
    rng = np.random.default_rng(seed)
    nc = 2 + seed
    X = _mixture(rng, nc, 2500, "laplace" if seed != 1 else "uniform")
    ours = alg.fit_ica(X, nc, contrast, "symmetric", init_start=seed + 3, max_iter=500, tol=1e-9)
    W0 = np.random.default_rng([seed + 3, 4242]).standard_normal((nc, nc))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sk_decomp.FastICA(algorithm="parallel", whiten=False, fun=contrast, w_init=W0, tol=1e-9, max_iter=500).fit(ours.whitening.Z.T)
    assert ours.converged
    assert np.allclose(ours.W, ref.components_, atol=1e-8)
    assert ours.n_iter == ref.n_iter_


@pytest.mark.parametrize("contrast", C.CONTRASTS)
def test_deflation_result_satisfies_the_fixed_point_condition(contrast):
    """Jede Zeile w ist Fixpunkt: E[z g(wᵀz)] − E[g'(wᵀz)]w, nach Gram-Schmidt gegen die früheren Zeilen, zeigt in Richtung w."""
    rng = np.random.default_rng(11)
    X = _mixture(rng, 4, 3000)
    m = alg.fit_ica(X, 4, contrast, "deflation", 2, max_iter=500, tol=1e-9)
    Z = m.whitening.Z
    T = Z.shape[1]
    assert m.converged and np.allclose(m.W @ m.W.T, np.eye(4), atol=1e-9)
    g, gp = alg.contrast_g(contrast, m.W @ Z)
    for p in range(4):
        w = Z @ g[p] / T - gp[p].mean() * m.W[p]
        for k in range(p):
            w = w - (w @ m.W[k]) * m.W[k]
        assert abs(w @ m.W[p]) / np.linalg.norm(w) > 1 - 1e-7


def test_whitening_equals_sklearn_pca_up_to_sign_and_normalisation():
    rng = np.random.default_rng(4)
    for nc, k in ((4, 4), (5, 3), (3, 1)):
        T = 1200
        X = _mixture(rng, nc, T) + 3.0 + 0.1 * rng.standard_normal((nc, T))
        Z = alg.whiten(X, k).Z
        ref = sk_decomp.PCA(n_components=k, whiten=True, svd_solver="full").fit_transform(X.T).T * math.sqrt(T / (T - 1))
        assert np.allclose(np.abs(np.einsum("ij,ij->i", Z, ref)) / T, 1.0, atol=1e-8)


def test_assignment_equals_scipy_linear_sum_assignment_including_ties_and_rectangular():
    rng = np.random.default_rng(5)
    for _ in range(60):
        k, nc = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        cm = np.round(rng.random((k, nc)), 1 if rng.random() < 0.4 else 6)
        idx = ev.assign(cm)
        r, c = sp_opt.linear_sum_assignment(cm, maximize=True)
        used = [j for j in idx if j >= 0]
        assert len(used) == len(set(used)) == min(k, nc)
        assert sum(cm[i, j] for i, j in enumerate(idx) if j >= 0) == pytest.approx(cm[r, c].sum(), abs=1e-9)


def test_correlation_matrix_equals_corrcoef_and_matched_alignment_is_least_squares():
    rng = np.random.default_rng(6)
    S = rng.laplace(size=(3, 500))
    S = (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
    E = rng.standard_normal((4, 500)) * rng.uniform(0.2, 8, (4, 1)) + 2.0
    E[1] += 0.8 * S[2]
    assert np.allclose(ev.correlation_matrix(S, E), np.abs(np.corrcoef(np.vstack([S, E]))[:3, 3:]), atol=1e-10)
    idx, corr, aligned = ev.matched(S, E)
    for i, j in enumerate(idx):
        Xd = np.column_stack([E[j], np.ones(500)])
        coef = np.linalg.lstsq(Xd, S[i], rcond=None)[0]
        assert np.allclose(Xd @ coef, aligned[i], atol=1e-8)


def _amari_by_definition(P):
    P = np.abs(P)
    n, m = P.shape
    s = sum(P[i].sum() / P[i].max() - 1 for i in range(n)) + sum(P[:, j].sum() / P[:, j].max() - 1 for j in range(m))
    d = max(n, m)
    return s / (2 * d * (d - 1))


def test_amari_index_equals_the_textbook_definition_and_is_bounded():
    rng = np.random.default_rng(7)
    for n in (2, 3, 5, 6):
        P = rng.standard_normal((n, n))
        assert ev.amari_index(P) == pytest.approx(_amari_by_definition(P), abs=1e-12)
        assert 0.0 <= ev.amari_index(P) <= 1.0 + 1e-12
        assert ev.amari_index(np.ones((n, n))) == pytest.approx(1.0)


def test_excess_kurtosis_equals_scipy_fisher_kurtosis():
    x = np.random.default_rng(8).laplace(size=(3, 5000)) * np.array([[1.0], [3.0], [0.2]]) + 2.0
    assert np.allclose(ev.excess_kurtosis(x), sp_stats.kurtosis(x, axis=1, fisher=True, bias=True), atol=1e-10)


def _loop_rebuild(m, ne, g, delay, T, seed, kind):
    """Mischung, Verzögerung und Quellen in einfachen Schleifen nachgebaut (reines Python/math)."""
    ds = sc.make_dataset(m, ne, g, kind, 0.0, delay, T, seed)
    pos = [0.5] if ne == 1 else [x for x in np.linspace(0, 1, ne)]
    A = np.zeros((ne, m + g))
    dist = np.zeros((ne, m + g))
    for i in range(m):
        d = np.array([math.hypot(p - C.NEURON_POSITIONS[i][0], 0.0 - C.NEURON_POSITIONS[i][1]) for p in pos])
        a = 1.0 / (d ** 2 + C.DISTANCE_EPS)
        A[:, i], dist[:, i] = C.NEURON_AMPLITUDES[i] * a / a.max(), d
    for j in range(g):
        A[:, m + j] = C.BACKGROUND_AMPLITUDE * (np.ones(ne) if j == 0 else (np.linspace(-1, 1, ne) if ne > 1 else np.ones(1)))
    delays = np.zeros((ne, m + g), int)
    if delay > 0:
        for i in range(m):
            for j in range(ne):
                delays[j, i] = int(round(delay * (dist[j, i] - dist[:, i].min())))
    X = np.zeros((ne, T))
    for j in range(ne):
        for i in range(m + g):
            d = delays[j, i]
            X[j, d:] += A[j, i] * (ds.S[i][:T - d] if d else ds.S[i])
    return ds, A, delays, X


@pytest.mark.parametrize("m,ne,g,delay,kind", [(4, 6, 0, 0, "gauss"), (3, 4, 2, 7, "rhythm"), (2, 1, 1, 0, "gauss"), (5, 8, 2, 12, "gauss")])
def test_dataset_mixture_and_delays_equal_a_loop_rebuild(m, ne, g, delay, kind):
    ds, A, delays, X = _loop_rebuild(m, ne, g, delay, 2000, 17, kind)
    assert np.allclose(ds.A, A) and np.array_equal(ds.delays, delays) and np.allclose(ds.X_clean, X)


def test_neuron_sources_equal_waveform_placed_at_the_firing_times():
    for i in range(5):
        starts = sc.firing_times(i, 3000, 9)
        s = sc.spike_waveform(i)
        src = np.zeros(3000)
        for st in starts:
            for k in range(C.WAVEFORM_LENGTH):
                if st + k < 3000:
                    src[st + k] += s[k]
        z = (src - src.mean()) / src.std()
        ds = sc.make_dataset(5, 6, 0, "gauss", 0.0, 0, 3000, 9)
        assert np.allclose(z, ds.S[i], atol=1e-9)
        assert len(starts) < 2 or np.diff(starts).min() >= C.REFRACTORY - 1
