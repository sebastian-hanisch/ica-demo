import itertools

import numpy as np
import pytest

import ica_algorithm as alg
import ica_constants as C
import ica_evaluation as ev
import ica_scenario as sc


def _dataset(**kw):
    return ev.make_dataset(**kw)


# --- Weißen ---------------------------------------------------------------------------------------------------------------------


def test_whitened_data_has_identity_covariance_and_zero_mean():
    ds = _dataset()
    wh = alg.whiten(ds.X, 4)
    assert wh.Z.shape == (4, ds.X.shape[1])
    assert np.allclose(wh.Z @ wh.Z.T / ds.X.shape[1], np.eye(4), atol=1e-9) and np.abs(wh.Z.mean(axis=1)).max() < 1e-9
    assert list(wh.eigenvalues) == sorted(wh.eigenvalues, reverse=True)


def test_whitening_keeps_the_strongest_directions():
    ds = _dataset(g=1)
    full = alg.whiten(ds.X, ds.n_electrodes)
    part = alg.whiten(ds.X, 3)
    assert np.allclose(part.Z, full.Z[:3], atol=1e-9)


# --- Kontrastfunktionen ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", C.CONTRASTS)
def test_g_is_the_derivative_of_G_and_gprime_the_derivative_of_g(name):
    u = np.linspace(-3, 3, 41)
    h = 1e-6
    g, gp = alg.contrast_g(name, u)
    assert np.allclose(g, (alg.contrast_G(name, u + h) - alg.contrast_G(name, u - h)) / (2 * h), atol=1e-5)
    assert np.allclose(gp, (alg.contrast_g(name, u + h)[0] - alg.contrast_g(name, u - h)[0]) / (2 * h), atol=1e-5)


@pytest.mark.parametrize("name", C.CONTRASTS)
def test_gaussian_reference_and_non_gaussianity(name):
    rng = np.random.default_rng(3)
    gauss = rng.standard_normal(200_000)
    laplace = rng.laplace(size=200_000)
    laplace = (laplace - laplace.mean()) / laplace.std()
    assert alg.non_gaussianity(name, gauss) < 1e-4
    assert alg.non_gaussianity(name, laplace) > 20 * alg.non_gaussianity(name, gauss)


def test_gaussian_reference_values_are_known():
    assert abs(alg.gaussian_reference("cube") - 0.75) < 0.02
    assert abs(alg.gaussian_reference("exp") + 1 / np.sqrt(2)) < 0.005
    assert abs(alg.gaussian_reference("logcosh") - 0.3746) < 0.005


# --- FastICA -----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("contrast", C.CONTRASTS)
@pytest.mark.parametrize("method", C.METHODS)
def test_unmixing_rotation_is_orthogonal_and_outputs_are_uncorrelated(contrast, method):
    ds = _dataset(noise=0.0)
    m = alg.fit_ica(ds.X, 4, contrast, method, 1)
    assert np.allclose(m.W @ m.W.T, np.eye(4), atol=1e-8)
    S = m.sources
    assert np.allclose(S @ S.T / S.shape[1], np.eye(4), atol=1e-8)
    assert m.unmixing.shape == (4, ds.n_electrodes) and np.allclose(m.transform(ds.X), S, atol=1e-8)


@pytest.mark.parametrize("contrast,method", [("logcosh", "symmetric"), ("cube", "symmetric"), ("cube", "deflation"), ("logcosh", "deflation")])
def test_noise_free_mixture_is_separated_almost_exactly(contrast, method):
    ds = _dataset(noise=0.0)
    m = alg.fit_ica(ds.X, 4, contrast, method, 1)
    assert m.converged
    assert ev.matched(ds.S, m.sources)[1].min() > 0.995


def test_fit_is_deterministic_and_history_ends_at_the_result():
    ds = _dataset()
    a = alg.fit_ica(ds.X, 4, "logcosh", "symmetric", 2)
    b = alg.fit_ica(ds.X, 4, "logcosh", "symmetric", 2)
    assert np.array_equal(a.W, b.W) and a.n_iter == b.n_iter == len(a.history) and np.allclose(a.history[-1], a.W)


def test_flipping_the_sign_of_an_output_does_not_change_the_contrast():
    ds = _dataset()
    m = alg.fit_ica(ds.X, 4)
    for name in C.CONTRASTS:
        for row in m.sources:
            assert abs(alg.non_gaussianity(name, row) - alg.non_gaussianity(name, -row)) < 0.5 * alg.non_gaussianity(name, row)   # gerade Funktionen von G: Vorzeichen egal (nur endliche Schiefe)


def test_pca_components_are_the_whitened_data_without_rotation():
    ds = _dataset()
    assert np.allclose(alg.pca_components(ds.X, 4), alg.whiten(ds.X, 4).Z)


def test_matches_scikit_learn_fastica_on_the_same_data():
    from sklearn.decomposition import FastICA
    ds = _dataset(noise=0.05)
    ours = alg.fit_ica(ds.X, 4, "logcosh", "symmetric", 1).sources
    sk = FastICA(n_components=4, algorithm="parallel", fun="logcosh", whiten="unit-variance", max_iter=500, tol=1e-6, random_state=0).fit_transform(ds.X.T).T
    assert ev.correlation_matrix(ours, sk).max(axis=1).min() > 0.99 and ev.matched(ours, sk)[1].min() > 0.99


# --- Zuordnung, Kennzahlen -----------------------------------------------------------------------------------------------------------


def test_assign_matches_brute_force_on_random_matrices():
    rng = np.random.default_rng(5)
    for k, nc in ((4, 4), (5, 3), (3, 5), (6, 6)):
        cm = rng.random((k, nc))
        best = max(sum(cm[i, j] for i, j in enumerate(p) if j >= 0) for p in itertools.permutations(list(range(nc)) + [-1] * max(k - nc, 0), k))
        idx = ev.assign(cm)
        assert abs(sum(cm[i, j] for i, j in enumerate(idx) if j >= 0) - best) < 1e-9 and len({j for j in idx if j >= 0}) == len([j for j in idx if j >= 0])


def test_matched_undoes_permutation_sign_and_scale():
    rng = np.random.default_rng(6)
    S = rng.standard_normal((4, 3000))
    est = np.array([3.0 * S[2], -0.5 * S[0], 7.0 * S[3], -2.0 * S[1]])
    idx, corr, aligned = ev.matched(S, est)
    assert idx == [1, 3, 0, 2] and np.allclose(corr, 1.0) and np.allclose(aligned, S)


def test_amari_index_hand_instances():
    P = np.array([[0, 2.0, 0], [-3.0, 0, 0], [0, 0, 0.5]])
    assert ev.amari_index(P) < 1e-12
    assert abs(ev.amari_index(np.ones((3, 3))) - 1.0) < 1e-12
    assert ev.amari_index(np.eye(3) + 0.1) < ev.amari_index(np.eye(3) + 0.5)


def test_sir_from_correlation():
    assert abs(ev.sir_db([np.sqrt(0.5)])[0]) < 1e-9 and abs(ev.sir_db([np.sqrt(0.99)])[0] - 10 * np.log10(99)) < 1e-9 and np.isfinite(ev.sir_db([0.0, 1.0])).all()


def test_spike_detection_finds_planted_spikes_and_f1_hand_instances():
    x = np.zeros(2000)
    truth = [200, 700, 1300]
    for t in truth:
        x[t - 3: t + 4] -= np.array([0.2, 0.5, 0.9, 1.0, 0.9, 0.5, 0.2])
    x += 0.01 * np.random.default_rng(0).standard_normal(2000)
    found = ev.detect_spikes(x)
    assert len(found) == 3 and np.abs(found - np.array(truth)).max() <= 1
    assert ev.spike_f1(found, np.array(truth)) == 1.0 and ev.spike_f1(np.array([], dtype=int), np.array(truth)) == 0.0
    assert abs(ev.spike_f1(np.array([200, 700, 1000, 1500]), np.array(truth)) - 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3)) < 1e-12


def test_spike_detection_ignores_a_noise_free_residual():
    """Rauschfreies Signal: die Schwelle darf nicht bei 0 liegen (MAD ist dort 0), sonst würde jeder kleine Rest gemeldet."""
    ds = _dataset(noise=0.0)
    idx, corr, aligned = ev.matched(ds.S, alg.fit_ica(ds.X, 4).sources)
    assert ev.spike_f1(ev.detect_spikes(aligned[0]), ds.spike_times[0]) > 0.99


def test_excess_kurtosis_reference_values():
    rng = np.random.default_rng(7)
    assert abs(ev.excess_kurtosis(rng.standard_normal(400_000))) < 0.05
    assert abs(ev.excess_kurtosis(np.sin(np.linspace(0, 200 * np.pi, 100_000))) + 1.5) < 0.01


# --- Szenario -----------------------------------------------------------------------------------------------------------------------


def test_sources_have_unit_variance_and_neurons_are_spiky():
    ds = _dataset(g=2, kind="rhythm")
    assert np.allclose(ds.S.std(axis=1), 1.0) and np.allclose(ds.S.mean(axis=1), 0.0, atol=1e-12)
    kurt = ev.excess_kurtosis(ds.S)
    assert (kurt[:4] > 20).all() and np.allclose(kurt[4:], -1.5, atol=0.1) and ds.kinds == ("neuron",) * 4 + ("rhythm",) * 2


def test_clean_mixture_is_exactly_the_mixing_matrix_times_the_sources_without_delay():
    ds = _dataset(noise=0.0, g=1)
    assert np.allclose(ds.X, ds.A @ ds.S) and ds.noise_sigma == 0.0 and not ds.delays.any()


def test_delay_shifts_far_electrodes_later_and_keeps_the_nearest_at_zero():
    ds = _dataset(noise=0.0, delay=6)
    assert (ds.delays.min(axis=0) == 0).all() and ds.delays.max() > 0 and (ds.delays[:, :4].max(axis=0) > 0).all()
    j, i = np.unravel_index(np.argmax(ds.delays), ds.delays.shape)
    d = ds.delays[j, i]
    assert np.allclose(ds.X_clean[j, d + 50: d + 400] - sum(ds.A[j, q] * sc.shift(ds.S[q], ds.delays[j, q])[d + 50: d + 400] for q in range(4)), 0)


def test_neuron_streams_do_not_depend_on_the_other_settings():
    a = _dataset(m=2, n=3, seed=11)
    b = _dataset(m=5, n=8, g=2, seed=11)
    assert np.array_equal(a.spike_times[0], b.spike_times[0]) and np.allclose(a.S[0], b.S[0]) and np.allclose(a.S[1], b.S[1])


def test_spike_times_point_at_the_negative_peak_of_the_source():
    ds = _dataset(noise=0.0)
    for i in range(4):
        for t in ds.spike_times[i][:20]:
            window = ds.S[i][t - 3: t + 4]
            assert window.argmin() == 3


def test_noise_is_relative_to_the_neuron_signal_and_independent_of_the_background():
    clean_neurons = _dataset(noise=0.5)
    with_background = _dataset(noise=0.5, g=2)
    assert abs(clean_neurons.noise_sigma - with_background.noise_sigma) < 1e-12 and _dataset(noise=0.0).noise_sigma == 0.0
    assert abs((clean_neurons.X - clean_neurons.X_clean).std() - clean_neurons.noise_sigma) / clean_neurons.noise_sigma < 0.02


def test_layout_constants_are_consistent():
    assert len(C.NEURON_SIGMAS) == len(C.NEURON_RATES) == len(C.NEURON_AMPLITUDES) == len(C.NEURON_POSITIONS) == C.N_NEURONS_MAX
    assert len(C.GAUSS_AR) == len(C.RHYTHM_FREQUENCIES) == C.N_BACKGROUND_MAX
    ds = _dataset(n=1)
    assert ds.X.shape[0] == 1 and ds.A.shape == (1, 4)
