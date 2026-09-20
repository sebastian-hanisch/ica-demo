"""Auswertung und die Aussagen der App als Tests: jede Zahl in den Hilfetexten, Tabellen und Presets ist hier über die festen Sweep-Datensätze belegt (Toleranzen bewusst weit)."""

import numpy as np
import pytest

import ica_algorithm as alg
import ica_constants as C
import ica_evaluation as ev


def _mean_over_seeds(fn, seeds=C.SWEEP_SEEDS, **kw):
    return float(np.mean([fn(ev.analyse(ev.make_dataset(seed=s, **kw), ev.Settings())) for s in seeds]))


def _corr(a):
    return a.methods["ica"].neuron_corr


def test_analysis_has_the_expected_structure_on_the_default_scene():
    ds = ev.make_dataset()
    a = ev.analyse(ds, ev.Settings())
    assert set(a.methods) == {"electrodes", "pca", "ica"} and a.model.converged and np.isnan(a.bg_sir) and a.ref_clean > 0.99 and a.ref_instant == a.methods["ica"].neuron_corr
    assert a.methods["ica"].aligned.shape == ds.S.shape and set(a.amari) == {"pca", "ica"} and a.amari["ica"] < a.amari["pca"]
    assert len(a.kurtosis["sources"]) == 4 and len(ev.iteration_curve(a)) == a.model.n_iter + 1
    assert abs(ev.iteration_curve(a)[-1] - a.methods["ica"].neuron_corr) < 1e-12 and ev.iteration_curve(a)[0] < ev.iteration_curve(a)[-1]


def test_reference_runs_split_the_loss_into_noise_and_delay_and_are_skipped_for_clean_instant_data():
    ds = ev.make_dataset(delay=6, noise=0.3)
    a = ev.analyse(ds, ev.Settings())
    assert a.ref_clean > a.ref_instant > a.methods["ica"].neuron_corr
    assert np.isnan(ev.analyse(ev.make_dataset(noise=0.0), ev.Settings()).ref_clean)
    b = ev.analyse(ev.make_dataset(noise=0.3), ev.Settings())
    assert b.ref_instant == b.methods["ica"].neuron_corr and b.ref_clean > b.ref_instant


def test_the_bitmask_assignment_leaves_sources_unmatched_only_when_there_are_fewer_estimates():
    a = ev.analyse(ev.make_dataset(n=3), ev.Settings())
    assert -1 in a.methods["ica"].idx and a.methods["ica"].estimates.shape[0] == 3
    b = ev.analyse(ev.make_dataset(), ev.Settings())
    assert -1 not in b.methods["ica"].idx


@pytest.mark.parametrize("kwargs,code", [
    ({}, "ica_wins"),
    ({"g": 1}, "ica_wins"),
    ({"g": 2, "kind": "rhythm"}, "ica_wins"),
    ({"g": 2, "kind": "gauss"}, "gauss_pair"),
    ({"n": 3}, "underdetermined"),
    ({"n": 1}, "underdetermined"),
    ({"delay": 8}, "delay"),
    ({"noise": 0.7}, "noise"),
])
def test_verdict_codes_hold_on_several_datasets(kwargs, code):
    for seed in (7, 100000, 100001):
        a = ev.analyse(ev.make_dataset(seed=seed, **kwargs), ev.Settings())
        assert ev.verdict(a)[1] == code, (seed, ev.verdict(a)[1])


def test_neutral_verdict_when_there_is_no_clear_advantage():
    a = ev.analyse(ev.make_dataset(noise=1.0, m=2, n=2), ev.Settings())
    assert ev.verdict(a)[1] in ("noise", "neutral")


def test_not_converged_verdict_is_reported():
    ds = ev.make_dataset()
    model = alg.fit_ica(ds.X, 4, "logcosh", "symmetric", 1, max_iter=1)
    assert not model.converged


def test_sweep_and_gauss_seeds_are_separate_from_demo_seeds():
    assert min(C.SWEEP_SEEDS) >= 100_000 > C.DEFAULT_SEED and len(C.SWEEP_SEEDS) >= 5


def test_sweeps_are_deterministic_and_have_the_expected_shape():
    a = ev.sweep("noise", values=(0.0, 0.4))
    assert a == ev.sweep("noise", values=(0.0, 0.4)) and [r["x"] for r in a] == [0.0, 0.4] and a[0]["ica"] > a[1]["ica"] and all(np.isfinite(r["ica_std"]) for r in a)
    assert all(set(r) >= {"x", "ica", "pca", "electrodes", "f1", "amari", "ica_min"} for r in a)


# --- Aussagen der App ---------------------------------------------------------------------------------------------------------------------


def test_default_scene_ica_beats_pca_and_single_electrode_clearly():
    """Beleg für Preset-Hilfe und Erfolgsmeldung: ICA 0.98, PCA 0.65, beste Einzelelektrode 0.73, Spitzen-F1 1.0 gegen etwa 0.65."""
    a = [ev.analyse(ev.make_dataset(seed=s), ev.Settings()) for s in C.SWEEP_SEEDS]
    assert 0.97 < np.mean([x.methods["ica"].neuron_corr for x in a]) < 0.99
    assert 0.60 < np.mean([x.methods["pca"].neuron_corr for x in a]) < 0.70
    assert 0.68 < np.mean([x.methods["electrodes"].neuron_corr for x in a]) < 0.78
    assert np.mean([x.methods["ica"].f1 for x in a]) > 0.98 and 0.55 < np.mean([x.methods["pca"].f1 for x in a]) < 0.75


@pytest.mark.parametrize("noise,expected", [(0.05, 0.98), (0.4, 0.69), (1.0, 0.43)])
def test_noise_help_text(noise, expected):
    """Beleg für die Hilfe zum Rauschen: Korrelation 0.98 (0.05), 0.69 (0.4), 0.43 (1.0)."""
    assert abs(_mean_over_seeds(_corr, noise=noise) - expected) < 0.04


@pytest.mark.parametrize("delay,expected", [(0, 0.98), (2, 0.92), (8, 0.76)])
def test_delay_help_text(delay, expected):
    """Beleg für die Hilfe zur Verzögerung: 0.98 ohne, 0.92 bei 2, 0.76 bei 8."""
    assert abs(_mean_over_seeds(_corr, delay=delay) - expected) < 0.03


@pytest.mark.parametrize("n,expected", [(4, 0.98), (3, 0.68), (2, 0.41)])
def test_electrode_help_text(n, expected):
    """Beleg für die Hilfe zu den Elektroden (4 Neuronen): 0.98 bei 4, 0.68 bei 3, 0.41 bei 2."""
    assert abs(_mean_over_seeds(_corr, n=n) - expected) < 0.03


def test_short_recordings_are_enough_at_this_noise():
    """Beleg für die Hilfe zur Länge: schon 1000 Abtastwerte reichen für Korrelation 0.98."""
    assert _mean_over_seeds(_corr, n_samples=1000) > 0.96


def test_ica_beats_the_best_single_electrode_only_below_about_half_noise():
    """Beleg für die Hilfe 'Starkes Rauschen': bei 0.7 ist die beste Einzelelektrode besser als ICA, bei 0.4 ist ICA noch vorn."""
    el = lambda a: a.methods["electrodes"].neuron_corr
    assert _mean_over_seeds(_corr, noise=0.4) > _mean_over_seeds(el, noise=0.4) + 0.005
    assert _mean_over_seeds(_corr, noise=0.7) < _mean_over_seeds(el, noise=0.7) - 0.03


def test_spike_detection_drops_with_delay_and_missing_electrodes():
    """Beleg für die Preset-Hilfen: Spitzen-F1 fällt bei Verzögerung 8 auf etwa 0.65 und bei drei Elektroden auf etwa 0.7."""
    f1 = lambda a: a.methods["ica"].f1
    assert 0.55 < _mean_over_seeds(f1, delay=8) < 0.78 and 0.6 < _mean_over_seeds(f1, n=3) < 0.8


def test_neuron_kurtosis_is_in_the_stated_range_and_rhythm_is_minus_one_and_a_half():
    """Beleg für die Texte 'Kurtosis um 30-100' und 'Rhythmus -1.5' (auch in der Preset-Hilfe: Neuronen +35 bis +99 für den Standard-Seed)."""
    for s in (7,) + C.SWEEP_SEEDS:
        k = ev.excess_kurtosis(ev.make_dataset(g=2, kind="rhythm", seed=s).S)
        assert (k[:4] > 25).all() and (k[:4] < 130).all() and np.allclose(k[4:], -1.5, atol=0.1)
    k7 = ev.excess_kurtosis(ev.make_dataset().S)
    assert 30 < k7.min() < 40 and 90 < k7.max() < 105


def test_pca_components_are_more_gaussian_than_the_sources_and_ica_components_are_not():
    """Beleg für die Bildunterschrift der Kurtosis-Grafik."""
    ratios_pca, ratios_ica = [], []
    for s in C.SWEEP_SEEDS:
        a = ev.analyse(ev.make_dataset(seed=s), ev.Settings())
        src = a.kurtosis["sources"].mean()
        ratios_pca.append(a.kurtosis["pca"].mean() / src)
        ratios_ica.append(a.kurtosis["ica"].mean() / src)
    assert np.mean(ratios_pca) < 0.5 and np.mean(ratios_ica) > 0.75


def test_contrast_functions_table():
    """Beleg für die Hilfe zur Kontrastfunktion: log cosh und Kurtosis liefern bei allen Starts dasselbe, die Gauß-Ableitung landet bei einzelnen Starts schlechter; Kurtosis braucht die wenigsten Iterationen."""
    rows = {(r["contrast"], r["method"]): r for r in ev.contrast_table()}
    assert len(rows) == 6
    for key in (("logcosh", "symmetric"), ("cube", "symmetric")):
        assert rows[key]["spread"] < 1e-3 and rows[key]["worst"] > 0.97 and rows[key]["converged"] == 1.0
    assert rows[("exp", "symmetric")]["worst"] < 0.93 and rows[("exp", "symmetric")]["spread"] > 0.01
    assert rows[("cube", "symmetric")]["iterations"] < rows[("logcosh", "symmetric")]["iterations"] < rows[("exp", "symmetric")]["iterations"]


def test_start_only_changes_order_and_sign_for_the_robust_contrasts():
    """Beleg für die Hilfe zum Start und den Abschnitt 🔀."""
    ds = ev.make_dataset()
    runs = ev.ambiguity(ds, starts=(1, 2, 3, 4))
    assert max(r["corr"].mean() for r in runs) - min(r["corr"].mean() for r in runs) < 1e-6
    assert len({tuple(r["idx"]) for r in runs}) > 1 and len({tuple(r["signs"]) for r in runs}) > 1
    assert all(len(r["estimates"]) == 4 for r in runs) and ev.rescaling_invariance(ds) < 1e-12


def test_gauss_limit_table():
    """Beleg für Verdict-Text und Tabelle: eine Gauß-Quelle ist trennbar (SIR wie beim Rhythmus), zwei nicht (klar darunter, mit großer Streuung); das Rhythmus-Paar wird in jedem Datensatz gleich gut getrennt."""
    rows = {(r["kind"], r["g"]): r for r in ev.gauss_limit()}
    assert abs(rows[("gauss", 1)]["sir"] - rows[("rhythm", 1)]["sir"]) < 1.5
    assert rows[("gauss", 2)]["sir"] < rows[("gauss", 1)]["sir"] - 2.0 and rows[("gauss", 2)]["sir"] < rows[("rhythm", 2)]["sir"] - 2.0
    assert rows[("rhythm", 2)]["sir_max"] - rows[("rhythm", 2)]["sir_min"] < 1.0 and rows[("gauss", 2)]["sir_max"] - rows[("gauss", 2)]["sir_min"] > 3.0
    assert -3.0 < rows[("gauss", 2)]["sir_min"] and rows[("gauss", 2)]["sir_max"] < 8.5
    assert all(rows[key]["neuron_corr"] > 0.9 for key in rows)


def test_didactic_pair_finds_the_true_directions():
    ds = ev.make_dataset()
    p = ev.didactic_pair(ds)
    assert p.S.shape == p.X.shape == p.Z.shape == p.Y.shape == (2, ds.S.shape[1])
    d = np.abs(np.sort(np.degrees(p.true_angles)) - np.sort(np.degrees(p.ica_angles)))
    assert d.max() < 3.0 and abs(abs(np.degrees(p.true_angles[0] - p.true_angles[1])) - 90) < 3.0     # im weißen Raum stehen die Richtungen (fast) senkrecht: die Quellen sind in der Stichprobe nur ungefähr unkorreliert
    top = p.angles[np.argsort(p.contrast)[-5:]]
    assert min(abs(np.degrees(top) - np.degrees(p.true_angles[0]))) < 6 or min(abs(np.degrees(top) - np.degrees(p.true_angles[1]))) < 6


def test_footprint_amplitude_falls_with_distance_and_shifts_with_delay():
    ds = ev.make_dataset(noise=0.0, delay=6)
    t, w = ev.footprint(ds, 0)
    assert w.shape == (6, len(t)) and np.abs(w).max(axis=1)[0] > np.abs(w).max(axis=1)[-1]
    peak_positions = w.argmin(axis=1)
    assert peak_positions.max() > peak_positions.min()
    t0, w0 = ev.footprint(ev.make_dataset(noise=0.0), 0)
    assert len(set(w0.argmin(axis=1))) == 1
