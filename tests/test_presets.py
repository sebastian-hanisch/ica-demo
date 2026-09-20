"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert, bewusst weit)."""

import pytest

import ica_constants as C
from ica_evaluation import Settings, analyse, make_dataset, verdict


def _measure(p):
    ds = make_dataset(p["m"], p["n"], p["g"], p["kind"], p["noise"], p["delay"], p["n_samples"], p["seed"])
    a = analyse(ds, Settings(contrast=p["contrast"], init_start=p["init_start"]))
    code, data = verdict(a)[1:]
    return {"verdict": code, "ica": a.methods["ica"].neuron_corr, "pca": a.methods["pca"].neuron_corr, "electrodes": a.methods["electrodes"].neuron_corr, "f1": a.methods["ica"].f1, "bg_sir": a.bg_sir}


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_NEURONS_MIN <= p["m"] <= C.N_NEURONS_MAX and C.N_ELECTRODES_MIN <= p["n"] <= C.N_ELECTRODES_MAX and C.N_BACKGROUND_MIN <= p["g"] <= C.N_BACKGROUND_MAX
        assert p["kind"] in C.BACKGROUND_KINDS and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX and C.DELAY_MIN <= p["delay"] <= C.DELAY_MAX
        assert C.N_SAMPLES_MIN <= p["n_samples"] <= C.N_SAMPLES_MAX and p["n_samples"] % 1000 == 0 and p["contrast"] in C.CONTRASTS and p["init_start"] in C.INIT_STARTS


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if isinstance(expected, str):
            assert value == expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"


def test_gauss_pair_preset_is_worse_than_the_rhythm_preset_at_separating_the_pair():
    """Beleg für die Preset-Hilfen: Rhythmus-Paar um 8 dB in jedem Datensatz; Gauß-Paar zwischen 0 und 7 dB (Mittel klar darunter)."""
    rhythm = [_measure({**C.PRESETS["Zwei Rhythmus-Quellen"], "seed": s})["bg_sir"] for s in (7, 100000, 100001, 100002)]
    gauss = [_measure({**C.PRESETS["Zwei Gauß-Quellen"], "seed": s})["bg_sir"] for s in (7, 100000, 100001, 100002)]
    assert all(7.0 < v < 9.0 for v in rhythm) and all(-3.0 < v < 8.5 for v in gauss) and sum(gauss) / 4 < sum(rhythm) / 4 - 2.0
