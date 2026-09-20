"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe ae_presets.py in autoencoder-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ica_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_neurons_slider": SettingSpec("m", int, C.DEFAULT_N_NEURONS, C.N_NEURONS_MIN, C.N_NEURONS_MAX),
    "n_electrodes_slider": SettingSpec("n", int, C.DEFAULT_N_ELECTRODES, C.N_ELECTRODES_MIN, C.N_ELECTRODES_MAX),
    "n_background_slider": SettingSpec("g", int, C.DEFAULT_N_BACKGROUND, C.N_BACKGROUND_MIN, C.N_BACKGROUND_MAX),
    "kind_select": SettingSpec("kind", _choice(C.BACKGROUND_KINDS), C.DEFAULT_BACKGROUND_KIND),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "delay_slider": SettingSpec("delay", int, C.DEFAULT_DELAY, C.DELAY_MIN, C.DELAY_MAX),
    "n_samples_slider": SettingSpec("T", int, C.DEFAULT_N_SAMPLES, C.N_SAMPLES_MIN, C.N_SAMPLES_MAX),
    "contrast_select": SettingSpec("contrast", _choice(C.CONTRASTS), C.DEFAULT_CONTRAST),
    "init_start_select": SettingSpec("start", int, C.DEFAULT_INIT_START, min(C.INIT_STARTS), max(C.INIT_STARTS)),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"m": "n_neurons_slider", "n": "n_electrodes_slider", "g": "n_background_slider", "kind": "kind_select", "noise": "noise_slider", "delay": "delay_slider",
               "n_samples": "n_samples_slider", "contrast": "contrast_select", "init_start": "init_start_select", "seed": "seed_input"}


def init_session_state_defaults():
    """Fehlende Zustände auffüllen; die Art des Hintergrunds (bei 0 Hintergrundquellen ausgeblendet, dann vom Widget-Zustand gelöscht) kehrt zum zuletzt gewählten Wert zurück."""
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get("_kind_kept", spec.default) if state_key == "kind_select" else spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["init_start_select"] = int(min(C.INIT_STARTS, key=lambda s: abs(s - st.session_state.get("init_start_select", C.DEFAULT_INIT_START))))
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
    st.session_state["_kind_kept"] = C.PRESETS[name]["kind"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
