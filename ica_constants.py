"""Defaults, Slider-Grenzen und feste Szenario-Größen der ICA-Demo (Presets folgen in ica_presets.py nach den Messungen)."""

# --- Szenario (fest) -----------------------------------------------------------------------------------------------------------
SAMPLE_RATE = 10_000                       # Hz
WAVEFORM_LENGTH = 30                       # Abtastwerte je Spike
PEAK_INDEX = 8                             # Lage der negativen Spitze in der Wellenform
OVERSHOOT = 0.4                            # Höhe des positiven Nachschlags
REFRACTORY = 20                            # Abtastwerte (2 ms)
NEURON_SIGMAS = (2.0, 3.0, 2.5, 4.0, 3.5)          # Breite der Spitze je Neuron
NEURON_RATES = (20.0, 28.0, 35.0, 24.0, 31.0)      # Feuerrate in Hz
NEURON_AMPLITUDES = (1.0, 0.8, 1.2, 0.7, 0.9)      # Spitzenamplitude an der nächsten Elektrode
NEURON_POSITIONS = ((0.10, 0.30), (0.35, 0.20), (0.60, 0.35), (0.85, 0.25), (0.50, 0.55))   # Elektroden liegen bei y = 0, x in [0, 1]
DISTANCE_EPS = 0.05
BACKGROUND_AMPLITUDE = 0.8
GAUSS_AR = (0.95, 0.5)                     # Autokorrelation der beiden Gauß-Hintergrundquellen
RHYTHM_FREQUENCIES = (10.0, 23.0)          # Hz
BACKGROUND_KINDS = ("gauss", "rhythm")
BACKGROUND_LABELS = {"gauss": "Gauß-Rauschen", "rhythm": "Rhythmus (sinusförmig)"}

# --- Regler ---------------------------------------------------------------------------------------------------------------------
DEFAULT_N_NEURONS = 4
N_NEURONS_MIN, N_NEURONS_MAX = 2, 5
DEFAULT_N_ELECTRODES = 6
N_ELECTRODES_MIN, N_ELECTRODES_MAX = 1, 8
DEFAULT_N_BACKGROUND = 0
N_BACKGROUND_MIN, N_BACKGROUND_MAX = 0, 2
DEFAULT_BACKGROUND_KIND = "gauss"
DEFAULT_NOISE = 0.05
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_DELAY = 0
DELAY_MIN, DELAY_MAX = 0, 12               # Abtastwerte je Einheitsabstand
DEFAULT_N_SAMPLES = 20_000
N_SAMPLES_MIN, N_SAMPLES_MAX = 1_000, 40_000
CONTRASTS = ("logcosh", "exp", "cube")
CONTRAST_LABELS = {"logcosh": "log cosh (robust)", "exp": "Gauß-Ableitung (sehr robust)", "cube": "Kurtosis (u³)"}
DEFAULT_CONTRAST = "logcosh"
METHODS = ("symmetric", "deflation")
METHOD_LABELS = {"symmetric": "symmetrisch (alle gleichzeitig)", "deflation": "Deflation (einzeln nacheinander)"}
DEFAULT_METHOD = "symmetric"
INIT_STARTS = (1, 2, 3, 4, 5)
DEFAULT_INIT_START = 1
DEFAULT_SEED = 7
MAX_ITER = 200
TOL = 1e-6

# --- Auswertung -----------------------------------------------------------------------------------------------------------------
SWEEP_SEEDS = (100000, 100001, 100002, 100003, 100004)            # feste Datensätze der Sweeps, getrennt vom Demo-Seed
SWEEP_VALUES = {
    "noise": (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
    "delay": (0, 1, 2, 4, 8, 12),
    "n_electrodes": (1, 2, 3, 4, 5, 6, 8),
    "n_samples": (1_000, 2_000, 5_000, 10_000, 20_000, 40_000),
}
SWEEP_LABELS = {"noise": "Rauschen (relativ zum Neuronen-Signal)", "delay": "Verzögerung (Abtastwerte je Einheitsabstand)",
                "n_electrodes": "Anzahl Elektroden", "n_samples": "Länge (Abtastwerte)"}

# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(m=DEFAULT_N_NEURONS, n=DEFAULT_N_ELECTRODES, g=DEFAULT_N_BACKGROUND, kind=DEFAULT_BACKGROUND_KIND, noise=DEFAULT_NOISE, delay=DEFAULT_DELAY, n_samples=DEFAULT_N_SAMPLES,
                contrast=DEFAULT_CONTRAST, init_start=DEFAULT_INIT_START, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Vier Neuronen, sechs Elektroden": _preset(),
    "Zwei Rhythmus-Quellen": _preset(g=2, kind="rhythm"),
    "Zwei Gauß-Quellen": _preset(g=2, kind="gauss"),
    "Zu wenige Elektroden": _preset(n=3),
    "Laufzeitverzögerung": _preset(delay=8),
    "Starkes Rauschen": _preset(noise=0.7),
}
PRESET_HELP = {
    "Vier Neuronen, sechs Elektroden": "Der Grundfall: vier Neuronen, sechs Elektroden, wenig Rauschen, keine Verzögerung. ICA trennt die vier Neuronen fast vollständig (Korrelation 0.98, alle Spitzen gefunden), "
                                       "die PCA nicht (0.65). Die beste Einzelelektrode liegt bei 0.73.",
    "Zwei Rhythmus-Quellen": "Zwei sinusförmige Hintergrundrhythmen kommen dazu (sechs Quellen, sechs Elektroden). Sinusquellen sind unter-Gaußsch (Kurtosis -1.5, Neuronen +35 bis +99) - ICA trennt beide Arten; "
                             "das Rhythmus-Paar erreicht in jedem Datensatz einen Signal-zu-Interferenz-Wert um 8 dB.",
    "Zwei Gauß-Quellen": "Zwei Gauß'sche Hintergrundquellen: eine einzelne wäre trennbar, zwei nicht (sie lassen sich beliebig drehen, ohne die Nicht-Gaußianität zu ändern). Die Neuronen werden weiter getrennt (0.92), "
                         "die Trennung des Paares ist Zufall: Signal-zu-Interferenz zwischen 0 und 7 dB je nach Datensatz.",
    "Zu wenige Elektroden": "Nur drei Elektroden für vier Neuronen: die Mischung ist nicht mehr umkehrbar. ICA findet nur drei Komponenten und die Korrelation fällt von 0.98 auf 0.68 (Spitzen-Trefferquote von 1.0 auf etwa 0.7) - "
                            "das ist der Fall der Sparse Component Analysis.",
    "Laufzeitverzögerung": "Das Signal erreicht ferne Elektroden später (bis 8 Abtastwerte je Einheitsabstand = 0.8 ms). Die Mischung ist nicht mehr momentan, die ICA-Annahme verletzt: Korrelation 0.76 statt 0.98, "
                           "Trefferquote der Spitzen etwa 0.65. Genau hier setzt die Verzögerungsgraph-Methode des Spike-Sortings an.",
    "Starkes Rauschen": "Sensorrauschen von 70 % des Neuronen-Signals. Entmischen verstärkt das Rauschen mit: ICA (Korrelation etwa 0.5) liegt sogar unter der besten Einzelelektrode (0.59); "
                        "erst bei weniger Rauschen (bis etwa 0.4) ist ICA wieder vorn.",
}
# Bänder: |Korrelation| der Neuronen (ica, pca, electrodes), Spitzen-F1 der ICA (f1), Urteil (verdict) - mit dem ausgelieferten Code kalibriert, bewusst weit
PRESET_EXPECTED_BANDS = {
    "Vier Neuronen, sechs Elektroden": {"ica": (0.95, 1.0), "pca": (0.55, 0.75), "electrodes": (0.65, 0.82), "f1": (0.95, 1.0), "verdict": "ica_wins"},
    "Zwei Rhythmus-Quellen": {"ica": (0.86, 0.97), "pca": (0.5, 0.65), "f1": (0.95, 1.0), "bg_sir": (5.0, 11.0), "verdict": "ica_wins"},
    "Zwei Gauß-Quellen": {"ica": (0.86, 0.97), "pca": (0.5, 0.65), "f1": (0.95, 1.0), "verdict": "gauss_pair"},
    "Zu wenige Elektroden": {"ica": (0.6, 0.75), "pca": (0.48, 0.6), "f1": (0.55, 0.8), "verdict": "underdetermined"},
    "Laufzeitverzögerung": {"ica": (0.68, 0.84), "pca": (0.52, 0.66), "f1": (0.45, 0.9), "verdict": "delay"},
    "Starkes Rauschen": {"ica": (0.4, 0.62), "pca": (0.34, 0.48), "electrodes": (0.52, 0.66), "verdict": "noise"},
}
