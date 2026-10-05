"""Auswertung der ICA-Demo: Zuordnung, Kennzahlen (Korrelation, SIR, Amari-Index, Spike-Treffer), Baselines."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import ica_algorithm as alg
import ica_constants as C
import ica_scenario as sc


@dataclass(frozen=True)
class Settings:
    contrast: str = C.DEFAULT_CONTRAST
    method: str = C.DEFAULT_METHOD
    init_start: int = C.DEFAULT_INIT_START


def make_dataset(m=C.DEFAULT_N_NEURONS, n=C.DEFAULT_N_ELECTRODES, g=C.DEFAULT_N_BACKGROUND, kind=C.DEFAULT_BACKGROUND_KIND, noise=C.DEFAULT_NOISE,
                 delay=C.DEFAULT_DELAY, n_samples=C.DEFAULT_N_SAMPLES, seed=C.DEFAULT_SEED):
    return sc.make_dataset(m, n, g, kind, noise, delay, n_samples, seed)


def n_components(ds):
    """Die Zahl der Quellen wird als bekannt angenommen (Standardannahme von FastICA); bei weniger Elektroden als Quellen bleibt nur n."""
    return min(ds.n_electrodes, ds.S.shape[0])


def correlation_matrix(S, estimates):
    """|Korrelation| (k, nc) zwischen wahren Quellen und Schätzungen."""
    a = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    b = estimates - estimates.mean(axis=1, keepdims=True)
    sd = b.std(axis=1, keepdims=True)
    b = b / np.where(sd > 0, sd, 1.0)
    return np.abs(a @ b.T) / S.shape[1]


def assign(C_abs):
    """Optimale Zuordnung Quelle -> Schätzung (jede Schätzung höchstens einer Quelle), Summe der |Korrelationen| maximal. Exakt per Bitmasken-DP.
    Rückgabe: Liste je Quelle mit dem Index der Schätzung oder -1 (nicht zugeordnet, nur wenn es weniger Schätzungen als Quellen gibt)."""
    k, nc = C_abs.shape
    best = {}

    def solve(row, used):
        if row == k:
            return 0.0, ()
        key = (row, used)
        if key in best:
            return best[key]
        result = (solve(row + 1, used)[0], (-1,) + solve(row + 1, used)[1])
        for j in range(nc):
            if not used >> j & 1:
                value, rest = solve(row + 1, used | 1 << j)
                if value + C_abs[row, j] > result[0] + 1e-12:
                    result = (value + C_abs[row, j], (j,) + rest)
        best[key] = result
        return result

    return list(solve(0, 0)[1])


def matched(S, estimates):
    """Zuordnung + Vorzeichen-/Skalenkorrektur per Regression: (Zuordnung, |Korrelation| je Quelle, Schätzquellen in Skala und Vorzeichen der wahren Quellen)."""
    cm = correlation_matrix(S, estimates)
    idx = assign(cm)
    corr = np.array([cm[i, j] if j >= 0 else 0.0 for i, j in enumerate(idx)])
    aligned = np.zeros_like(S)
    for i, j in enumerate(idx):
        if j >= 0:
            e = estimates[j] - estimates[j].mean()
            aligned[i] = (e @ (S[i] - S[i].mean()) / (e @ e)) * e + S[i].mean()
    return idx, corr, aligned


def sir_db(corr):
    """Signal-zu-Interferenz in dB aus der Korrelation: rho^2 / (1 - rho^2)."""
    r2 = np.clip(np.asarray(corr) ** 2, 1e-9, 1.0 - 1e-9)
    return 10.0 * np.log10(r2 / (1.0 - r2))


def amari_index(P):
    """Amari-Index einer (nc, k)-Matrix P = Entmischung x Mischung; 0 = perfekt (nur Permutation und Skalierung), 1 = schlechtestmöglich; verallgemeinert auf nicht quadratische P."""
    P = np.abs(P)
    k = P.shape[1]
    rows = (P.sum(axis=1) / P.max(axis=1) - 1.0).sum()
    cols = (P.sum(axis=0) / P.max(axis=0) - 1.0).sum()
    d = max(k, P.shape[0])
    return float((rows + cols) / (2.0 * d * (d - 1))) if d > 1 else 0.0


# --- Spike-Erkennung auf den Schätzquellen ------------------------------------------------------------------------------------
DETECT_MIN_SEPARATION = 15         # Abtastwerte zwischen zwei erkannten Spitzen
DETECT_TOLERANCE = 4               # erlaubte Abweichung zur wahren Spitze
DETECT_SIGMA_FACTOR = 4.0          # Schwelle: 4 robuste Standardabweichungen (MAD) ...
DETECT_DEPTH_FRACTION = 0.3        # ... mindestens aber 30 % der typischen Spitzentiefe (sonst würde ein rauschfreies Signal jeden Rest melden)


def detect_spikes(x):
    """Negative Spitzen von x (Vorzeichen bereits wie beim Neuron): tiefste zuerst, Mindestabstand; Schwelle max(4 sigma_MAD, 0.3 x typische Tiefe); typische Tiefe = Median der (höchstens) 10 tiefsten Spitzen."""
    sigma = np.median(np.abs(x - np.median(x))) / 0.6745
    threshold = -DETECT_SIGMA_FACTOR * sigma
    taken = np.zeros(len(x), bool)
    peaks = []
    for t in np.argsort(x):
        if x[t] > threshold:
            break
        if taken[max(0, t - DETECT_MIN_SEPARATION): t + DETECT_MIN_SEPARATION + 1].any():
            continue
        taken[t] = True
        peaks.append(t)
        if len(peaks) == 10:                                             # typische Tiefe = Median der 10 tiefsten Spitzen (ab hier gilt die Schwelle laufend)
            threshold = min(threshold, DETECT_DEPTH_FRACTION * float(np.median(x[peaks])))
    if 0 < len(peaks) < 10:                                              # weniger als 10 Spitzen: typische Tiefe = Median der gefundenen
        threshold = min(threshold, DETECT_DEPTH_FRACTION * float(np.median(x[peaks])))
        peaks = [t for t in peaks if x[t] <= threshold]
    return np.sort(np.array(peaks, dtype=int))


def spike_f1(detected, true):
    """F1 der erkannten gegen die wahren Spitzenzeiten (Zuordnung je wahrer Spitze zur nächsten, jede erkannte höchstens einmal)."""
    if len(detected) == 0 or len(true) == 0:
        return 0.0
    used = np.zeros(len(detected), bool)
    tp = 0
    for t in true:
        d = np.abs(detected - t)
        j = int(np.argmin(d))
        if d[j] <= DETECT_TOLERANCE and not used[j]:
            used[j] = True
            tp += 1
    if tp == 0:
        return 0.0
    precision, recall = tp / len(detected), tp / len(true)
    return 2 * precision * recall / (precision + recall)


# --- Kennzahlen je Verfahren ------------------------------------------------------------------------------------------------------


def excess_kurtosis(x):
    x = np.asarray(x, dtype=float)
    x = (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)
    return (x ** 4).mean(axis=-1) - 3.0


@dataclass(frozen=True)
class Method:
    name: str
    idx: list                     # je Quelle: Index der zugeordneten Schätzung (-1 = keine)
    corr: np.ndarray              # |Korrelation| je Quelle
    aligned: np.ndarray           # (k, T) zugeordnete Schätzungen in Vorzeichen und Skala der wahren Quellen
    neuron_corr: float
    neuron_sir: float
    f1: float
    estimates: np.ndarray         # (nc, T) rohe Schätzungen


def evaluate_method(name, ds, estimates):
    idx, corr, aligned = matched(ds.S, estimates)
    m = ds.n_neurons
    f1 = float(np.mean([spike_f1(detect_spikes(aligned[i]), ds.spike_times[i]) for i in range(m)]))
    return Method(name, idx, corr, aligned, float(corr[:m].mean()), float(sir_db(corr[:m]).mean()), f1, estimates)


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    model: alg.ICAModel
    methods: dict                 # "electrodes", "pca", "ica" -> Method
    amari: dict                   # "pca", "ica"
    bg_sir: float                 # mittlere SIR der Hintergrundquellen (nan ohne Hintergrund)
    ref_clean: float              # ICA auf demselben Datensatz ohne Rauschen und ohne Verzögerung (Obergrenze); nan wenn nicht nötig
    ref_instant: float            # ICA auf demselben Datensatz ohne Verzögerung, mit demselben Rauschen; nan wenn nicht nötig
    kurtosis: dict                # "sources", "pca", "ica" -> Exzess-Kurtosis je Zeile


def fit(ds, settings):
    return alg.fit_ica(ds.X, n_components(ds), settings.contrast, settings.method, settings.init_start)


def analyse(ds, settings):
    nc = n_components(ds)
    model = fit(ds, settings)
    pca = alg.pca_components(ds.X, nc)
    methods = {"electrodes": evaluate_method("electrodes", ds, ds.X), "pca": evaluate_method("pca", ds, pca), "ica": evaluate_method("ica", ds, model.sources)}
    amari = {"pca": amari_index(alg.whiten(ds.X, nc).K @ ds.A), "ica": amari_index(model.unmixing @ ds.A)}
    m = ds.n_neurons
    ica_m = methods["ica"]
    bg_sir = float(sir_db(ica_m.corr[m:]).mean()) if ds.S.shape[0] > m else float("nan")
    ref_clean = ref_instant = float("nan")
    if ds.delays.any() or ds.noise_sigma > 0:
        base = dict(m=m, n=ds.n_electrodes, g=ds.S.shape[0] - m, kind=ds.kinds[-1] if ds.S.shape[0] > m else C.DEFAULT_BACKGROUND_KIND, n_samples=ds.S.shape[1], seed=ds.seed)
        noise_rel = ds.noise_sigma / max(_neuron_signal_std(ds), 1e-12)
        clean = make_dataset(noise=0.0, delay=0, **base)
        ref_clean = _neuron_corr(clean, settings)
        ref_instant = _neuron_corr(make_dataset(noise=noise_rel, delay=0, **base), settings) if ds.delays.any() else float(methods["ica"].neuron_corr)
    kurtosis = {"sources": excess_kurtosis(ds.S), "pca": excess_kurtosis(pca), "ica": excess_kurtosis(model.sources)}
    return Analysis(ds, settings, model, methods, amari, bg_sir, ref_clean, ref_instant, kurtosis)


def _neuron_signal_std(ds):
    part = np.zeros((ds.n_electrodes, ds.S.shape[1]))
    for j in range(ds.n_electrodes):
        for i in range(ds.n_neurons):
            part[j] += ds.A[j, i] * sc.shift(ds.S[i], ds.delays[j, i])
    return float(part.std())


def _neuron_corr(ds, settings):
    model = fit(ds, settings)
    return float(matched(ds.S, model.sources)[1][:ds.n_neurons].mean())

# --- Sweeps, Tabellen, Zusatzexperimente -------------------------------------------------------------------------------------------


def _row(x, values):
    out = {"x": x}
    for key in ("ica", "pca", "electrodes", "f1", "amari"):
        arr = np.array([v[key] for v in values])
        out[key], out[key + "_std"] = float(arr.mean()), float(arr.std())
    out["ica_min"] = float(min(v["ica"] for v in values))
    return out


_SWEEP_KEYWORD = {"noise": "noise", "delay": "delay", "n_electrodes": "n", "n_samples": "n_samples"}


def sweep(parameter, values=None, settings=Settings(), **base):
    """ICA-Qualität (Mittel und Streuung über die festen Sweep-Datensätze) in Abhängigkeit von einem Regler; alle anderen Regler bleiben wie in `base`."""
    values = C.SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        kw = dict(base)
        kw[_SWEEP_KEYWORD[parameter]] = x
        per_seed = []
        for seed in C.SWEEP_SEEDS:
            a = analyse(make_dataset(seed=seed, **kw), settings)
            per_seed.append({"ica": a.methods["ica"].neuron_corr, "pca": a.methods["pca"].neuron_corr, "electrodes": a.methods["electrodes"].neuron_corr,
                             "f1": a.methods["ica"].f1, "amari": a.amari["ica"]})
        rows.append(_row(x, per_seed))
    return rows


def contrast_table(**base):
    """Kontrastfunktion x Verfahren: mittlere und schlechteste Korrelation über 5 Startwerte und die festen Datensätze, Iterationen, Anteil konvergierter Läufe."""
    rows = []
    for contrast in C.CONTRASTS:
        for method in C.METHODS:
            corrs, iters, conv, spread = [], [], [], []
            for seed in C.SWEEP_SEEDS:
                ds = make_dataset(seed=seed, **base)
                one = []
                for start in C.INIT_STARTS:
                    model = alg.fit_ica(ds.X, n_components(ds), contrast, method, start, keep_history=False)
                    one.append(float(matched(ds.S, model.sources)[1][:ds.n_neurons].mean()))
                    iters.append(model.n_iter), conv.append(model.converged)
                corrs += one
                spread.append(max(one) - min(one))
            rows.append({"contrast": contrast, "method": method, "mean": float(np.mean(corrs)), "worst": float(np.min(corrs)), "iterations": float(np.mean(iters)),
                         "converged": float(np.mean(conv)), "spread": float(np.mean(spread))})
    return rows


def gauss_limit(**base):
    """Gauß-Grenze: Trennung der Hintergrundquellen (SIR in dB, Mittel und Spanne über die Datensätze) je Art und Anzahl."""
    rows = []
    for kind, g in (("gauss", 0), ("gauss", 1), ("gauss", 2), ("rhythm", 1), ("rhythm", 2)):
        sirs, neurons = [], []
        for seed in C.SWEEP_SEEDS:
            a = analyse(make_dataset(g=g, kind=kind, seed=seed, **base), Settings())
            neurons.append(a.methods["ica"].neuron_corr)
            if g:
                sirs.append(a.bg_sir)
        rows.append({"kind": kind, "g": g, "neuron_corr": float(np.mean(neurons)), "sir": float(np.mean(sirs)) if sirs else float("nan"),
                     "sir_min": float(np.min(sirs)) if sirs else float("nan"), "sir_max": float(np.max(sirs)) if sirs else float("nan")})
    return rows


def ambiguity(ds, starts=(1, 2), contrast=C.DEFAULT_CONTRAST):
    """Zwei Startwerte, dieselben Daten: Reihenfolge, Vorzeichen und Skala der Schätzungen unterscheiden sich, die Zuordnung nach Regression nicht."""
    out = []
    for start in starts:
        model = alg.fit_ica(ds.X, n_components(ds), contrast, "symmetric", start, keep_history=False)
        est = model.sources
        idx, corr, aligned = matched(ds.S, est)
        signs, scales = [], []
        for i, j in enumerate(idx):
            if j < 0:
                signs.append(0), scales.append(float("nan"))
                continue
            e = est[j] - est[j].mean()
            coef = float(e @ (ds.S[i] - ds.S[i].mean()) / (e @ e))
            signs.append(int(np.sign(coef))), scales.append(abs(coef))
        out.append({"start": start, "idx": idx, "signs": signs, "scales": scales, "corr": corr, "aligned": aligned, "estimates": est})
    return out


def rescaling_invariance(ds, factors=None):
    """X = A S = (A D^-1)(D S): jede Skalierung der Quellen lässt sich in die Mischung schieben - größte Abweichung der Elektrodensignale."""
    k = ds.S.shape[0]
    d = np.array(factors if factors is not None else [3.0, -0.5, 7.0, 0.2, -2.0, 4.0, 1.5][:k])
    return float(np.abs(ds.A @ ds.S - (ds.A / d) @ (ds.S * d[:, None])).max())


# --- Zwei-Quellen-Anschauung ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    S: np.ndarray                 # (2, T) zwei Neuronen
    X: np.ndarray                 # (2, T) gemischt (2 Elektroden, ohne Rauschen und Verzögerung)
    Z: np.ndarray                 # (2, T) weiß (PCA)
    Y: np.ndarray                 # (2, T) nach der ICA-Rotation
    angles: np.ndarray            # Winkel 0..pi
    contrast: np.ndarray          # Nicht-Gaußianität der Projektion cos(a) z1 + sin(a) z2
    true_angles: np.ndarray       # Richtungen der beiden wahren Quellen im weißen Raum (0..pi)
    ica_angles: np.ndarray        # Zeilen von W als Winkel (0..pi)


def _best_electrode_pair(A2):
    """Die beiden Elektroden, an denen sich zwei Neuronen am besten unterscheiden (größte normierte Determinante der 2x2-Mischung)."""
    n = A2.shape[0]
    best, rows = -1.0, [0, min(1, n - 1)]
    for a in range(n):
        for b in range(a + 1, n):
            sub = A2[[a, b]]
            score = abs(np.linalg.det(sub)) / (np.linalg.norm(sub[0]) * np.linalg.norm(sub[1]) + 1e-12)
            if score > best:
                best, rows = score, [a, b]
    return rows


def didactic_pair(ds, contrast=C.DEFAULT_CONTRAST, pair=(0, 1)):
    """Dieselbe Idee in zwei Dimensionen: zwei der Neuronen, die zwei Elektroden mit der bestkonditionierten Mischung, ohne Rauschen und Verzögerung."""
    i, j = pair
    S2 = ds.S[[i, j]]
    A2 = ds.A[_best_electrode_pair(ds.A[:, [i, j]])][:, [i, j]]
    X2 = A2 @ S2
    wh = alg.whiten(X2, 2)
    model = alg.fit_ica(X2, 2, contrast, "symmetric", 1, keep_history=False)
    angles = np.linspace(0, np.pi, 181)
    J = np.array([alg.non_gaussianity(contrast, np.cos(a) * wh.Z[0] + np.sin(a) * wh.Z[1]) for a in angles])
    true_dirs = wh.K @ A2
    true_angles = np.mod(np.arctan2(true_dirs[1], true_dirs[0]), np.pi)
    ica_angles = np.mod(np.arctan2(model.W[:, 1], model.W[:, 0]), np.pi)
    return Pair(S2, X2, wh.Z, model.sources, angles, J, true_angles, ica_angles)


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

VERDICT_DELAY_GAP = 0.08          # Verlust durch die Verzögerung (Korrelation gegenüber demselben Datensatz ohne Verzögerung)
VERDICT_NOISE_GAP = 0.20          # Verlust durch das Rauschen (gegenüber demselben Datensatz ohne Rauschen)
VERDICT_WIN_MARGIN = 0.15         # ICA-Vorsprung vor PCA (Korrelation der Neuronen)


def verdict(a):
    """(Art, Code, Kennzahlen). Nur mit großer Marge - kein Urteil nahe an einer Schwelle."""
    ds = a.ds
    k, m = ds.S.shape[0], ds.n_neurons
    g = k - m
    ica, pca = a.methods["ica"], a.methods["pca"]
    data = {"ica": ica.neuron_corr, "pca": pca.neuron_corr, "f1": ica.f1, "pca_f1": pca.f1, "electrodes_f1": a.methods["electrodes"].f1,
            "amari": a.amari["ica"], "n_iter": a.model.n_iter, "ref_clean": a.ref_clean, "ref_instant": a.ref_instant, "bg_sir": a.bg_sir}
    if ds.n_electrodes < k:
        return "warning", "underdetermined", data
    if not a.model.converged:
        return "warning", "not_converged", data
    if g >= 2 and ds.kinds[-1] == "gauss":
        return "warning", "gauss_pair", data
    if ds.delays.any() and a.ref_instant - ica.neuron_corr > VERDICT_DELAY_GAP:
        return "warning", "delay", data
    if ds.noise_sigma > 0 and a.ref_clean - a.ref_instant > VERDICT_NOISE_GAP:
        return "warning", "noise", data
    if ica.neuron_corr - pca.neuron_corr > VERDICT_WIN_MARGIN:
        return "success", "ica_wins", data
    return "info", "neutral", data


# --- Verlaufs- und Anschauungsdaten ---------------------------------------------------------------------------------------------------


def iteration_curve(a):
    """Mittlere |Korrelation| der Neuronen nach jeder Iteration; Index 0 = zufälliger Start (symmetrisches Verfahren, sonst nur der Endwert)."""
    model, ds = a.model, a.ds
    Z = model.whitening.Z
    m = ds.n_neurons
    if not model.history:
        return [a.methods["ica"].neuron_corr]
    rng = np.random.default_rng([a.settings.init_start, 4242])
    W0 = alg._sym_decorrelate(rng.standard_normal((Z.shape[0], Z.shape[0])))
    return [float(matched(ds.S, W @ Z)[1][:m].mean()) for W in (W0,) + model.history]


def footprint(ds, neuron=0, before=10, after=25):
    """Mittlere Spitzenform (ohne Rauschen) eines Neurons an jeder Elektrode: (Zeitachse in ms relativ zur Spitze an der nächsten Elektrode, (n, L) Wellenformen)."""
    peaks = [t for t in ds.spike_times[neuron] if before <= t < ds.X_clean.shape[1] - after - int(ds.delays.max())]
    segments = np.array([ds.X_clean[:, t - before: t + after] for t in peaks])
    return (np.arange(-before, after) * 1000.0 / C.SAMPLE_RATE), segments.mean(axis=0)
