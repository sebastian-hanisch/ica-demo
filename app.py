"""Unabhängige Komponentenanalyse (ICA) an Mehrelektroden-Signalen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - die ICA (FastICA) - und lässt stattdessen das Beispiel wachsen.
Erstes Stück der Quellentrennung-Linie der "Konzepte"-Reihe: Wurzel, von der SOBI, NMF, Sparse Component Analysis und der Spike-Sorting-Zweig abgehen. Was die ICA gegenüber der bloßen
PCA-Weißung bringt und wo ihre Annahmen enden, wird hier gemessen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import ica_constants as C
from ica_evaluation import (
    Settings, ambiguity, analyse, contrast_table, correlation_matrix, didactic_pair, footprint, gauss_limit, iteration_curve, make_dataset, rescaling_invariance, sweep, verdict,
)
from ica_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ica_visualization import (
    BLUE,
    GREEN,
    ORANGE,
    build_ambiguity,
    build_contrast_curve,
    build_corr_heatmap,
    build_footprint,
    build_gauss_limit,
    build_iterations,
    build_kurtosis,
    build_layout,
    build_method_bars,
    build_pair_scatter,
    build_sweep,
    build_traces,
    source_color,
    source_labels,
)

st.set_page_config(page_title="ICA – Sebastian Hanisch", layout="wide")

STEP_LABELS = {1: "1 · Quellen", 2: "2 · Mischung", 3: "3 · Weißen", 4: "4 · Rotation suchen", 5: "5 · Ergebnis"}
WINDOW_WIDTH_MS = 60
SWEEP_OPTIONS = {"noise": "Rauschen", "delay": "Verzögerung", "n_electrodes": "Anzahl Elektroden", "n_samples": "Länge der Aufnahme"}


@st.cache_data(show_spinner=False)
def _dataset(m, n, g, kind, noise, delay, n_samples, seed):
    return make_dataset(m, n, g, kind, noise, delay, n_samples, seed)


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse(make_dataset(*data_params), settings)


@st.cache_data(show_spinner=False)
def _pair(data_params, contrast):
    return didactic_pair(make_dataset(*data_params), contrast)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings):
    m, n, g, kind, noise, delay, n_samples = base
    return sweep(parameter, settings=settings, m=m, n=n, g=g, kind=kind, noise=noise, delay=delay, n_samples=n_samples)


@st.cache_data(show_spinner=False)
def _contrast_table(base):
    m, n, g, kind, noise, delay, n_samples = base
    return contrast_table(m=m, n=n, g=g, kind=kind, noise=noise, delay=delay, n_samples=n_samples)


@st.cache_data(show_spinner=False)
def _gauss_limit(base):
    m, n, _g, _kind, noise, delay, n_samples = base
    return gauss_limit(m=m, n=n, noise=noise, delay=delay, n_samples=n_samples)


@st.cache_data(show_spinner=False)
def _ambiguity(data_params):
    return ambiguity(make_dataset(*data_params), starts=(1, 2, 3, 4))


st.title("🧠 Unabhängige Komponentenanalyse (ICA) an Mehrelektroden-Signalen")
st.markdown(
    """
Ein **Elektrodenarray** misst nicht einzelne Nervenzellen, sondern an jeder Elektrode eine **Mischung**: jedes Neuron ist an nahen Elektroden laut, an fernen leise - und alle feuern gleichzeitig.
Die **ICA** (FastICA) ist das erste Stück der **Quellentrennung-Linie** und geht von einer einzigen Annahme aus: Die Quellen sind **statistisch unabhängig** und **nicht Gauß'sch**, die Mischung ist **momentan und linear**.
Daraus gewinnt sie die Quellen zurück - **ohne das Array oder die Neuronen zu kennen**. Die PCA kann das nicht: Sie macht die Mischung nur **unkorreliert** (Weißen), aber
*unkorreliert ist nicht unabhängig* - es bleibt eine Drehung offen, die erst die ICA findet. Die Demo misst, was das bringt und wo die Annahmen enden: Gauß'sche Quellen, zu wenige Elektroden,
Laufzeitverzögerungen zwischen den Elektroden, Rauschen. Wie das Verfahren funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - erstes Stück (Wurzel) der Quellentrennung-Linie der \"Konzepte\"-Reihe - **ein** Verfahren "
    "an einem wachsenden Beispiel. Die Kante zur Dimensionsreduktion ist nur die PCA-Weißung, die hier als Schritt 3 vorkommt."
)

with st.expander("So funktioniert die ICA", expanded=True):
    st.markdown(
        """
**Das Modell.** Jede Elektrode misst eine gewichtete Summe der Quellen: $x(t) = A\\,s(t) + \\text{Rauschen}$. Gesucht ist die Entmischung $W$ mit $\\hat s = W x$ - ohne $A$ zu kennen.

1. **Weißen (PCA).** Die Elektrodensignale werden zentriert und so gedreht und skaliert, dass sie **unkorreliert mit Varianz 1** sind. Danach ist die Mischung nur noch eine **Drehung** der unbekannten Quellen -
   aber *welche* Drehung, sagt die Korrelation nicht.
2. **Rotation suchen (ICA).** Für jede Drehrichtung wird gemessen, wie **nicht-Gauß'sch** die Projektion der weißen Daten ist. Nach dem zentralen Grenzwertsatz sind Mischungen unabhängiger Quellen
   *Gauß'scher* als die Quellen selbst - die Richtungen mit der **größten Nicht-Gaußianität** sind also die Quellen. FastICA sucht sie mit einem schnellen **Fixpunkt-Verfahren**
   ($w \\leftarrow E[z\\,g(w^\\top z)] - E[g'(w^\\top z)]\\,w$, dann orthogonalisieren) und einer **Kontrastfunktion** $g$ (log cosh, Gauß-Ableitung oder Kurtosis).
3. **Ergebnis.** Die Schätzungen $\\hat s$ stimmen mit den Quellen überein - **bis auf Reihenfolge, Vorzeichen und Skalierung**, die prinzipiell nicht bestimmbar sind (Abschnitt 🔀).

Spikes sind ein dankbarer Fall: kurze, seltene, große Ausschläge - **stark nicht-Gauß'sch** (Kurtosis um 30-100, Gauß = 0). Die Grenzen zeigen die Abschnitte darunter: **zwei Gauß'sche Quellen** sind nicht trennbar,
bei **weniger Elektroden als Quellen** ist die Mischung nicht umkehrbar, und eine **Laufzeitverzögerung** zwischen den Elektroden macht die Mischung nicht mehr momentan.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_neurons = st.slider(
        "Neuronen", *bounds("n_neurons_slider"), key="n_neurons_slider",
        help="Spitzenartige Quellen mit festem Ort, fester Spitzenform und fester Feuerrate (20-35 Hz). Die Quellenzahl wird der ICA als bekannt vorgegeben.",
    )
    n_electrodes = st.slider(
        "Elektroden", *bounds("n_electrodes_slider"), key="n_electrodes_slider",
        help="Aufnahmestellen auf einer Zeile. Mit weniger Elektroden als Quellen ist die Mischung nicht umkehrbar: bei 4 Neuronen fällt die Korrelation von 0.98 (4 Elektroden) auf 0.68 (3) und 0.41 (2).",
    )
    n_background = st.slider(
        "Hintergrundquellen", *bounds("n_background_slider"), key="n_background_slider",
        help="Zusätzliche flächige Quellen, die auf alle Elektroden wirken (etwa Feldpotenzial-Rhythmen). Sie zählen zur Quellenzahl.",
    )
    if n_background > 0:
        kind = st.selectbox(
            "Art des Hintergrunds", C.BACKGROUND_KINDS, key="kind_select", format_func=lambda k: C.BACKGROUND_LABELS[k],
            help="Gauß-Rauschen ist Gauß'sch (Kurtosis 0) - eine solche Quelle ist trennbar, zwei nicht. Sinusförmige Rhythmen sind unter-Gauß'sch (Kurtosis -1.5) und trennbar.",
        )
        st.session_state["_kind_kept"] = kind
    else:
        kind = st.session_state.get("_kind_kept", C.DEFAULT_BACKGROUND_KIND)
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Sensorrauschen relativ zur Stärke des Neuronen-Signals. Die Korrelation der Neuronen sinkt von 0.98 (Rauschen 0.05) auf 0.69 (0.4) und 0.43 (1.0).",
    )
    delay = st.slider(
        "Laufzeitverzögerung", *bounds("delay_slider"), key="delay_slider",
        help="Verzögerung je Einheitsabstand in Abtastwerten (0.1 ms). 0 = Mischung momentan, ICA-Annahme erfüllt. Schon 2 senken die Korrelation von 0.98 auf 0.92, 8 auf 0.76.",
    )
    n_samples = st.slider(
        "Länge der Aufnahme", *bounds("n_samples_slider"), key="n_samples_slider", step=1000,
        help="Abtastwerte bei 10 kHz. Erstaunlich unkritisch: schon 1000 Abtastwerte (0.1 s) reichen für Korrelation 0.98 - bei diesem Rauschen begrenzt nicht die Datenmenge, sondern das Rauschen.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**FastICA**")
    contrast = st.selectbox(
        "Kontrastfunktion", C.CONTRASTS, key="contrast_select", format_func=lambda c: C.CONTRAST_LABELS[c],
        help="Wie 'Nicht-Gaußianität' gemessen wird. log cosh und Kurtosis liefern bei allen Starts dasselbe; die Gauß-Ableitung landet bei einzelnen Starts schlechter (schlechtester Start 0.84 gegen 0.98). Kurtosis braucht die wenigsten Iterationen.",
    )
    init_start = st.selectbox(
        "Start der Entmischung", C.INIT_STARTS, key="init_start_select", format_func=lambda i: f"Start {i}",
        help="Zufällige Anfangsrichtung. Bei log cosh und Kurtosis ändert sie nichts am Ergebnis - nur an Reihenfolge und Vorzeichen der Komponenten (Abschnitt 🔀).",
    )

    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für Spikezeiten, Hintergrund und Rauschen.")

sync_query_params({
    "n_neurons_slider": int(n_neurons), "n_electrodes_slider": int(n_electrodes), "n_background_slider": int(n_background), "kind_select": kind, "noise_slider": noise, "delay_slider": int(delay),
    "n_samples_slider": int(n_samples), "contrast_select": contrast, "init_start_select": int(init_start), "seed_input": int(seed),
})

data_params = (int(n_neurons), int(n_electrodes), int(n_background), kind, float(noise), int(delay), int(n_samples), int(seed))
settings = Settings(contrast=contrast, init_start=int(init_start))
with st.spinner("Trenne die Quellen..."):
    ds = _dataset(*data_params)
    analysis = _analysis(data_params, settings)
model = analysis.model
methods = analysis.methods
k = ds.S.shape[0]
labels = source_labels(ds)
colors = [source_color(i) for i in range(k)]
level, code, vd = verdict(analysis)
data_key = data_params + (settings,)

# --- ICA in Aktion ------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 ICA in Aktion")
if "ica_step" not in st.session_state or st.session_state.get("ica_step_owner") != data_key:
    st.session_state["ica_step"] = 1
    st.session_state["ica_step_owner"] = data_key
duration_ms = ds.S.shape[1] * 1000.0 / C.SAMPLE_RATE
max_start = int(duration_ms - WINDOW_WIDTH_MS)
if st.session_state.get("window_start", 0) > max_start:
    st.session_state["window_start"] = 0
step_col, play_col, win_col = st.columns([4, 2, 3])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="ica_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
with win_col:
    window = st.slider(f"Zeitfenster ({WINDOW_WIDTH_MS} ms) ab [ms]", 0, max_start, key="window_start", step=10, help="Welchen Ausschnitt der Aufnahme die Signalspuren zeigen.")
pair_ok = ds.n_electrodes >= 2
pair = _pair(data_params, contrast) if pair_ok else None
peaks = ds.spike_times
view_slot = st.empty()


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Die wahren Quellen** (unbekannt in der Praxis; hier zum Vergleich bekannt)")
            c1.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors, [peaks[i] if i < ds.n_neurons else [] for i in range(k)]), width="stretch", key="step_sources")
            c2.markdown("**Ort von Neuronen und Elektroden**")
            c2.plotly_chart(build_layout(ds), width="stretch", key="step_layout")
        elif current_step == 2:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Was die Elektroden messen**: jede Spur mischt alle Quellen")
            c1.plotly_chart(build_traces([f"E{j + 1}" for j in range(ds.n_electrodes)], list(ds.X), window, WINDOW_WIDTH_MS, height=None, normalise=False), width="stretch", key="step_electrodes")
            c2.markdown("**Spitzenform von Neuron 1 an jeder Elektrode** (ohne Rauschen)")
            t_axis, wave = footprint(ds, 0)
            c2.plotly_chart(build_footprint(t_axis, wave, source_color(0)), width="stretch", key="step_footprint")
        elif current_step == 3:
            if pair_ok:
                st.markdown("**Dasselbe in zwei Dimensionen** (Neuron 1 und 2, die zwei Elektroden mit der bestkonditionierten Mischung, ohne Rauschen und Verzögerung)")
                st.plotly_chart(build_pair_scatter(pair), width="stretch", key="step_pair")
            else:
                st.info("Mit nur einer Elektrode gibt es nichts zu weißen oder zu drehen - die Zwei-Quellen-Anschauung braucht mindestens zwei Elektroden.")
        elif current_step == 4:
            c1, c2 = st.columns(2)
            if pair_ok:
                c1.markdown("**Nicht-Gaußianität über den Drehwinkel** (zwei Quellen)")
                c1.plotly_chart(build_contrast_curve(pair, C.CONTRAST_LABELS[contrast]), width="stretch", key="step_contrast")
            else:
                c1.info("Die Kontrastkurve braucht mindestens zwei Elektroden.")
            c2.markdown("**FastICA an allen Quellen**: Korrelation mit den wahren Neuronen je Iteration")
            c2.plotly_chart(build_iterations(iteration_curve(analysis), model.n_iter if model.converged and model.history else None), width="stretch", key="step_iterations")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Wahre Quellen**")
            c1.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors), width="stretch", key="step_result_true")
            c2.markdown("**ICA-Schätzungen** (Reihenfolge, Vorzeichen und Skala der wahren Quellen angepasst)")
            c2.plotly_chart(build_traces(labels, list(methods["ica"].aligned), window, WINDOW_WIDTH_MS, colors), width="stretch", key="step_result_ica")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

if step == 1:
    st.caption(f"{ds.n_neurons} Neuronen{' und ' + str(k - ds.n_neurons) + ' Hintergrundquelle(n)' if k > ds.n_neurons else ''}. Die Dreiecke markieren die Zeitpunkte der Spitzen; jede Spur ist auf ihr eigenes Maximum skaliert. Spitzen sind selten und groß - deshalb stark nicht-Gauß'sch (Kurtosis {', '.join(f'{v:.0f}' for v in analysis.kurtosis['sources'][:ds.n_neurons])}).")
elif step == 2:
    st.caption(f"{ds.n_electrodes} Elektrode(n) messen jeweils eine gewichtete Summe aller {k} Quellen plus Rauschen (Rauschen = {noise:.2f} der Stärke des Neuronen-Signals). Rechts: dieselbe Spitze von Neuron 1 an jeder Elektrode - leiser mit dem Abstand"
               f"{', und bei Verzögerung ' + str(delay) + ' später an fernen Elektroden' if delay else ''}.")
elif step == 3:
    st.caption("Zwei Quellen als Punktwolke: unabhängige spitzenartige Quellen bilden ein **Kreuz**. Die Mischung verzerrt es. **Weißen** (PCA) macht die Achsen unkorreliert und gleich lang - aber das Kreuz steht noch schief (breit grün: wahre Quellenrichtungen, "
               "rot: was die ICA findet). Die Korrelation kann diese Drehung nicht sehen, die Form der Verteilung schon.")
elif step == 4:
    st.caption("Links: je Drehwinkel wird gemessen, wie nicht-Gauß'sch die Projektion ist - die Maxima liegen bei den wahren Quellenrichtungen (grün), FastICA landet dort (rot). "
               f"Rechts: derselbe Mechanismus im vollen Problem. {'Konvergiert nach ' + str(model.n_iter) + ' Iterationen.' if model.converged else 'Nicht konvergiert nach ' + str(model.n_iter) + ' Iterationen.'}")
else:
    st.caption("Reihenfolge, Vorzeichen und Skala der ICA-Ausgabe sind beliebig und hier nachträglich an die wahren Quellen angepasst (Zuordnung + Regression).")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die ICA gefunden hat - gegen PCA und die beste Einzelelektrode")
st.caption(
    "Kennzahlen: |Korrelation| jedes Neurons mit der ihm zugeordneten Schätzung (Zuordnung optimal), Signal-zu-Interferenz (SIR) daraus, Spitzen-F1 (Schwelle auf der zugeordneten Spur), "
    "Amari-Index der Entmischung (0 = perfekt bis auf Reihenfolge und Skala)."
)
ica_m, pca_m, el_m = methods["ica"], methods["pca"], methods["electrodes"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Korrelation der Neuronen (ICA)", f"{ica_m.neuron_corr:.2f}", delta=f"{ica_m.neuron_corr - pca_m.neuron_corr:+.2f} ggü. PCA", delta_color="normal",
          help="Mittlere |Korrelation| der Neuronen mit ihrer Schätzung. 1 = perfekt.")
m2.metric("Spitzen-F1 (ICA)", f"{ica_m.f1:.2f}", delta=f"{ica_m.f1 - pca_m.f1:+.2f} ggü. PCA", delta_color="normal",
          help="Wie viele der wahren Spitzen auf der Schätzung gefunden werden (Toleranz ±4 Abtastwerte), harmonisches Mittel aus Treffer- und Genauigkeit.")
m3.metric("Signal-zu-Interferenz", f"{ica_m.neuron_sir:.1f} dB", delta=f"{ica_m.neuron_sir - pca_m.neuron_sir:+.1f} dB ggü. PCA", delta_color="normal",
          help="Aus der Korrelation: ρ²/(1-ρ²) in dB - wie viel stärker die richtige Quelle ist als alle anderen in der Schätzung.")
m4.metric("Amari-Index (ICA)", f"{analysis.amari['ica']:.3f}", delta=f"{analysis.amari['ica'] - analysis.amari['pca']:+.3f} ggü. PCA", delta_color="inverse",
          help="Abstand von 'Entmischung mal Mischung' zu einer Permutations-Skalierungs-Matrix, 0 bis 1. Bei Verzögerung gegen die Amplituden-Mischung ohne Verzögerung gerechnet.")

if code == "underdetermined":
    st.warning(f"⚠️ Nur {ds.n_electrodes} Elektrode(n) für {k} Quellen: die Mischung ist nicht umkehrbar, ICA findet nur {min(ds.n_electrodes, k)} Komponenten. Korrelation der Neuronen {vd['ica']:.2f}, Spitzen-F1 {vd['f1']:.2f} - "
               "mehr Quellen als Sensoren ist der Fall der Sparse Component Analysis (nächste Stücke der Linie).")
elif code == "not_converged":
    st.warning(f"⚠️ Die Fixpunkt-Iteration ist nach {vd['n_iter']} Iterationen nicht konvergiert - das Ergebnis ist unzuverlässig. Anderer Start oder andere Kontrastfunktion probieren.")
elif code == "gauss_pair":
    st.warning(f"⚠️ Zwei Gauß'sche Hintergrundquellen: sie lassen sich beliebig drehen, ohne dass sich ihre Nicht-Gaußianität ändert - ICA kann sie prinzipiell nicht trennen (SIR des Paars hier {vd['bg_sir']:.1f} dB, je nach Datensatz "
               f"zwischen 0 und 7 dB, Zufall; Abschnitt 🚧). Die Neuronen werden trotzdem getrennt (Korrelation {vd['ica']:.2f}).")
elif code == "delay":
    st.warning(f"⚠️ Die Laufzeitverzögerung verletzt die Annahme der momentanen Mischung: Korrelation {vd['ica']:.2f} statt {vd['ref_instant']:.2f} ohne Verzögerung (gleiche Daten), Spitzen-F1 {vd['f1']:.2f}. "
               "Eine Entmischung mit einer einzigen Matrix kann Verzögerungen nicht ausgleichen - das ist der Ansatzpunkt der Verzögerungsgraph-Methode im Spike-Sorting.")
elif code == "noise":
    st.warning(f"⚠️ Das Rauschen kostet viel: Korrelation {vd['ica']:.2f} statt {vd['ref_clean']:.2f} ohne Rauschen. Entmischen verstärkt das Rauschen mit"
               f"{' - die beste Einzelelektrode (' + format(el_m.neuron_corr, '.2f') + ') ist bei diesem Rauschen sogar besser als die ICA' if el_m.neuron_corr > vd['ica'] else ''}.")
elif code == "ica_wins":
    st.success(f"✅ ICA trennt die Neuronen: Korrelation {vd['ica']:.2f} gegen {vd['pca']:.2f} (PCA) und {el_m.neuron_corr:.2f} (beste Einzelelektrode), Spitzen-F1 {vd['f1']:.2f} gegen {vd['pca_f1']:.2f} (PCA) und {vd['electrodes_f1']:.2f}. "
               f"Die PCA macht die Mischung nur unkorreliert; die Drehung zur Unabhängigkeit findet erst die ICA.")
else:
    st.info(f"ℹ️ Korrelation ICA {vd['ica']:.2f}, PCA {vd['pca']:.2f} - kein deutlicher Vorsprung.")

t1, t2 = st.columns(2)
with t1:
    st.markdown("**Was die Elektroden messen**")
    st.plotly_chart(build_traces([f"E{j + 1}" for j in range(ds.n_electrodes)], list(ds.X), window, WINDOW_WIDTH_MS, normalise=False), width="stretch", key="res_electrodes")
with t2:
    st.markdown("**Wahre Quellen**")
    st.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors, [peaks[i] if i < ds.n_neurons else [] for i in range(k)]), width="stretch", key="res_true")
t3, t4 = st.columns(2)
with t3:
    st.markdown("**PCA (nur weißen, keine Rotation)**")
    st.plotly_chart(build_traces(labels, list(pca_m.aligned), window, WINDOW_WIDTH_MS, colors), width="stretch", key="res_pca")
with t4:
    st.markdown("**ICA**")
    st.plotly_chart(build_traces(labels, list(ica_m.aligned), window, WINDOW_WIDTH_MS, colors), width="stretch", key="res_ica")

h1, h2, h3 = st.columns([2, 2, 2])
est_labels_pca = [f"PC{j + 1}" for j in range(pca_m.estimates.shape[0])]
est_labels_ica = [f"IC{j + 1}" for j in range(ica_m.estimates.shape[0])]
with h1:
    st.plotly_chart(build_corr_heatmap(correlation_matrix(ds.S, pca_m.estimates), labels, est_labels_pca, "PCA: |Korrelation|"), width="stretch", key="heat_pca")
with h2:
    st.plotly_chart(build_corr_heatmap(correlation_matrix(ds.S, ica_m.estimates), labels, est_labels_ica, "ICA: |Korrelation|"), width="stretch", key="heat_ica")
with h3:
    st.plotly_chart(build_method_bars(methods), width="stretch", key="method_bars")
st.caption("Links (Zeilen = wahre Quellen, Spalten = Komponenten): eine saubere Trennung hat in jeder Zeile und Spalte genau einen hellen Eintrag - die PCA verteilt jede Quelle auf mehrere Komponenten, die ICA ordnet jeder Quelle eine zu. Rechts: Korrelation und Spitzen-F1.")

st.markdown("---")

# --- Nicht-Gaußianität und Kontrastfunktion -----------------------------------------------------------------------------------------

st.subheader("📊 Nicht-Gaußianität und Kontrastfunktion")
pca_idx, ica_idx = methods["pca"].idx, methods["ica"].idx
kurt_true = analysis.kurtosis["sources"]
kurt_ica = np.array([analysis.kurtosis["ica"][j] if j >= 0 else np.nan for j in ica_idx])
kurt_pca = np.array([analysis.kurtosis["pca"][j] if j >= 0 else np.nan for j in pca_idx])
st.plotly_chart(build_kurtosis(labels, [("wahre Quellen", kurt_true, GREEN), ("PCA-Komponenten", kurt_pca, ORANGE), ("ICA-Komponenten", kurt_ica, BLUE)]), width="stretch", key="kurtosis")
st.caption(
    "Exzess-Kurtosis je Quelle (Gauß = 0; Spitzen weit darüber, Sinusrhythmen bei -1.5). Jede PCA-Komponente ist eine Mischung mehrerer Quellen und damit **Gauß'scher** (kleinere Kurtosis) als die Quellen; "
    "die ICA-Komponenten kommen ihnen nahe, weil die ICA genau diese Nicht-Gaußianität maximiert."
)
if st.button("Kontrastfunktionen und Startwerte vergleichen (dauert einige Sekunden)", key="contrast_start"):
    st.session_state["contrast_on"] = True
if st.session_state.get("contrast_on"):
    with st.spinner("Vergleiche 3 Kontrastfunktionen × 2 Verfahren × 5 Starts × 5 Datensätze..."):
        rows = _contrast_table(data_params[:7])
    st.table({
        "Kontrastfunktion": [C.CONTRAST_LABELS[r["contrast"]] for r in rows],
        "Verfahren": [C.METHOD_LABELS[r["method"]] for r in rows],
        "Korrelation (Mittel)": [f"{r['mean']:.3f}" for r in rows],
        "schlechtester Start": [f"{r['worst']:.3f}" for r in rows],
        "Iterationen": [f"{r['iterations']:.0f}" for r in rows],
        "konvergiert": [f"{100 * r['converged']:.0f} %" for r in rows],
    })
    st.caption("Mittel über 5 feste Datensätze und 5 Startwerte, mit den aktuellen Einstellungen der Aufnahme. 'Schlechtester Start' zeigt, ob ein Start in einem schlechteren Optimum landet.")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Rauschen, Verzögerung, Elektrodenzahl und Länge ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_OPTIONS), format_func=lambda p: SWEEP_OPTIONS[p], key="sweep_select")
current = {"noise": float(noise), "delay": int(delay), "n_electrodes": int(n_electrodes), "n_samples": int(n_samples)}[sweep_param]
with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
    rows = _sweep(sweep_param, data_params[:7], settings)
st.plotly_chart(build_sweep(rows, C.SWEEP_LABELS[sweep_param], current=current, log=(sweep_param == "n_samples")), width="stretch", key="sweep_chart")
st.caption("Mittel und Streuung der Korrelation der Neuronen über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste. Grün gestrichelt: Spitzen-F1 der ICA.")

st.markdown("---")

# --- Mehrdeutigkeiten -----------------------------------------------------------------------------------------------------------------

st.subheader("🔀 Was ICA nicht festlegen kann: Reihenfolge, Vorzeichen, Skala")
amb = _ambiguity(data_params)
neuron_pick = st.selectbox("Neuron", list(range(ds.n_neurons)), format_func=lambda i: labels[i], key="amb_neuron")
runs_traces = [(f"Start {run['start']}: IC{run['idx'][neuron_pick] + 1}", run["estimates"][run["idx"][neuron_pick]]) for run in amb if run["idx"][neuron_pick] >= 0]
st.plotly_chart(build_ambiguity(ds.S[neuron_pick], runs_traces, window, WINDOW_WIDTH_MS, neuron_pick), width="stretch", key="ambiguity")
st.table({
    "Start": [f"Start {run['start']}" for run in amb],
    f"Komponente von {labels[neuron_pick]}": [f"IC{run['idx'][neuron_pick] + 1}" if run["idx"][neuron_pick] >= 0 else "-" for run in amb],
    "Vorzeichen": [{1: "gleich", -1: "gespiegelt (−)", 0: "-"}[run["signs"][neuron_pick]] for run in amb],
    "Korrelation nach Zuordnung": [f"{run['corr'][neuron_pick]:.3f}" for run in amb],
})
st.caption(
    f"Dieselben Daten, vier Startwerte (log cosh, symmetrisch): dieselbe Quelle steckt in verschiedenen Komponenten und mit verschiedenem Vorzeichen - die Korrelation nach Zuordnung ist gleich. Grund: "
    f"Reihenfolge, Vorzeichen und Skala der Quellen ändern das Modell nicht: $x = A s = (A D^{{-1}} P^\\top)(P D s)$ mit beliebiger Permutation $P$ und Diagonalmatrix $D$ "
    f"(auch mit negativen Einträgen). Nachgerechnet: die Elektrodensignale mit umskalierten Quellen und Mischung weichen um höchstens {rescaling_invariance(ds):.1e} ab. "
    "Die Skala legt FastICA willkürlich auf Varianz 1 fest; die wahre Amplitude steckt in der Mischung."
)

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden - und wer danach kommt")
if st.button("Gauß'sche und sinusförmige Hintergrundquellen vergleichen", key="gauss_start"):
    st.session_state["gauss_on"] = True
if st.session_state.get("gauss_on"):
    with st.spinner("Rechne 5 Konfigurationen × 5 Datensätze..."):
        grows = _gauss_limit(data_params[:7])
    st.plotly_chart(build_gauss_limit(grows), width="stretch", key="gauss_limit_chart")
    st.table({
        "Hintergrund": ["keiner" if r["g"] == 0 else f"{r['g']} × {'Gauß' if r['kind'] == 'gauss' else 'Rhythmus'}" for r in grows],
        "Korrelation der Neuronen": [f"{r['neuron_corr']:.3f}" for r in grows],
        "SIR der Hintergrundquellen (Mittel)": ["-" if r["g"] == 0 else f"{r['sir']:.1f} dB" for r in grows],
        "Spanne über die Datensätze": ["-" if r["g"] == 0 else f"{r['sir_min']:.1f} bis {r['sir_max']:.1f} dB" for r in grows],
    })
    st.caption("Mit den Rauschen-, Verzögerungs- und Längen-Einstellungen der Aufnahme (Neuronen und Elektroden wie in der Seitenleiste). Zwei Gauß'sche Quellen liegen im Mittel klar unter zwei Rhythmen und streuen von Datensatz zu Datensatz, "
               "weil ihre Drehung nicht durch die Verteilung bestimmt ist, sondern nur durch kleine Stichprobenzufälligkeiten.")

st.markdown(
    """
| Annahme der ICA | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **nicht Gauß'sch** | Zwei Gauß'sche Quellen sind beliebig drehbar - nicht trennbar. Eine einzelne bleibt trennbar. | **SOBI**: nutzt die zeitliche Struktur (verschiedene Autokorrelation) statt der Verteilungsform |
| **mindestens so viele Sensoren wie Quellen** | Die Mischung ist nicht umkehrbar (Abschnitte 🎯 und 📐). | **Sparse Component Analysis**: mehr Quellen als Sensoren, wenn die Quellen selten gleichzeitig aktiv sind |
| **momentane Mischung** | Laufzeitunterschiede zwischen den Elektroden verschlechtern die Trennung (Abschnitte 🎯 und 📐). | **Spike-Sorting-Zweig** inkl. Verzögerungsgraph und Vorlagenabgleich |
| **Vorzeichen frei** | Bei Quellen, die nur positiv sein können (Leistung, Verbrauch), ist das Vorzeichen nicht beliebig. | **NMF**: Nichtnegativität statt Unabhängigkeit |
"""
)
st.caption("Die genannten Verfahren sind die nächsten Stücke der Quellentrennung-Linie; hier steht nur, welche Annahme sie jeweils lockern.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** $x(t) = A\,s(t) + \varepsilon(t)$ mit $x \in \mathbb{R}^n$ (Elektroden), $s \in \mathbb{R}^k$ (Quellen, statistisch unabhängig, höchstens eine Gauß'sch), $A \in \mathbb{R}^{n \times k}$, $n \ge k$.
Gesucht $W$ mit $\hat s = W x$. Identifizierbar nur bis auf Permutation und Skalierung (Comon, 1994).

**Weißen.** Mit $C = \tfrac1T \tilde X \tilde X^\top = V \Lambda V^\top$ ($\tilde X$ zentriert) ist $K = \Lambda_k^{-1/2} V_k^\top$ und $Z = K \tilde X$ hat die Kovarianz $I$. Danach ist $Z = R\,s$ mit orthogonalem $R$ - übrig
bleibt die Suche nach der Drehung. (Weißen = Hauptkomponenten, skaliert - die einzige Verbindung zur Dimensionsreduktion-Linie.)

**Kontrast.** Für eine Projektion $y = w^\top z$, $\lVert w \rVert = 1$: $J(y) = \left(E[G(y)] - E[G(\nu)]\right)^2$ mit $\nu \sim \mathcal N(0,1)$ und $G$ = $\log\cosh y$, $-e^{-y^2/2}$ oder $y^4/4$ (Kurtosis).
$g = G'$ ist $\tanh y$, $y\,e^{-y^2/2}$ bzw. $y^3$.

**FastICA (symmetrisch).** $W \leftarrow E[g(WZ)\,Z^\top] - \operatorname{diag}(E[g'(WZ)])\,W$, dann $W \leftarrow (WW^\top)^{-1/2} W$ bis $\max_i \big|\,|w_i^{\text{neu}\top} w_i| - 1\big| < 10^{-6}$. **Deflation:** eine Zeile nach der anderen,
jeweils orthogonal zu den bisherigen. Die Schätzquellen sind $\hat s = W K \tilde X$.

**Kennzahlen.** Zuordnung Quelle - Komponente: exakt (Bitmasken-DP) über die Summe der $|\rho|$; Vorzeichen und Skala per Regression. SIR $= 10\log_{10}\frac{\rho^2}{1-\rho^2}$. **Amari-Index** von $P = W K A$:
$\frac{1}{2d(d-1)}\Big[\sum_i \big(\tfrac{\sum_j |p_{ij}|}{\max_j |p_{ij}|} - 1\big) + \sum_j \big(\tfrac{\sum_i |p_{ij}|}{\max_i |p_{ij}|} - 1\big)\Big]$, $d = \max(k, \text{Zeilen})$.
**Spitzen-F1**: Schwelle $\max(4\sigma_{\text{MAD}},\, 0.3 \times$ Median der 10 tiefsten Spitzen$)$ auf der zugeordneten, im Vorzeichen der Quelle ausgerichteten Spur; Treffer = innerhalb von ±4 Abtastwerten.

**Grenzen.** (1) *Gauß'sche Quellen*: die Verteilung ändert sich unter Drehungen nicht, $J$ ist konstant - zwei solche Quellen sind nicht trennbar. (2) *Weniger Sensoren als Quellen*: $A$ ist nicht umkehrbar.
(3) *Gefaltete Mischung*: $x(t) = \sum_\tau A_\tau s(t - \tau)$ - eine einzige Matrix $W$ genügt nicht. (4) *Rauschen* wird mit entmischt und verstärkt. (5) Die **Quellenzahl** wird als bekannt angenommen.
(6) FastICA nutzt nur die Verteilung einzelner Zeitpunkte, nicht deren zeitliche Reihenfolge.

Implementiert in `ica_algorithm.py` (Weißen, FastICA, Kontrastfunktionen), `ica_scenario.py` (Mehrelektroden-Generator), `ica_evaluation.py` (Zuordnung, Kennzahlen, Spitzenerkennung, Sweeps, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Quellentrennung: von ICA bis Verzögerungsgraph](https://sebastianhanisch.net/konzepte-quellentrennung.html)."
)
