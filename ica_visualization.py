"""Plotly-Visualisierungen der ICA-Demo: Elektrodenlayout, Signalspuren, Zwei-Quellen-Streudiagramme, Kontrastkurve, Iterationsverlauf, Zuordnungsmatrix, Kennzahlen-Balken,
Kurtosis, Sweeps, Spitzenform je Elektrode, Gauß-Grenze und Mehrdeutigkeit. Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios: keine Zoom-/Pan-Gesten im Chart)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ica_constants as C
from ica_scenario import electrode_positions

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf"
NEURON_COLORS = ("#1f77b4", "#d68a2e", "#2ca02c", "#8e5fbf", "#c2185b")
BACKGROUND_COLORS = ("#7f7f7f", "#a0a0a0")
METHOD_COLORS = {"electrodes": GRAY, "pca": ORANGE, "ica": BLUE}
METHOD_NAMES = {"electrodes": "beste Einzelelektrode", "pca": "PCA (nur weißen)", "ica": "ICA"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def source_color(i):
    return NEURON_COLORS[i] if i < len(NEURON_COLORS) else BACKGROUND_COLORS[(i - len(NEURON_COLORS)) % 2]


def source_labels(ds):
    labels = [f"Neuron {i + 1}" for i in range(ds.n_neurons)]
    for j, kind in enumerate(ds.kinds[ds.n_neurons:]):
        labels.append(f"{'Gauß-Hintergrund' if kind == 'gauss' else 'Rhythmus'} {j + 1}")
    return labels


def _thin(points, keep=6000, seed=0):
    """Streudiagramm-Ausdünnung: alle Punkte mit großem Radius (die Spitzen) plus eine Stichprobe der übrigen."""
    n = points.shape[1]
    if n <= keep:
        return np.arange(n)
    radius = np.hypot(points[0], points[1])
    far = np.flatnonzero(radius > np.quantile(radius, 0.9))
    near = np.flatnonzero(radius <= np.quantile(radius, 0.9))
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([far, rng.choice(near, max(keep - len(far), 0), replace=False)]))


def build_layout(ds):
    """Elektroden (Quadrate auf y = 0) und Neuronen (Kreise, Größe = Spitzenamplitude); Hintergrund wirkt flächig auf alle Elektroden."""
    pos = electrode_positions(ds.n_electrodes)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=[f"E{j + 1}" for j in range(ds.n_electrodes)], textposition="bottom center", name="Elektroden",
                             marker=dict(symbol="square", size=14, color=GRAY), hoverinfo="skip"))
    for i in range(ds.n_neurons):
        x, y = C.NEURON_POSITIONS[i]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"N{i + 1}"], textposition="top center", name=f"Neuron {i + 1}", hoverinfo="skip",
                                 marker=dict(size=10 + 14 * C.NEURON_AMPLITUDES[i], color=source_color(i), opacity=0.85)))
        if ds.delays.any():
            j = int(np.argmin(np.linalg.norm(pos - np.array([x, y]), axis=1)))
            fig.add_annotation(x=pos[j, 0], y=pos[j, 1], ax=x, ay=y, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor=source_color(i), opacity=0.4)
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[-0.1, 1.1], title="Ort (willkürliche Einheit)", zeroline=False),
                      yaxis=dict(range=[-0.15, 0.7], title="Abstand", zeroline=False), showlegend=False)
    return lock_axes(fig)


def build_traces(labels, arrays, t0, width, colors=None, spike_times=None, height=None, normalise=True):
    """Gestapelte Spuren eines Zeitfensters [t0, t0 + width) in ms (Abtastrate 10 kHz); optional Markierungen der wahren Spitzen je Zeile."""
    fs = C.SAMPLE_RATE
    lo, hi = int(t0 * fs / 1000), int((t0 + width) * fs / 1000)
    fig = go.Figure()
    n = len(arrays)
    scale_all = max(float(np.abs(a).max()) for a in arrays) if not normalise else None
    for r, (label, y) in enumerate(zip(labels, arrays)):
        seg = y[lo:hi]
        scale = float(np.abs(y).max()) if normalise else scale_all
        offset = (n - 1 - r) * 1.3
        color = colors[r] if colors else BLUE
        fig.add_trace(go.Scatter(x=np.arange(lo, hi) * 1000.0 / fs, y=offset + seg / max(scale, 1e-12), mode="lines", line=dict(color=color, width=1.2), name=label, hoverinfo="skip"))
        if spike_times is not None and r < len(spike_times):
            marks = [t for t in spike_times[r] if lo <= t < hi]
            if marks:
                fig.add_trace(go.Scatter(x=np.array(marks) * 1000.0 / fs, y=[offset + 0.75] * len(marks), mode="markers", marker=dict(symbol="triangle-down", size=7, color=color), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=height or max(180, 42 * n + 60), margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis=dict(title="Zeit [ms]"), yaxis=dict(tickmode="array", tickvals=[(n - 1 - r) * 1.3 for r in range(n)], ticktext=list(labels), zeroline=False))
    return lock_axes(fig)


def build_pair_scatter(pair, names=("Quellen (unabhängig)", "Elektroden (gemischt)", "nach dem Weißen (PCA)", "nach der ICA-Rotation")):
    """Vier Blicke auf dieselben Zwei-Quellen-Daten: die unabhängigen Quellen (Kreuz), die Mischung (verzerrt), das Weißen (unkorreliert, aber gedreht) und die ICA-Rotation (Kreuz wieder gerade).
    Achsen auf das 99.8-%-Quantil beschnitten, damit die Arme des Kreuzes sichtbar bleiben (Spitzen sind selten und groß)."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=names, horizontal_spacing=0.08, vertical_spacing=0.12)
    limits = []
    for panel, data in enumerate((pair.S, pair.X, pair.Z, pair.Y), start=1):
        r, c = (panel - 1) // 2 + 1, (panel - 1) % 2 + 1
        idx = _thin(data)
        fig.add_trace(go.Scattergl(x=data[0][idx], y=data[1][idx], mode="markers", marker=dict(size=3, color=BLUE, opacity=0.4), hoverinfo="skip"), row=r, col=c)
        lim = float(np.quantile(np.abs(data), 0.998)) * 1.15
        limits.append(lim)
        fig.update_xaxes(range=[-lim, lim], scaleanchor=f"y{panel if panel > 1 else ''}", row=r, col=c, zeroline=True)
        fig.update_yaxes(range=[-lim, lim], row=r, col=c, zeroline=True)
    lim = limits[2]
    for angles, color, width, opacity in ((pair.true_angles, GREEN, 6, 0.55), (pair.ica_angles, RED, 2, 1.0)):
        for ang in angles:
            fig.add_trace(go.Scatter(x=[-lim * np.cos(ang), lim * np.cos(ang)], y=[-lim * np.sin(ang), lim * np.sin(ang)], mode="lines", line=dict(color=color, width=width), opacity=opacity,
                                     showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return lock_axes(fig)


def build_contrast_curve(pair, contrast_label):
    """Nicht-Gaußianität der Projektion cos(a)·z1 + sin(a)·z2 über den Winkel a: Maxima = die unabhängigen Richtungen."""
    fig = go.Figure(go.Scatter(x=np.degrees(pair.angles), y=pair.contrast, mode="lines", line=dict(color=BLUE, width=2), name="Nicht-Gaußianität", hoverinfo="skip"))
    for ang in pair.true_angles:
        fig.add_vline(x=float(np.degrees(ang)), line=dict(color=GREEN, dash="dash", width=2))
    for ang in pair.ica_angles:
        fig.add_vline(x=float(np.degrees(ang)), line=dict(color=RED, width=1.5))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Drehwinkel der Projektionsrichtung [°]", range=[0, 180]),
                      yaxis=dict(title=f"Nicht-Gaußianität ({contrast_label})", rangemode="tozero"), showlegend=False)
    return lock_axes(fig)


def build_iterations(corr_curve, converged_at):
    """Mittlere Korrelation der Neuronen mit ihren Schätzungen nach jeder Fixpunkt-Iteration (0 = zufälliger Start)."""
    fig = go.Figure(go.Scatter(x=np.arange(len(corr_curve)), y=corr_curve, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=6), hoverinfo="skip"))
    if converged_at is not None:
        fig.add_vline(x=converged_at, line=dict(color=GREEN, dash="dash"), annotation_text="konvergiert", annotation_position="bottom right")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Iteration"), yaxis=dict(title="mittlere |Korrelation| der Neuronen", range=[0, 1.02]), showlegend=False)
    return lock_axes(fig)


def build_corr_heatmap(cm, row_labels, col_labels, title):
    """|Korrelation| wahre Quelle (Zeile) gegen Schätzung (Spalte): ein sauberes Bild hat in jeder Zeile und Spalte genau einen hellen Eintrag."""
    fig = go.Figure(go.Heatmap(z=cm, x=col_labels, y=row_labels, zmin=0, zmax=1, colorscale="Blues", text=np.round(cm, 2), texttemplate="%{text}", showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=60 + 40 * len(row_labels), margin=dict(l=10, r=10, t=40, b=10))
    return lock_axes(fig)


def build_method_bars(methods):
    """Mittlere |Korrelation| der Neuronen und Spitzen-F1 für beste Einzelelektrode, PCA und ICA."""
    fig = go.Figure()
    for name in ("electrodes", "pca", "ica"):
        m = methods[name]
        fig.add_trace(go.Bar(x=["Korrelation der Neuronen", "Spitzen-F1"], y=[m.neuron_corr, m.f1], name=METHOD_NAMES[name], marker_color=METHOD_COLORS[name],
                             text=[f"{m.neuron_corr:.2f}", f"{m.f1:.2f}"], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1.12]), legend=dict(orientation="h", y=-0.15))
    return lock_axes(fig)


def build_kurtosis(labels, groups):
    """Exzess-Kurtosis (0 = Gauß'sch) je Quelle: wahre Quellen, PCA-Komponenten, ICA-Komponenten (in der Reihenfolge der Quellen)."""
    fig = go.Figure()
    for name, values, color in groups:
        fig.add_trace(go.Bar(x=labels, y=values, name=name, marker_color=color, hoverinfo="skip"))
    fig.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Exzess-Kurtosis (0 = Gauß)"), legend=dict(orientation="h", y=-0.2))
    return lock_axes(fig)


def build_sweep(rows, xlabel, current=None, log=False, show_f1=True):
    fig = go.Figure()
    xs = [r["x"] for r in rows]
    for key, name, color in (("ica", "ICA", BLUE), ("pca", "PCA", ORANGE), ("electrodes", "beste Einzelelektrode", GRAY)):
        y = np.array([r[key] for r in rows])
        sd = np.array([r[key + "_std"] for r in rows])
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(y + sd) + list(y - sd)[::-1], fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=y, mode="lines+markers", name=name + " (Korrelation)", line=dict(color=color, width=2), hoverinfo="skip"))
    if show_f1:
        fig.add_trace(go.Scatter(x=xs, y=[r["f1"] for r in rows], mode="lines+markers", name="ICA: Spitzen-F1", line=dict(color=GREEN, width=2, dash="dot"), hoverinfo="skip"))
    if current is not None:
        fig.add_vline(x=current, line=dict(color=RED, dash="dash"), annotation_text="aktuell", annotation_position="top")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis=dict(title=xlabel, type="log" if log else "linear"), yaxis=dict(title="Mittel über die Sweep-Datensätze", range=[0, 1.05]),
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_footprint(t_axis, waveforms, color):
    """Mittlere Spitzenform eines Neurons an jeder Elektrode: Amplitude fällt mit dem Abstand, bei Verzögerung verschiebt sich die Spitze."""
    fig = go.Figure()
    scale = float(np.abs(waveforms).max())
    for j, w in enumerate(waveforms):
        fig.add_trace(go.Scatter(x=t_axis, y=w / scale + (len(waveforms) - 1 - j) * 1.2, mode="lines", line=dict(color=color, width=1.8), name=f"E{j + 1}", hoverinfo="skip"))
    fig.update_layout(height=max(200, 34 * len(waveforms) + 80), margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis=dict(title="Zeit relativ zur Spitze [ms]"),
                      yaxis=dict(tickmode="array", tickvals=[(len(waveforms) - 1 - j) * 1.2 for j in range(len(waveforms))], ticktext=[f"E{j + 1}" for j in range(len(waveforms))], zeroline=False))
    return lock_axes(fig)


def build_gauss_limit(rows):
    """Trennung der Hintergrundquellen (SIR in dB, Spanne über die Sweep-Datensätze) je Art und Anzahl."""
    rows = [r for r in rows if r["g"] > 0]
    labels = [f"{r['g']} × {'Gauß' if r['kind'] == 'gauss' else 'Rhythmus'}" for r in rows]
    fig = go.Figure(go.Bar(x=labels, y=[r["sir"] for r in rows], marker_color=[RED if (r["kind"] == "gauss" and r["g"] >= 2) else BLUE for r in rows],
                           error_y=dict(type="data", symmetric=False, array=[r["sir_max"] - r["sir"] for r in rows], arrayminus=[r["sir"] - r["sir_min"] for r in rows]),
                           text=[f"{r['sir']:.1f}" for r in rows], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="SIR der Hintergrundquellen [dB]"), showlegend=False)
    return lock_axes(fig)


def build_ambiguity(truth, runs, t0, width, neuron):
    """Ein Neuron: wahre Quelle und die rohen Schätzungen dreier Starts (Vorzeichen und Skala beliebig)."""
    fs = C.SAMPLE_RATE
    lo, hi = int(t0 * fs / 1000), int((t0 + width) * fs / 1000)
    x = np.arange(lo, hi) * 1000.0 / fs
    fig = go.Figure()
    rows = [("wahre Quelle", truth, GREEN)] + [(label, y, BLUE) for label, y in runs]
    n = len(rows)
    for r, (label, y, color) in enumerate(rows):
        scale = max(float(np.abs(y).max()), 1e-12)
        fig.add_trace(go.Scatter(x=x, y=(n - 1 - r) * 1.3 + y[lo:hi] / scale, mode="lines", line=dict(color=color, width=1.4), hoverinfo="skip"))
    fig.update_layout(height=80 * n + 60, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis=dict(title="Zeit [ms]"),
                      yaxis=dict(tickmode="array", tickvals=[(n - 1 - r) * 1.3 for r in range(n)], ticktext=[r[0] for r in rows], zeroline=False))
    return lock_axes(fig)
