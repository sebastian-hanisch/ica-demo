# Unabhängige Komponentenanalyse (ICA) an Mehrelektroden-Signalen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-ica-demo.streamlit.app/)**

Erstes Stück (**Wurzel**) der **Quellentrennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **FastICA** – an einem wachsenden Beispiel.
Vehikel: ein **Mehrelektroden-Array**. Neuronen (spitzenartige Quellen mit fester Lage, Spitzenform und Feuerrate) und optionale Hintergrundquellen werden von einer Zeile aus 1–8 Elektroden **gemischt**;
gesucht sind die Quellen, ohne das Array oder die Neuronen zu kennen. Das ist der Vorgriff auf den Spike-Sorting-Zweig der Linie.

**Einordnung in die Reihe (die Kanten des Graphen):** ICA ist die Wurzel der Quellentrennung-Linie. Die einzige Kante zurück in die Dimensionsreduktion ist die **PCA-Weißung** (Schritt 3 der Demo) – ICA gehört bewusst nicht zur
Dimensionsreduktion-Linie: sie ist linear und behebt eine andere PCA-Annahme ("unkorreliert genügt"), und sie trennt Quellen statt Dimensionen zu verringern.
```
ica-demo (Wurzel) → SOBI        (zeitliche Struktur statt Nicht-Gaußianität)
                  → NMF         (Nichtnegativität statt Unabhängigkeit)
                  → Sparse Component Analysis   (mehr Quellen als Sensoren)
                  → Spike-Sorting-Zweig         (Standard-Pipeline, Vorlagenabgleich, Verzögerungsgraph)
```
Die Nachfolger sind noch nicht gebaut; die Demo markiert nur, welche ICA-Annahme sie jeweils lockern.

| Frage | Ergebnis (4 Neuronen, 6 Elektroden, wenn nicht anders angegeben; Mittel über 5 feste Datensätze, Seeds 100000–100004) |
|---|---|
| ICA gegen PCA und Einzelelektrode | ✅ Korrelation der Neuronen mit ihrer Schätzung **0.98** gegen PCA 0.65 und beste Einzelelektrode 0.73; Spitzen-F1 **1.00** gegen 0.65 (PCA) und 0.61 (Elektrode) |
| Unkorreliert ≠ unabhängig | ✅ die PCA-Komponenten sind Mischungen und damit Gauß'scher: mittlere Kurtosis unter der Hälfte der Quellen-Kurtosis; die ICA-Komponenten erreichen über 75 % davon |
| Rauschen | ⚠️ Korrelation 0.98 (Rauschen 0.05) → 0.94 (0.1) → 0.84 (0.2) → 0.69 (0.4) → 0.51 (0.7) → 0.43 (1.0); ab etwa 0.5 ist die **beste Einzelelektrode besser als ICA** (0.7: 0.59 gegen 0.51) – Entmischen verstärkt das Rauschen |
| Laufzeitverzögerung | ❌ Korrelation 0.98 → 0.97 (1) → 0.92 (2) → 0.84 (4) → 0.76 (8) → 0.71 (12); Spitzen-F1 1.00 → 0.96 → 0.85 → 0.84 → 0.65 → 0.66 (Verzögerung in Abtastwerten je Einheitsabstand, 1 Abtastwert = 0.1 ms) |
| Zu wenige Elektroden (4 Neuronen) | ❌ 0.16 (1) → 0.41 (2) → 0.68 (3) → **0.98 (4)** → 0.98 (5–8); Spitzen-F1 0.16 / 0.36 / 0.69 / 1.00 |
| Gauß'sche Hintergrundquellen | ⚠️ eine Gauß'sche Quelle ist trennbar (SIR 7.3 dB, wie ein Rhythmus 7.6 dB), **zwei nicht sicher**: SIR im Mittel 3.7 dB, Spanne 0.1 bis 7.3 dB je Datensatz; das Rhythmus-Paar erreicht in jedem Datensatz 7.9–8.1 dB. Die Neuronen bleiben trennbar (0.92) |
| Länge der Aufnahme | ✅ kaum relevant: Korrelation 0.98 schon bei 1000 Abtastwerten (0.1 s), 0.965 bei 500 – bei diesem Rauschen begrenzt das Rauschen, nicht die Datenmenge |
| Kontrastfunktion | ⚠️ log cosh und Kurtosis: bei allen 5 Starten dasselbe (Streuung < 1e-6), Kurtosis am schnellsten (6 Iterationen gegen 13); Gauß-Ableitung: einzelne Starts landen schlechter (schlechtester 0.84, Streuung 0.027), 22 Iterationen |
| Rechenzeit | ✅ 0.08 s (T = 20000, 4 Neuronen), 0.17 s (T = 40000, 5 Neuronen, 8 Elektroden) |

## Was die Demo zeigt

1. **ICA in Aktion** (Schritt-Slider + Abspielen, Zeitfenster-Regler): **Quellen** (wahre Neuronen, Ort von Neuronen und Elektroden) → **Mischung** (Elektrodensignale, mittlere Spitzenform je Elektrode: leiser mit dem Abstand, bei Verzögerung später) →
   **Weißen** (dieselbe Idee in zwei Dimensionen: Quellen = Kreuz, Mischung = schief, Weißen = unkorreliert aber gedreht, ICA = Kreuz wieder gerade) → **Rotation suchen** (Nicht-Gaußianität über den Drehwinkel, Iterationsverlauf von FastICA) → **Ergebnis** (wahre Quellen gegen Schätzungen).
2. **Was die ICA gefunden hat – gegen PCA und die beste Einzelelektrode:** Korrelation, Spitzen-F1, Signal-zu-Interferenz, Amari-Index; die vier Spurgruppen (Elektroden, wahre Quellen, PCA, ICA); Zuordnungsmatrizen |Korrelation| Quelle × Komponente; Urteil (Codes: zu wenige Elektroden → nicht konvergiert → zwei Gauß-Quellen → Verzögerung → Rauschen → ICA vorn → neutral).
3. **📊 Nicht-Gaußianität und Kontrastfunktion:** Exzess-Kurtosis der Quellen, PCA- und ICA-Komponenten; Experiment auf Abruf: 3 Kontrastfunktionen × 2 Verfahren (symmetrisch/Deflation) × 5 Starts × 5 Datensätze.
4. **📐 Sweeps** über Rauschen, Verzögerung, Elektrodenzahl und Länge (feste Datensätze ab 100000, mit Streuung, aktueller Wert markiert, Spitzen-F1 gestrichelt).
5. **🔀 Was ICA nicht festlegen kann:** Reihenfolge, Vorzeichen, Skala – vier Startwerte auf denselben Daten, dieselbe Quelle in verschiedenen Komponenten und mit verschiedenem Vorzeichen; die Skalierungs-Invarianz ist nachgerechnet.
6. **🚧 Grenzen:** Experiment auf Abruf für Gauß'sche und sinusförmige Hintergrundquellen; Tabelle "welche Annahme, was passiert, wer setzt an" (SOBI, Sparse Component Analysis, Spike-Sorting-Zweig, NMF).

Regler: Neuronen (2–5), Elektroden (1–8), Hintergrundquellen (0–2) mit Art (Gauß-Rauschen oder Rhythmus; bei 0 ausgeblendet, Auswahl bleibt erhalten), Rauschen, Laufzeitverzögerung, Länge der Aufnahme, Kontrastfunktion, Start der Entmischung.

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | ICA-Korrelation | PCA | beste Elektrode | Spitzen-F1 (ICA) | Urteil |
|---|---|---|---|---|---|
| Vier Neuronen, sechs Elektroden | 0.98 | 0.66 | 0.73 | 1.00 | ICA vorn |
| Zwei Rhythmus-Quellen | 0.92 | 0.57 | 0.57 | 1.00 | ICA vorn; SIR des Rhythmus-Paars ≈ 8 dB |
| Zwei Gauß-Quellen | 0.92 | 0.57 | 0.56 | 1.00 | Gauß-Paar nicht trennbar |
| Zu wenige Elektroden (3) | 0.68 | 0.55 | 0.58 | 0.67–0.70 | zu wenige Elektroden |
| Laufzeitverzögerung (8) | 0.76 | 0.59 | 0.73 | 0.56–0.77 | Verzögerung |
| Starkes Rauschen (0.7) | 0.49–0.55 | 0.41 | 0.59 | 0.52–0.67 | Rauschen (Einzelelektrode besser) |

Kurtosis der Neuronen im Standard-Seed: 99 / 35 / 41 / 43 (Gauß = 0), Sinusrhythmen −1.5.

## Modell und Verfahren

- **Szenario** (`ica_scenario.py`): 10 kHz; Neuronen mit biphasischer Wellenform (30 Abtastwerte, negative Spitze), Poisson-artigem Feuern (20–35 Hz) mit 2 ms Refraktärzeit; Hintergrund: AR(1)-Gauß-Rauschen (φ = 0.95 bzw. 0.5) oder Sinusrhythmus (10 bzw. 23 Hz).
  Mischung: Elektroden auf einer Zeile, Neuronen an festen Orten, Gewicht ∝ 1/(d² + ε); Hintergrund flächig (gleich stark bzw. Rampe). Verzögerung = round(δ · Abstand zur nächsten Elektrode) Abtastwerte, nur für Neuronen.
  Rauschen relativ zum Neuronen-Signal (unabhängig von der Zahl der Hintergrundquellen). Jede Quelle hat einen eigenen Zufallsstrom (Seed, Nummer): Neuron 1 feuert für einen Seed immer gleich, egal wie viele Neuronen oder Elektroden eingestellt sind.
- **FastICA** (`ica_algorithm.py`, numpy von Grund auf, Hyvärinen 1999): Weißen per Eigenzerlegung (Kovarianz exakt I), Fixpunkt-Iteration `w ← E[z g(wᵀz)] − E[g'(wᵀz)] w`, **symmetrisch** (W(WᵀW)^-½) oder **Deflation**; Kontrastfunktionen log cosh, Gauß-Ableitung, Kurtosis; die Quellenzahl wird als bekannt vorgegeben.
  Gegen `sklearn.decomposition.FastICA` getestet (Zuordnungs-Korrelation > 0.99).
- **Auswertung** (`ica_evaluation.py`): exakte Zuordnung Quelle ↔ Komponente per Bitmasken-DP, Vorzeichen und Skala per Regression; |Korrelation|, SIR = 10·log10(ρ²/(1−ρ²)), Amari-Index von W·K·A (auf [0, 1] normiert, für nicht quadratische Matrizen verallgemeinert),
  Spitzen-F1 (Schwelle max(4 σ_MAD, 0.3 × Median der 10 tiefsten Spitzen), Toleranz ±4 Abtastwerte); Referenzläufe ohne Rauschen und ohne Verzögerung zur Trennung beider Effekte im Urteil; Sweeps; Kontrast-Tabelle; Gauß-Grenze; Mehrdeutigkeits-Demo.

## Was nicht funktioniert hat / Grenzen

- **Die Gauß-Grenze ist in endlichen Stichproben unscharf:** bei rauschfreier Aufnahme und AR(1)-Gauß-Rauschen trennt die ICA das Paar teils trotzdem (SIR bis 29 dB), weil eine endliche Stichprobe nie exakt Gauß'sch ist. Deshalb zeigt die Demo die Spanne über
  die Datensätze und nicht einen Einzelwert; bei den Standard-Einstellungen (Rauschen 0.05) liegt das Gauß-Paar im Mittel 4 dB unter dem Rhythmus-Paar und streut von 0.1 bis 7.3 dB (Rhythmus: 7.9–8.1). Meine erste Annahme, "chance level = Korrelation 0.64", war falsch (nach optimaler Zuordnung ist es 0.90).
- **Die Spitzenerkennung hatte einen Fehler:** eine reine MAD-Schwelle liegt bei einer rauschfreien Aufnahme bei 0 und meldet jeden Rest (F1 0.65 bei perfekter Trennung); die Mindesttiefe (30 % der typischen Spitzentiefe) löst das. Als Test hinterlegt.
- **Vorzeichen der ICA-Ausgabe sind oft alle gleich:** bei symmetrischer FastICA und spitzenartigen Quellen wechselt jede Iteration das Vorzeichen aller Komponenten gemeinsam; das Vorzeichen hängt dann nur an der Parität der Iterationszahl. Die Mehrdeutigkeit ist real, zeigt sich aber erst über mehrere Starts (Start 3 und 4 im Standard-Seed).
- **Das Verfahren "symmetrisch/Deflation" ist kein Regler:** beide liefern bei log cosh und Kurtosis dasselbe (Korrelation 0.980 gegen 0.980); der Unterschied steht nur in der Kontrast-Tabelle (Iterationen, Startabhängigkeit bei der Gauß-Ableitung).
- **Die Länge der Aufnahme wirkt kaum:** 500 Abtastwerte reichen für 0.965. Der Regler bleibt (die Demo soll das zeigen), der Sweep ist bewusst flach.
- **Synthetische Daten:** jede Spitze eines Neurons hat exakt dieselbe Form (keine Amplitudenschwankung), die Mischung ist bis auf die Verzögerung exakt linear, das Rauschen ist weiß und Gauß'sch, Elektroden liegen auf einer Zeile. Echte Ableitungen sind schwieriger; überlappende Spitzen (Kollisionen) wurden nicht gesondert gemessen.
- **Die Quellenzahl wird als bekannt angenommen.** Bei unbekannter Zahl (Modellwahl) macht die Demo keine Aussage.
- **Amari-Index bei Verzögerung** ist gegen die Mischung ohne Verzögerung gerechnet – es gibt bei gefalteter Mischung keine einzelne wahre Matrix.

## Verifikation

- Weißen: Kovarianz = I, Mittel 0, Eigenwerte absteigend, Hauptrichtungen bleiben beim Kürzen erhalten. Kontrastfunktionen: g = G′ und g′ = (g)′ per finite Differenzen; Gauß-Referenzwerte (0.75 / −1/√2 / 0.3746); Nicht-Gaußianität Gauß < Laplace.
- FastICA: W orthogonal, Ausgaben unkorreliert, rauschfreie Mischung wird für vier Kombinationen (Kontrast × Verfahren) mit Korrelation > 0.995 getrennt, deterministisch bei festem Start, **Kreuzprüfung gegen scikit-learn**.
- Zuordnung gegen Brute-Force (exakt, auch bei ungleicher Zahl); Permutation × Vorzeichen × Skala wird vollständig zurückgenommen; Amari-Index und SIR per Handinstanzen; Spitzenerkennung mit geplanten Spitzen.
- Szenario: Varianz 1, Neuronen-Kurtosis > 25, Rhythmus −1.5, X = A·S ohne Verzögerung, Verzögerung verschiebt ferne Elektroden, Ströme unabhängig von den anderen Einstellungen, Spitzenzeiten zeigen auf das Minimum.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Rauschen, Verzögerung, Elektrodenzahl, Länge, Kurtosis-Bereiche, Kontrast-Tabelle, Gauß-Tabelle, Presets; jeweils Mittel über die festen Sweep-Datensätze mit weiten Toleranzen); Verdict-Codes über mehrere Datensätze; alle 6 Presets in Bändern;
  AppTest-Rauchtests (Default, jedes Preset, jeder Schritt, Randgrößen, Art des Hintergrunds ausgeblendet und wiederhergestellt, Fenster-Klemmung, Sweeps und Experimente auf Abruf), Achsensperre und explizite Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📊 Kontrast, 📐 Sweeps, 🔀 Mehrdeutigkeiten, 🚧 Grenzen, Mathe |
| `ica_algorithm.py` | Weißen, FastICA (symmetrisch/Deflation), Kontrastfunktionen |
| `ica_scenario.py`, `ica_constants.py` | Mehrelektroden-Generator, Konstanten, Presets |
| `ica_evaluation.py` | Zuordnung, Kennzahlen, Spitzenerkennung, Sweeps, Urteil, Zwei-Quellen-Anschauung |
| `ica_presets.py`, `ica_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Algorithmus, sklearn-Kreuzvergleich, Szenario, Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
