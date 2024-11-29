# Datenanalyse-Dashboard
Durchführung statistischer Analysen auf Datensätzen

# Datenanalyse Dashboard

Dieses Python-Projekt bietet ein interaktives Dashboard zur Durchführung verschiedener statistischer Analysen und Datenvisualisierungen. Es nutzt Streamlit als Webframework und Plotly für die Visualisierung der Ergebnisse. Mit diesem Dashboard kannst du deine Datensätze analysieren und visualisieren, einschließlich Korrelationsanalysen, Hypothesentests, K-Means Clustering, Zeitreihenanalysen und vieles mehr.

## Funktionen

Das Dashboard bietet folgende Funktionen:

1. **Korrelationsanalyse**: Berechnet und visualisiert die Korrelationsmatrix zwischen den numerischen Spalten deines Datensatzes.
2. **Verteilungsanalyse**: Visualisiert die Verteilung einer ausgewählten Spalte und führt den Shapiro-Wilk-Test zur Überprüfung der Normalverteilung durch.
3. **Hypothesentest (t-Test)**: Führt einen t-Test durch, um zu testen, ob der Mittelwert einer Spalte signifikant von einem Referenzwert abweicht.
4. **Zusätzliche Statistiken**: Berechnet zusätzliche statistische Kennzahlen wie Median, Modus, Interquartilsabstand (IQR), Schiefe und Kurtosis für eine ausgewählte Spalte.
5. **Ausreißererkennung**: Identifiziert und visualisiert Ausreißer in einer numerischen Spalte basierend auf Z-Scores.
6. **Zeitreihenanalyse**: Zerlegt eine Zeitreihe in ihre Trend-, Saison- und Residuenkomponenten und visualisiert diese.
7. **K-Means Clustering**: Führt ein K-Means Clustering durch und visualisiert die Cluster auf einem Scatterplot.
8. **Feature Importance (Wichtigkeitsanalyse von Features)**: Berechnet und visualisiert die Wichtigkeit der Features eines Modells zur Vorhersage eines Zielwerts.
9. **t-SNE Visualisierung**: Visualisiert hochdimensionale Daten auf einer 2D-Fläche mithilfe der t-SNE-Technik.
10. **Bootstrap-Konfidenzintervalle**: Berechnet und visualisiert das 95%-Konfidenzintervall für eine Spalte mithilfe von Bootstrap.
11. **Überlebensanalyse**: Erzeugt eine Kaplan-Meier-Kurve, um die Überlebenswahrscheinlichkeit in einer überlebensbasierten Datenanalyse zu visualisieren.

## Installation

### Voraussetzungen

Stelle sicher, dass du Python 3.7 oder höher installiert hast. Es wird empfohlen, eine virtuelle Umgebung zu verwenden.

**Erstelle eine virtuelle Umgebung (optional, aber empfohlen):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Auf macOS/Linux
   .\venv\Scripts\activate   # Auf Windows

