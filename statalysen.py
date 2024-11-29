import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from lifelines import KaplanMeierFitter
from scipy.stats import (
    shapiro, zscore, ttest_1samp, iqr, skew, kurtosis
)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st


# Funktionen
def upload_csv():
    file_path = st.file_uploader("CSV-Datei hochladen", type=["csv"])  # Hier den Pfad zu deiner CSV-Datei angeben
    return pd.read_csv(file_path)


def summarize_data(data):
    st.write("### Allgemeine Übersicht:")
    st.write(data.info())
    st.write("### Statistische Beschreibung:")
    st.write(data.describe())


def correlation_analysis(data, method='pearson'):
    st.write("### Korrelationsmatrix:")
    corr_matrix = data.corr(method=method)
    st.write(corr_matrix)
    fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="plasma",
                    title=f"Korrelationsmatrix ({method.capitalize()})")
    st.plotly_chart(fig)


def distribution_analysis(data, column):
    fig = px.histogram(data, x=column, nbins=30, marginal="box", title=f"Verteilung: {column}")
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig)

    stat, p_value = shapiro(data[column].dropna())
    st.write(f"Shapiro-Wilk-Test für Normalität: Statistik={stat:.3f}, p-Wert={p_value:.3f}")
    if p_value > 0.05:
        st.write(f"{column} folgt vermutlich einer Normalverteilung.")
    else:
        st.write(f"{column} folgt vermutlich keiner Normalverteilung.")


def detect_outliers(data, column):
    z_scores = zscore(data[column].dropna())
    outliers = data[column][(z_scores > 3) | (z_scores < -3)]
    st.write(f"### Ausreißer in {column} (Z-Score > 3):")
    st.write(outliers)

    fig = px.scatter(data, x=column, title=f"Ausreißer in {column}")
    fig.add_trace(
        go.Scatter(x=outliers.index, y=outliers, mode='markers', marker=dict(color='red', size=8), name='Ausreißer'))
    st.plotly_chart(fig)


def hypothesis_testing(data, column, test_value):
    stat, p_value = ttest_1samp(data[column].dropna(), test_value)
    st.write(f"### t-Test für {column}: Statistik={stat:.3f}, p-Wert={p_value:.3f}")
    if p_value > 0.05:
        st.write(f"Kein signifikanter Unterschied zu {test_value}.")
    else:
        st.write(f"Signifikanter Unterschied zu {test_value}.")


def additional_statistics(data, column):
    med = data[column].median()
    mod = data[column].mode().iloc[0]
    iqr_val = iqr(data[column].dropna())
    skewness = skew(data[column].dropna())
    kurt = kurtosis(data[column].dropna())

    st.write(f"### Zusätzliche Statistiken für {column}:")
    st.write(f"Median: {med}")
    st.write(f"Modus: {mod}")
    st.write(f"Interquartilsabstand (IQR): {iqr_val}")
    st.write(f"Schiefe (Skewness): {skewness}")
    st.write(f"Kurtosis: {kurt}")


def time_series_analysis(data, column, freq):
    ts_data = data[column].dropna()
    decomposition = seasonal_decompose(ts_data, period=freq, model='additive')
    components = {
        'Trend': decomposition.trend,
        'Saisonalität': decomposition.seasonal,
        'Residuum': decomposition.resid
    }
    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=list(components.keys()))
    for i, (comp_name, comp_data) in enumerate(components.items(), 1):
        fig.add_trace(go.Scatter(x=ts_data.index, y=comp_data, mode='lines', name=comp_name), row=i, col=1)
    fig.update_layout(title=f"Zeitreihenanalyse: {column}", height=600)
    st.plotly_chart(fig)


def kmeans_clustering(data, columns, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data[columns].dropna())
    data['Cluster'] = clusters
    st.write("### Cluster-Zuordnung:")
    st.write(data[['Cluster'] + columns])

    fig = px.scatter_matrix(data, dimensions=columns, color='Cluster', title='K-Means Clustering', symbol='Cluster')
    st.plotly_chart(fig)


def feature_importance(data, target, features):
    model = RandomForestRegressor(random_state=42)
    model.fit(data[features], data[target])
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.write("### Feature Importance:")
    st.write(importance)

    fig = px.bar(importance, x='Importance', y='Feature', orientation='h', title='Feature Importance',
                 color='Importance')
    st.plotly_chart(fig)


def tsne_visualization(data, columns, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(data[columns].dropna())
    fig = px.scatter(x=tsne_result[:, 0], y=tsne_result[:, 1], title="t-SNE Visualisierung",
                     labels={'x': 'Dimension 1', 'y': 'Dimension 2'})
    st.plotly_chart(fig)


def bootstrap(data, column, n_iterations=1000):
    boot_means = [np.mean(np.random.choice(data[column].dropna(), size=len(data[column]), replace=True))
                  for _ in range(n_iterations)]
    lower = np.percentile(boot_means, 2.5)
    upper = np.percentile(boot_means, 97.5)
    st.write(f"### Bootstrap-Konfidenzintervall für {column}:")
    st.write(f"95%-Konfidenzintervall: [{lower:.2f}, {upper:.2f}]")

    fig = px.histogram(boot_means, nbins=30, title="Bootstrap-Verteilung", color_discrete_sequence=['green'])
    st.plotly_chart(fig)


def survival_analysis(data, duration_col, event_col):
    kmf = KaplanMeierFitter()
    kmf.fit(data[duration_col], event_observed=data[event_col])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_["KM_estimate"], mode="lines",
                             name="Überlebenskurve"))
    fig.update_layout(title="Kaplan-Meier Überlebenskurve", xaxis_title="Zeit",
                      yaxis_title="Überlebenswahrscheinlichkeit")
    st.plotly_chart(fig)


# Hauptskript
def main():
    st.title("Datenanalyse Dashboard")
    data = upload_csv()
    summarize_data(data)

    while True:
        # Füge `key`-Parameter zu jedem `selectbox` hinzu, um Duplikate zu vermeiden
        option = st.selectbox("Wähle eine statistische Berechnung:", [
            "Korrelationsanalyse", "Verteilungsanalyse", "Hypothesentest", "Zusätzliche Statistiken",
            "Ausreißererkennung", "Zeitreihenanalyse", "K-Means Clustering", "Feature Importance",
            "t-SNE Visualisierung", "Bootstrap-Konfidenzintervalle", "Überlebensanalyse", "Beenden"], key="option_selectbox")

        if option == "Korrelationsanalyse":
            method = st.selectbox("Korrelationsmethode wählen:", ["pearson", "spearman", "kendall"], key="method_selectbox")
            numerical_cols = list(data.select_dtypes(include=['float64', 'int64']).columns)
            correlation_analysis(data[numerical_cols], method=method)
        elif option == "Verteilungsanalyse":
            column = st.selectbox("Spalte für Verteilungsanalyse auswählen:", data.columns, key="column_dist_analysis")
            distribution_analysis(data, column)
        elif option == "Hypothesentest":
            column = st.selectbox("Spalte für Hypothesentest auswählen:", data.columns, key="column_hypothesis")
            test_value = st.number_input("Referenzwert für den Test:", value=0.0)
            hypothesis_testing(data, column, test_value)
        elif option == "Zusätzliche Statistiken":
            column = st.selectbox("Spalte für zusätzliche Statistiken auswählen:", data.columns, key="column_additional_stats")
            additional_statistics(data, column)
        elif option == "Ausreißererkennung":
            column = st.selectbox("Spalte für Ausreißeranalyse auswählen:", data.columns, key="column_outliers")
            detect_outliers(data, column)
        elif option == "Zeitreihenanalyse":
            column = st.selectbox("Spalte für Zeitreihenanalyse auswählen:", data.columns, key="column_time_series")
            freq = st.number_input("Frequenz für saisonale Zerlegung angeben:", min_value=1)
            time_series_analysis(data, column, freq)
        elif option == "K-Means Clustering":
            columns = st.text_input("Spalten für Clustering auswählen (kommagetrennt):", key="columns_clustering").split(",")
            n_clusters = st.number_input("Anzahl der Cluster:", min_value=2, value=3)
            kmeans_clustering(data, columns, n_clusters)
        elif option == "Feature Importance":
            target = st.selectbox("Zielvariable (Target):", data.columns, key="target_feature_importance")
            features = st.text_input("Feature-Spalten auswählen (kommagetrennt):", key="features_importance").split(",")
            feature_importance(data, target, features)
        elif option == "t-SNE Visualisierung":
            columns = st.text_input("Spalten für t-SNE auswählen (kommagetrennt):", key="columns_tsne").split(",")
            tsne_visualization(data, columns)
        elif option == "Bootstrap-Konfidenzintervalle":
            column = st.selectbox("Spalte für Bootstrap-Konfidenzintervalle auswählen:", data.columns, key="column_bootstrap")
            bootstrap(data, column)
        elif option == "Überlebensanalyse":
            duration_col = st.selectbox("Dauer-Spalte auswählen:", data.columns, key="duration_column")
            event_col = st.selectbox("Ereignis-Spalte auswählen:", data.columns, key="event_column")
            survival_analysis(data, duration_col, event_col)
        elif option == "Beenden":
            st.write("Programm beendet.")
            break


if __name__ == "__main__":
    main()
