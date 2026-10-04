"""Streamlit monitoring dashboard.

Reads the reference training data + the live production logs written by the
API, and shows the metrics Chloé asked for: predicted-score distribution,
API latency, and feature/score drift versus the reference dataset.

Usage:
    uv run streamlit run dashboard/monitoring_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scoring_api.features import RAW_COLUMNS, prepare_features
from scoring_api.monitoring.drift import build_drift_report, load_production_logs, psi

# --- Palette (fixed, semantic — not cycled) --------------------------------
COLOR_REFERENCE = "#4C6EF5"  # blue
COLOR_CURRENT = "#F76707"  # orange — colorblind-safe pairing with blue
STATUS_GOOD = "#2F9E44"
STATUS_WARNING = "#F08C00"
STATUS_CRITICAL = "#E03131"

REFERENCE_PATH = Path("data/processed/reference_sample.csv")
LOG_PATH = Path("logs/predictions.jsonl")
MODEL_DIR = Path("models")

st.set_page_config(page_title="Monitoring — Scoring Crédit", layout="wide")
st.title("Monitoring du modèle de scoring — Prêt à Dépenser")


@st.cache_data(ttl=15)
def load_data():
    reference_df = pd.read_csv(REFERENCE_PATH)[RAW_COLUMNS] if REFERENCE_PATH.exists() else pd.DataFrame()
    current_df = load_production_logs(LOG_PATH)
    return reference_df, current_df


@st.cache_resource
def load_pipeline():
    pipeline_path = MODEL_DIR / "pipeline.joblib"
    return joblib.load(pipeline_path) if pipeline_path.exists() else None


reference_df, current_df = load_data()
pipeline = load_pipeline()

if current_df.empty:
    st.warning(
        "Aucun log de production trouvé. Lancez l'API puis générez du trafic : "
        "`uv run python scripts/simulate_traffic.py --n-requests 300`"
    )
    st.stop()

# --- KPI row ----------------------------------------------------------------
n_requests = len(current_df)
refuse_rate = (current_df["decision"] == "REFUSE").mean()
latency = pd.to_numeric(current_df["latency_ms"], errors="coerce").dropna()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Requêtes traitées", f"{n_requests:,}".replace(",", " "))
col2.metric("Taux de refus", f"{refuse_rate:.1%}")
col3.metric("Latence moyenne", f"{latency.mean():.1f} ms")
col4.metric("Latence p95", f"{latency.quantile(0.95):.1f} ms")

st.divider()

# --- Score distribution: reference vs. production --------------------------
left, right = st.columns(2)

with left:
    st.subheader("Distribution des scores prédits")
    current_scores = pd.to_numeric(current_df["score"], errors="coerce").dropna()
    fig = go.Figure()
    if pipeline is not None and not reference_df.empty:
        reference_scores = pipeline.predict_proba(prepare_features(reference_df))[:, 1]
        fig.add_trace(
            go.Histogram(
                x=reference_scores,
                name="Référence (entraînement)",
                histnorm="probability density",
                marker_color=COLOR_REFERENCE,
                opacity=0.6,
                nbinsx=30,
            )
        )
    fig.add_trace(
        go.Histogram(
            x=current_scores,
            name="Production",
            histnorm="probability density",
            marker_color=COLOR_CURRENT,
            opacity=0.6,
            nbinsx=30,
        )
    )
    fig.update_layout(barmode="overlay", legend={"orientation": "h", "y": -0.2}, height=380)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Décisions")
    decision_counts = current_df["decision"].value_counts()
    fig = go.Figure(
        go.Bar(
            x=decision_counts.index,
            y=decision_counts.values,
            marker_color=[
                STATUS_GOOD if d == "ACCEPTE" else STATUS_CRITICAL for d in decision_counts.index
            ],
        )
    )
    fig.update_layout(height=380, yaxis_title="Nombre de demandes")
    st.plotly_chart(fig, use_container_width=True)

# --- Latency over time -------------------------------------------------------
st.subheader("Temps de réponse de l'API (par requête)")
latency_series = pd.to_numeric(current_df["latency_ms"], errors="coerce")
fig = go.Figure(
    go.Scatter(
        y=latency_series,
        mode="lines",
        line={"color": COLOR_REFERENCE, "width": 2},
        name="Latence",
    )
)
fig.add_hline(
    y=float(latency.quantile(0.95)),
    line_dash="dash",
    line_color=STATUS_WARNING,
    annotation_text="p95",
)
fig.update_layout(height=320, xaxis_title="Requête (ordre chronologique)", yaxis_title="ms")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Feature drift table -----------------------------------------------------
st.subheader("Dérive des données (PSI vs. référence)")
if reference_df.empty:
    st.info("Pas de données de référence — impossible de calculer le drift.")
else:
    score_psi = psi(
        pd.Series(pipeline.predict_proba(prepare_features(reference_df))[:, 1])
        if pipeline is not None
        else pd.Series(dtype=float),
        current_scores,
    )
    st.metric("PSI du score prédit", f"{score_psi:.3f}" if not np.isnan(score_psi) else "N/A")

    drift_table = build_drift_report(reference_df, current_df)

    def _highlight_level(row):
        color = {
            "stable": STATUS_GOOD,
            "drift modere": STATUS_WARNING,
            "drift significatif": STATUS_CRITICAL,
        }.get(row["level"], "#868E96")
        return [f"color: {color}; font-weight: 600" if c == "level" else "" for c in row.index]

    st.dataframe(
        drift_table.style.apply(_highlight_level, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "PSI < 0.10 : stable · 0.10-0.25 : drift modéré · > 0.25 : drift significatif "
        "(seuils standards en scoring crédit)."
    )
