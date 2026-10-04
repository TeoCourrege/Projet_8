"""Run a data-drift & operational-monitoring analysis comparing the
production logs collected by the API against the reference training data.

Writes ``monitoring_reports/feature_drift.csv`` and
``monitoring_reports/monitoring_summary.json``, and prints a summary — this
is the "script réalisant l'analyse automatique des données stockées"
deliverable. The Streamlit dashboard (``dashboard/monitoring_app.py``)
visualises the same data live.

Usage:
    uv run python scripts/run_drift_analysis.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scoring_api.features import RAW_COLUMNS, prepare_features
from scoring_api.monitoring.drift import build_drift_report, load_production_logs, psi, psi_level


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-data", type=Path, default=Path("data/processed/reference_sample.csv")
    )
    parser.add_argument("--logs", type=Path, default=Path("logs/predictions.jsonl"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("monitoring_reports"))
    args = parser.parse_args()

    if not args.reference_data.exists():
        raise SystemExit(f"Données de référence introuvables : {args.reference_data}")

    reference_df = pd.read_csv(args.reference_data)[RAW_COLUMNS]
    current_df = load_production_logs(args.logs)

    if current_df.empty:
        raise SystemExit(
            f"Aucun log de production trouvé dans {args.logs}.\n"
            "Lancez l'API puis : uv run python scripts/simulate_traffic.py"
        )

    drift_table = build_drift_report(reference_df, current_df)

    pipeline = joblib.load(args.model_dir / "pipeline.joblib")
    reference_scores = pipeline.predict_proba(prepare_features(reference_df))[:, 1]
    current_scores = pd.to_numeric(current_df["score"], errors="coerce").dropna()
    score_psi = psi(pd.Series(reference_scores), current_scores)

    latency = pd.to_numeric(current_df["latency_ms"], errors="coerce").dropna()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    drift_table.to_csv(args.output_dir / "feature_drift.csv", index=False)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_reference_rows": len(reference_df),
        "n_production_rows": len(current_df),
        "score_psi": round(float(score_psi), 4) if not np.isnan(score_psi) else None,
        "score_psi_level": psi_level(score_psi),
        "score_distribution": {
            "reference_mean": round(float(np.mean(reference_scores)), 4),
            "current_mean": round(float(current_scores.mean()), 4) if len(current_scores) else None,
        },
        "latency_ms": {
            "mean": round(float(latency.mean()), 2) if len(latency) else None,
            "p50": round(float(latency.quantile(0.5)), 2) if len(latency) else None,
            "p95": round(float(latency.quantile(0.95)), 2) if len(latency) else None,
            "p99": round(float(latency.quantile(0.99)), 2) if len(latency) else None,
        },
        "n_features_significant_drift": int((drift_table["level"] == "drift significatif").sum())
        if not drift_table.empty
        else 0,
        "n_features_moderate_drift": int((drift_table["level"] == "drift modere").sum())
        if not drift_table.empty
        else 0,
    }
    (args.output_dir / "monitoring_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nTop dérives par feature :")
    print(drift_table.head(10).to_string(index=False))
    print(f"\nRapports -> {args.output_dir}/feature_drift.csv, monitoring_summary.json")


if __name__ == "__main__":
    main()
