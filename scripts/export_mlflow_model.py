"""Export a model trained by the legacy MLflow runs into ``models/`` for the API.

Picks the best finished run of the ``model_experiments`` experiment (highest
``oof_roc_auc`` by default, or an explicit ``--run-id``), loads its logged
``pipeline`` model, and writes what the API loads at startup:

- ``models/pipeline.joblib``  — the sklearn pipeline (preprocessing + model),
- ``models/metadata.json``    — version, MLflow run, decision threshold (the
  ``best_threshold`` found by the legacy cost function 10×FN + FP), metrics.

It also writes ``data/processed/reference_sample.csv``: a sample of the
training data in the API's input format, used as the drift-monitoring
reference by ``scripts/run_drift_analysis.py`` and the dashboard.

Usage (after ``cd legacy_mlops_project && uv run --extra training python run.py``):
    uv run --extra training python scripts/export_mlflow_model.py
    uv run --extra training python scripts/export_mlflow_model.py --run-id <id> --register
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

# Keep everything local: no anonymous usage telemetry sent by MLflow.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from scoring_api.features import FEATURE_COLUMNS, HC_RAW_COLUMNS, from_home_credit_frame

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "model_experiments"  # legacy_mlops_project/utils.py
REPORTED_METRICS = (
    "oof_roc_auc",
    "cv_auc_mean",
    "cv_auc_std",
    "calibrated_oof_auc",
    "gs_cv_roc_auc",
    "min_custom_cost",
    "precision_05",
    "recall_05",
)


def select_run(client: MlflowClient, run_id: str | None, metric: str):
    if run_id:
        return client.get_run(run_id)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(
            f"Expérience MLflow '{EXPERIMENT_NAME}' introuvable. Lancez d'abord les runs "
            "legacy (legacy_mlops_project/run.py ou run2.py)."
        )
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED' and metrics.best_threshold >= 0",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(
            f"Aucun run terminé avec un 'best_threshold' dans '{EXPERIMENT_NAME}'."
        )
    return runs[0]


def check_input_contract(pipeline) -> None:
    """Refuse models trained on other columns than the API provides (e.g. runs
    made before the legacy code was restricted to ``HC_RAW_COLUMNS``)."""
    expected = getattr(pipeline, "feature_names_in_", None)
    if expected is None:
        return
    missing = sorted(set(expected) - set(FEATURE_COLUMNS))
    if missing:
        raise SystemExit(
            f"Ce modèle attend {len(expected)} colonnes dont {len(missing)} que l'API ne "
            f"fournit pas (ex. {missing[:5]}). Il a été entraîné avant la restriction aux "
            "colonnes de l'API : relancez les runs legacy puis ré-exportez."
        )


def export_reference_sample(train_data: Path, output: Path, n_rows: int, seed: int) -> None:
    if not train_data.exists():
        print(f"[!] {train_data} introuvable — échantillon de référence non généré.")
        return
    df = pd.read_csv(train_data, usecols=HC_RAW_COLUMNS + ["TARGET"])
    df = df[df["CODE_GENDER"] != "XNA"]
    df = df.sample(n=min(n_rows, len(df)), random_state=seed)
    reference = from_home_credit_frame(df)
    reference["TARGET"] = df["TARGET"].to_numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output, index=False)
    print(f"Référence drift   -> {output} ({len(reference)} lignes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--run-id", default=None, help="Run à exporter (défaut : le meilleur)")
    parser.add_argument("--metric", default="oof_roc_auc", help="Métrique de sélection (max)")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--register",
        metavar="NAME",
        nargs="?",
        const="credit-scoring",
        default=None,
        help="Enregistre aussi le modèle dans le Model Registry (alias 'champion')",
    )
    parser.add_argument(
        "--train-data", type=Path, default=PROJECT_ROOT / "data/raw/application_train.csv"
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/reference_sample.csv",
    )
    parser.add_argument("--reference-rows", type=int, default=20_000)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    run = select_run(client, args.run_id, args.metric)
    run_id = run.info.run_id
    metrics = run.data.metrics
    if "best_threshold" not in metrics:
        raise SystemExit(f"Le run {run_id} n'a pas de métrique 'best_threshold'.")

    model_uri = f"runs:/{run_id}/pipeline"
    print(f"Run sélectionné   : {run.info.run_name} ({run_id})")
    pipeline = mlflow.sklearn.load_model(model_uri)
    check_input_contract(pipeline)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_dir / "pipeline.joblib")

    label = run.data.params.get("model_label", "model").replace(" ", "_")
    trained_at = datetime.fromtimestamp(
        (run.info.end_time or run.info.start_time) / 1000, tz=UTC
    ).isoformat()
    metadata = {
        "model_version": f"{label}-{run_id[:8]}",
        "trained_at": trained_at,
        "threshold": float(metrics["best_threshold"]),
        "metrics": {k: round(float(metrics[k]), 5) for k in REPORTED_METRICS if k in metrics},
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "source_data": str(args.train_data.name),
        "mlflow": {
            "tracking_uri": args.tracking_uri,
            "run_id": run_id,
            "run_name": run.info.run_name,
            "model_uri": model_uri,
        },
    }

    if args.register:
        version = mlflow.register_model(model_uri, args.register)
        client.set_registered_model_alias(args.register, "champion", version.version)
        metadata["mlflow"]["registered_model"] = f"{args.register}/{version.version}"
        print(f"Model Registry    : {args.register} v{version.version} (alias 'champion')")

    (args.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Modèle            -> {args.model_dir / 'pipeline.joblib'}")
    print(f"Métadonnées       -> {args.model_dir / 'metadata.json'}")
    print(f"Seuil de décision : {metadata['threshold']:.4f}")

    export_reference_sample(
        args.train_data, args.reference_output, args.reference_rows, seed=1001
    )


if __name__ == "__main__":
    main()