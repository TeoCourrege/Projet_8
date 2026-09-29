"""Train a demo credit-scoring pipeline on synthetic data and export it for the API.

Fallback used when no MLflow model is available (CI, tests, fresh Docker
container). The production model comes from the legacy MLflow runs on the
Home Credit data, exported with ``scripts/export_mlflow_model.py``. Same
recipe as ``legacy_mlops_project/run.py``: LightGBM wrapped in isotonic
calibration inside a preprocessing pipeline, with the decision threshold
picked via a custom cost function that penalises false negatives (missed
defaults) ten times more than false positives.

Usage:
    uv run python scripts/train_model.py
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from scoring_api.features import FEATURE_COLUMNS, RAW_COLUMNS, build_preprocessor, prepare_features


def compute_cost_curve(
    y_true: np.ndarray, preds: np.ndarray, fn_weight: int = 10, fp_weight: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.linspace(0.01, 0.99, 197)
    costs = []
    for t in thresholds:
        y_pred = (preds >= t).astype(int)
        _, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
        costs.append(fn_weight * fn + fp_weight * fp)
    return thresholds, np.array(costs)


def find_best_threshold(thresholds: np.ndarray, costs: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmin(costs))
    return float(thresholds[idx]), float(costs[idx])


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df["TARGET"]
    X = prepare_features(df[RAW_COLUMNS])
    return X, y


def build_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()
    base_estimator = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=1001,
        n_jobs=-1,
        verbosity=-1,
    )
    calibrated = CalibratedClassifierCV(estimator=base_estimator, method="isotonic", cv=3)
    return Pipeline([("preprocess", preprocessor), ("model", calibrated)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/credit_applications.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--reference-output", type=Path, default=Path("data/processed/reference_sample.csv")
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Fichier introuvable : {args.data}\n"
            "Lancez d'abord : uv run python scripts/generate_synthetic_data.py"
        )

    X, y = load_dataset(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(
        f"Train: {len(X_train)} | Test: {len(X_test)} | "
        f"Taux de défaut (train) : {y_train.mean():.2%}"
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_proba)

    thresholds, costs = compute_cost_curve(y_test.values, test_proba)
    best_threshold, _ = find_best_threshold(thresholds, costs)

    y_pred = (test_proba >= best_threshold).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"AUC (holdout)        : {auc:.4f}")
    print(f"Seuil optimal (coût) : {best_threshold:.3f}")
    print(f"Précision @ seuil    : {precision:.4f}")
    print(f"Rappel @ seuil       : {recall:.4f}")

    print("Ré-entraînement sur 100% des données disponibles pour l'export...")
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, args.model_dir / "pipeline.joblib")

    metadata = {
        "model_version": datetime.now(UTC).strftime("v%Y%m%d_%H%M%S"),
        "trained_at": datetime.now(UTC).isoformat(),
        "threshold": best_threshold,
        "metrics": {
            "roc_auc_holdout": round(float(auc), 5),
            "precision_at_threshold": round(float(precision), 5),
            "recall_at_threshold": round(float(recall), 5),
            "n_train_rows": len(X),
            "default_rate": round(float(y.mean()), 5),
        },
        "n_features": int(X.shape[1]),
        "feature_columns": FEATURE_COLUMNS,
        "source_data": f"{args.data} (synthétique)",
    }
    (args.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Drift-monitoring reference, same format as scripts/export_mlflow_model.py.
    args.reference_output.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(args.data)[RAW_COLUMNS + ["TARGET"]].to_csv(args.reference_output, index=False)

    print(f"Modèle sauvegardé -> {args.model_dir}/pipeline.joblib")
    print(f"Métadonnées       -> {args.model_dir}/metadata.json")
    print(f"Référence drift   -> {args.reference_output}")


if __name__ == "__main__":
    main()
