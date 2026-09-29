"""
Model selection pipeline.

Evaluates a registry of models with stratified k-fold cross-validation and
logs every run to MLflow under the shared experiment defined in utils.

Run-name format:
    model_selection__{ModelLabel}__{param1=v1_param2=v2}__{YYYYMMDD_HHMMSS}
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier


sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    build_preprocessor,
    load_and_prepare,
    log_cost_curve_artifact,
    log_feature_importance_artifact,
    log_roc_curve_artifact,
    make_run_dir,
    now_ts,
    setup_mlflow,
    log_calibration_curve_artifact,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "../data/raw/application_train.csv"
EXPERIMENTS_DIR = "../experiments"
N_FOLDS = 5
STRATIFIED = True
DEBUG = False
RUN_PREFIX = "model_selection"

# Each entry defines a candidate model.
# - label       : human-readable name (used in run name and plots)
# - estimator   : unfitted sklearn-compatible estimator
# - log_params  : subset of params to embed in the MLflow run name and tags
MODELS: dict[str, dict[str, Any]] = {
    """
    "random_forest":{
        "label": "RandomForest",
        "estimator": CalibratedClassifierCV(
            estimator=RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=1001,
        ), 
        method="isotonic",
        cv=3,),
        "log_params": {"n_estimators": 100, "learning_rate": 0.05},             
    },
    

    "mlp": {
        "label": "MLP",
        "estimator": CalibratedClassifierCV(MLPClassifier(
            random_state=1001,

        ),
        method="isotonic",
        cv=3,),
        "log_params": {"hidden_layer_sizes":(100,), "activation":'relu',
                        "solver":'adam', "alpha":0.0001,},
    },
    """
    "lgbm": {
        "label": "LightGBM",
        "estimator": CalibratedClassifierCV(LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=1001,
            n_jobs=-1,
            verbosity=-1,
        ),
        method="isotonic",
        cv=3,),
        "log_params": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_run_name(label: str, log_params: dict[str, Any], ts: str) -> str:
    param_str = "_".join(f"{k}={v}" for k, v in log_params.items())
    safe_label = label.replace(" ", "_")
    return f"{RUN_PREFIX}__{safe_label}__{param_str}__{ts}"


def _cv_folds(n_folds: int, stratified: bool) -> KFold | StratifiedKFold:
    cls = StratifiedKFold if stratified else KFold
    return cls(n_splits=n_folds, shuffle=True, random_state=1001)


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def _evaluate_model(
    config: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    folds: KFold | StratifiedKFold,
    run_dir: Path,
) -> None:
    label: str = config["label"]

    preprocessor = build_preprocessor(X)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", copy.deepcopy(config["estimator"])),
    ])

    oof_preds = np.zeros(len(y))
    fold_aucs: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(folds.split(X, y)):
        fold_pipe = copy.deepcopy(pipe)
        fold_pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = fold_pipe.predict_proba(X.iloc[va_idx])[:, 1]
        oof_preds[va_idx] = preds

        fold_auc = roc_auc_score(y.iloc[va_idx], preds)
        fold_aucs.append(fold_auc)
        mlflow.log_metric("fold_auc", fold_auc, step=fold)
        print(f"  Fold {fold + 1}/{folds.n_splits}  AUC = {fold_auc:.5f}")

    # --- OOF aggregate metrics
    oof_auc = roc_auc_score(y, oof_preds)
    precision = precision_score(y, (oof_preds > 0.5).astype(int))
    recall = recall_score(y, (oof_preds > 0.5).astype(int))

    mlflow.log_metric("oof_roc_auc", oof_auc)
    mlflow.log_metric("cv_auc_mean", float(np.mean(fold_aucs)))
    mlflow.log_metric("cv_auc_std", float(np.std(fold_aucs)))
    mlflow.log_metric("precision_05", precision)
    mlflow.log_metric("recall_05", recall)

    print(f"\n  OOF AUC : {oof_auc:.5f}")
    print(f"  CV AUC  : {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f}")

    # --- Artifacts
    log_roc_curve_artifact(y.values, oof_preds, label, run_dir)
    log_cost_curve_artifact(y.values, oof_preds, run_dir)

    log_calibration_curve_artifact(
        y.values,
        oof_preds,
        label,
        run_dir,
    )

    # --- Final model on full data
    final_pipe = copy.deepcopy(pipe)
    final_pipe.fit(X, y)

    estimator = final_pipe["model"]
    if hasattr(estimator, "feature_importances_"):
        feature_names = list(final_pipe["preprocess"].get_feature_names_out())
        log_feature_importance_artifact(
            estimator.feature_importances_, feature_names, label, run_dir
        )

    model_path = run_dir / "pipeline.joblib"
    joblib.dump(final_pipe, model_path)
    mlflow.log_artifact(str(model_path))
    # cloudpickle rather than MLflow's default skops: skops rejects tree,
    # calibration and LightGBM types unless each is whitelisted, and
    # scripts/export_mlflow_model.py would need the same list to load it.
    mlflow.sklearn.log_model(
        final_pipe, name="pipeline", serialization_format="cloudpickle"
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_model_selection(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, dict[str, Any]] = MODELS,
    n_folds: int = N_FOLDS,
    stratified: bool = STRATIFIED,
) -> None:
    setup_mlflow()
    folds = _cv_folds(n_folds, stratified)

    for key, config in models.items():
        label: str = config["label"]
        log_params: dict = config.get("log_params", {})
        ts = now_ts()
        run_name = _build_run_name(label, log_params, ts)
        run_dir = make_run_dir(EXPERIMENTS_DIR, run_name)

        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"  Run : {run_name}")
        print(f"{'=' * 60}")

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("pipeline", RUN_PREFIX)
            mlflow.set_tag("model_key", key)
            mlflow.log_param("model_label", label)
            mlflow.log_param("n_folds", n_folds)
            mlflow.log_param("stratified", stratified)
            mlflow.log_params({f"model.{k}": v for k, v in log_params.items()})

            _evaluate_model(config, X, y, folds, run_dir)

        print(f"  Artifacts → {run_dir}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nrows = 10_000 if DEBUG else None
    X, y = load_and_prepare(DATA_PATH, nrows=nrows)
    run_model_selection(X, y)