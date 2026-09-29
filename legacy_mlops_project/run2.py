"""
Hyperparameter grid search pipeline.

Runs GridSearchCV over a configurable parameter grid for a single model,
evaluates the best estimator with OOF predictions, fits an optional
calibrated variant, and logs everything to MLflow.

Run-name format:
    grid_search__{ModelLabel}__{YYYYMMDD_HHMMSS}
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

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
DEBUG = False
RUN_PREFIX = "grid_search"

MODEL_LABEL = "LightGBM"
BASE_ESTIMATOR = LGBMClassifier(
    objective="binary",
    class_weight="balanced",
    random_state=1001,
    n_jobs=-1,
    verbosity=-1,
)

# Keys must follow the sklearn Pipeline convention: "model__<param_name>"
PARAM_GRID: dict[str, list[Any]] = {
    "model__n_estimators": [200, 500],
    "model__learning_rate": [0.03, 0.1],
    "model__num_leaves": [31, 63],
    #"model__max_depth": [-1, 8],
}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    base_estimator: Any = BASE_ESTIMATOR,
    param_grid: dict[str, list[Any]] = PARAM_GRID,
    model_label: str = MODEL_LABEL,
    n_folds: int = N_FOLDS,
) -> None:
    setup_mlflow()
    ts = now_ts()
    safe_label = model_label.replace(" ", "_")
    run_name = f"{RUN_PREFIX}__{safe_label}__{ts}"
    run_dir = make_run_dir(EXPERIMENTS_DIR, run_name)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1001)

    print(f"\n{'=' * 60}")
    print(f"  Grid Search — {model_label}")
    print(f"  Run : {run_name}")
    print(f"{'=' * 60}")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("pipeline", RUN_PREFIX)
        mlflow.set_tag("model", model_label)
        mlflow.log_param("n_folds", n_folds)
        mlflow.log_param("model_label", model_label)
        
        # Log the search space (stringify lists so MLflow accepts them)
        search_space = {
            k.removeprefix("model__"): str(v) for k, v in param_grid.items()
        }
        mlflow.log_params({f"grid.{k}": v for k, v in search_space.items()})
        
        # -----------------------------------------------------------------------
        # Grid search
        # -----------------------------------------------------------------------
        preprocessor = build_preprocessor(X)
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", copy.deepcopy(base_estimator)),
        ])
        
        grid = GridSearchCV(
            pipe,
            param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        grid.fit(X, y)
        print(1)
        best_params = {
            k.removeprefix("model__"): v for k, v in grid.best_params_.items()
        }
        mlflow.log_params({f"best.{k}": v for k, v in best_params.items()})
        mlflow.log_metric("gs_cv_roc_auc", grid.best_score_)
        print(1)
        

        print(f"\n  Best CV AUC : {grid.best_score_:.5f}")
        print(f"  Best params : {best_params}")

        # -----------------------------------------------------------------------
        # OOF evaluation with best estimator
        # -----------------------------------------------------------------------
        print("\n  OOF evaluation with best estimator:")
        oof_preds = np.zeros(len(y))
        fold_aucs: list[float] = []

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
            fold_pipe = copy.deepcopy(grid.best_estimator_)
            fold_pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = fold_pipe.predict_proba(X.iloc[va_idx])[:, 1]
            oof_preds[va_idx] = preds

            fold_auc = roc_auc_score(y.iloc[va_idx], preds)
            fold_aucs.append(fold_auc)
            mlflow.log_metric("fold_auc", fold_auc, step=fold)
            print(f"  Fold {fold + 1}/{n_folds}  AUC = {fold_auc:.5f}")

        # --- Aggregate metrics
        oof_auc = roc_auc_score(y, oof_preds)
        log_calibration_curve_artifact(
            y.values,
            oof_preds,
            f"{model_label}_uncalibrated",
            run_dir,
        )
        precision = precision_score(y, (oof_preds > 0.5).astype(int))
        recall = recall_score(y, (oof_preds > 0.5).astype(int))

        mlflow.log_metric("oof_roc_auc", oof_auc)
        mlflow.log_metric("cv_auc_mean", float(np.mean(fold_aucs)))
        mlflow.log_metric("cv_auc_std", float(np.std(fold_aucs)))
        mlflow.log_metric("precision_05", precision)
        mlflow.log_metric("recall_05", recall)

        print(f"\n  OOF AUC : {oof_auc:.5f}")
        print(f"  CV AUC  : {np.mean(fold_aucs):.5f} ± {np.std(fold_aucs):.5f}")

        # -----------------------------------------------------------------------
        # Artifacts — plots
        # -----------------------------------------------------------------------
        log_roc_curve_artifact(y.values, oof_preds, model_label, run_dir)
        log_cost_curve_artifact(y.values, oof_preds, run_dir)

        best_estimator = grid.best_estimator_["model"]
        if hasattr(best_estimator, "feature_importances_"):
            feature_names = list(grid.best_estimator_["preprocess"].get_feature_names_out())
            log_feature_importance_artifact(
                best_estimator.feature_importances_, feature_names, model_label, run_dir
            )

        # -----------------------------------------------------------------------
        # Calibrated model (OOF evaluation)
        # -----------------------------------------------------------------------
        print("\n  Fitting calibrated model (isotonic)…")
        calibrated = CalibratedClassifierCV(
            estimator=copy.deepcopy(grid.best_estimator_),
            method="isotonic",
            cv=3,
        )
        calibrated.fit(X, y)

        cal_oof = np.zeros(len(y))
        for tr_idx, va_idx in cv.split(X, y):
            cal_fold = copy.deepcopy(calibrated)
            cal_fold.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            cal_oof[va_idx] = cal_fold.predict_proba(X.iloc[va_idx])[:, 1]

        cal_auc = roc_auc_score(y, cal_oof)
        mlflow.log_metric("calibrated_oof_auc", cal_auc)
        print(f"  Calibrated OOF AUC : {cal_auc:.5f}")

        # -----------------------------------------------------------------------
        # Save models
        # -----------------------------------------------------------------------
        pipeline_path = run_dir / "best_pipeline.joblib"
        joblib.dump(grid.best_estimator_, pipeline_path)
        mlflow.log_artifact(str(pipeline_path))

        cal_path = run_dir / "calibrated_pipeline.joblib"
        joblib.dump(calibrated, cal_path)
        mlflow.log_artifact(str(cal_path))

        # Run summary
        summary_path = run_dir / "run_summary.txt"
        with open(summary_path, "w") as fh:
            fh.write(f"Model           : {model_label}\n")
            fh.write(f"Best Params     : {best_params}\n")
            fh.write(f"Grid CV AUC     : {grid.best_score_:.5f}\n")
            fh.write(f"OOF AUC         : {oof_auc:.5f}\n")
            fh.write(f"CV AUC mean     : {np.mean(fold_aucs):.5f}\n")
            fh.write(f"CV AUC std      : {np.std(fold_aucs):.5f}\n")
            fh.write(f"Calibrated AUC  : {cal_auc:.5f}\n")
        mlflow.log_artifact(str(summary_path))

        mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="pipeline")

    print(f"  Artifacts → {run_dir}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nrows = 10_000 if DEBUG else None
    X, y = load_and_prepare(DATA_PATH, nrows=nrows)
    run_grid_search(X, y)

