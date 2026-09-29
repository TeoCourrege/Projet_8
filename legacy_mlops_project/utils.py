"""Shared utilities for model selection and grid search pipelines."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# Keep everything local: no anonymous usage telemetry sent by MLflow.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve


EXPERIMENT_NAME = "model_experiments"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{(_PROJECT_ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = (_PROJECT_ROOT / "mlruns").as_uri()

# Feature engineering / preprocessing are shared with the API package so the
# models logged here can be served as-is (see scripts/export_mlflow_model.py).
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from scoring_api.features import (  # noqa: E402
    FEATURE_COLUMNS,
    HC_RAW_COLUMNS,
    engineer_features,
)
from scoring_api.features import build_preprocessor as _build_api_preprocessor  # noqa: E402


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_prepare(path: str, nrows: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
 # Only the application columns the API collects (scoring_api.features.
 # HC_RAW_COLUMNS) are used, so every model logged to MLflow is servable.
 df = pd.read_csv(path, nrows=nrows, usecols=HC_RAW_COLUMNS + ["TARGET"])
 df = df[df["CODE_GENDER"] != "XNA"].reset_index(drop=True)
 y = df["TARGET"]
 X = engineer_features(df[HC_RAW_COLUMNS])[FEATURE_COLUMNS]
 return X, y


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    return _build_api_preprocessor()


# ---------------------------------------------------------------------------
# Metrics & threshold
# ---------------------------------------------------------------------------

def compute_cost_curve(
 y_true: np.ndarray,
 preds: np.ndarray,
 fn_weight: int = 10,
 fp_weight: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
 thresholds = np.linspace(0, 1, 200)
 costs = []
 for t in thresholds:
    y_pred = (preds >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    costs.append(fn_weight * fn + fp_weight * fp)
 return thresholds, np.array(costs)


def find_best_threshold(
 thresholds: np.ndarray, costs: np.ndarray
) -> tuple[float, float]:
 idx = int(np.argmin(costs))
 return float(thresholds[idx]), float(costs[idx])


# ---------------------------------------------------------------------------
# MLflow artifact helpers
# ---------------------------------------------------------------------------

def log_cost_curve_artifact(
 y_true: np.ndarray,
 oof_preds: np.ndarray,
 run_dir: Path,
) -> tuple[float, float]:
 thresholds, costs = compute_cost_curve(y_true, oof_preds)
 best_t, best_cost = find_best_threshold(thresholds, costs)

 fig, ax = plt.subplots()
 ax.plot(thresholds, costs, label="Cost = 10×FN + FP")
 ax.axvline(best_t, linestyle="--", label=f"Best threshold = {best_t:.3f}")
 ax.set_xlabel("Threshold")
 ax.set_ylabel("Cost")
 ax.set_title("Custom Cost Function (OOF)")
 ax.legend()
 fig.tight_layout()
 path = run_dir / "cost_curve.png"
 fig.savefig(path)
 plt.close(fig)

 mlflow.log_artifact(str(path))
 mlflow.log_metric("best_threshold", best_t)
 mlflow.log_metric("min_custom_cost", best_cost)
 return best_t, best_cost


def log_roc_curve_artifact(
 y_true: np.ndarray,
 oof_preds: np.ndarray,
 model_label: str,
 run_dir: Path,
) -> float:
 auc = roc_auc_score(y_true, oof_preds)
 fpr, tpr, _ = roc_curve(y_true, oof_preds)

 fig, ax = plt.subplots()
 ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
 ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
 ax.set_xlabel("FPR")
 ax.set_ylabel("TPR")
 ax.set_title(f"ROC Curve — {model_label}")
 ax.legend()
 fig.tight_layout()
 path = run_dir / "roc_curve.png"
 fig.savefig(path)
 plt.close(fig)

 mlflow.log_artifact(str(path))
 return auc


def log_feature_importance_artifact(
 importances: np.ndarray,
 feature_names: list[str],
 model_label: str,
 run_dir: Path,
 top_n: int = 30,
) -> None:
 fi = (
 pd.Series(importances, index=feature_names)
 .sort_values(ascending=False)
 .head(top_n)
 )
 fig, ax = plt.subplots(figsize=(10, 8))
 fi.plot(kind="barh", ax=ax)
 ax.invert_yaxis()
 ax.set_title(f"Feature Importance (top {top_n}) — {model_label}")
 fig.tight_layout()
 path = run_dir / "feature_importance.png"
 fig.savefig(path)
 plt.close(fig)
 mlflow.log_artifact(str(path))


def log_calibration_curve_artifact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    run_dir: Path,
    n_bins: int = 10,
) -> None:
    """
    Log calibration curves using both uniform and quantile binning.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted probabilities.

    label : str
        Model label.

    run_dir : Path
        MLflow artifact directory.

    n_bins : int
        Number of calibration bins.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, strategy in zip(
        axes,
        ["uniform", "quantile"],
    ):
        frac_pos, mean_pred = calibration_curve(
            y_true,
            y_pred,
            n_bins=n_bins,
            strategy=strategy,
        )

        ax.plot(
            mean_pred,
            frac_pos,
            marker="o",
            linewidth=2,
            label=label,
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Perfect calibration",
        )

        ax.set_title(f"Calibration Curve ({strategy})")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    output_path = run_dir / "calibration_curve.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(str(output_path))



# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

def setup_mlflow(experiment_name: str = EXPERIMENT_NAME) -> None:
 mlflow.set_tracking_uri(TRACKING_URI)
 if mlflow.get_experiment_by_name(experiment_name) is None:
  mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_ROOT)
 mlflow.set_experiment(experiment_name)


def make_run_dir(base: str, run_name: str) -> Path:
 path = Path(base) / run_name
 path.mkdir(parents=True, exist_ok=True)
 return path


def now_ts() -> str:
 return time.strftime("%Y%m%d_%H%M%S")