"""Data-drift & operational-monitoring utilities.

Implements the Population Stability Index (PSI) and the Kolmogorov-Smirnov
test — the two standard, dependency-light techniques used for credit-scoring
model monitoring — so this module has no hard dependency on a specific
third-party drift library (Evidently, NannyML, ... are valid alternatives,
see README).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from scoring_api.features import RAW_CATEGORICAL_COLUMNS, RAW_NUMERIC_COLUMNS

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25


def psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions.

    < 0.1  : stable · 0.1-0.25 : moderate drift · > 0.25 : significant drift.
    """
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return float("nan")

    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0], quantiles[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=quantiles)
    cur_counts, _ = np.histogram(current, bins=quantiles)

    ref_perc = np.clip(ref_counts / max(len(reference), 1), 1e-4, None)
    cur_perc = np.clip(cur_counts / max(len(current), 1), 1e-4, None)

    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))


def psi_level(value: float) -> str:
    if value is None or np.isnan(value):
        return "N/A"
    if value < PSI_MODERATE_THRESHOLD:
        return "stable"
    if value < PSI_SIGNIFICANT_THRESHOLD:
        return "drift modere"
    return "drift significatif"


def categorical_drift(reference: pd.Series, current: pd.Series) -> float:
    """Total variation distance between two categorical distributions."""
    ref_dist = reference.value_counts(normalize=True, dropna=True)
    cur_dist = current.value_counts(normalize=True, dropna=True)
    categories = set(ref_dist.index) | set(cur_dist.index)
    return float(sum(abs(ref_dist.get(c, 0) - cur_dist.get(c, 0)) for c in categories) / 2)


def build_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature drift table, sorted by severity (highest PSI first)."""
    rows: list[dict] = []

    for col in RAW_NUMERIC_COLUMNS:
        if col not in reference_df or col not in current_df:
            continue
        score = psi(reference_df[col], current_df[col])
        ref_vals = pd.to_numeric(reference_df[col], errors="coerce").dropna()
        cur_vals = pd.to_numeric(current_df[col], errors="coerce").dropna()
        if len(ref_vals) > 1 and len(cur_vals) > 1:
            _, ks_p = ks_2samp(ref_vals, cur_vals)
        else:
            ks_p = float("nan")
        rows.append(
            {
                "feature": col,
                "type": "numeric",
                "psi": round(score, 4) if not np.isnan(score) else None,
                "level": psi_level(score),
                "ks_pvalue": round(float(ks_p), 4) if not np.isnan(ks_p) else None,
            }
        )

    for col in RAW_CATEGORICAL_COLUMNS:
        if col not in reference_df or col not in current_df:
            continue
        score = categorical_drift(reference_df[col], current_df[col])
        rows.append(
            {
                "feature": col,
                "type": "categorical",
                "psi": round(score, 4),
                "level": psi_level(score),
                "ks_pvalue": None,
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def load_production_logs(log_path: Path) -> pd.DataFrame:
    """Parse the JSON-lines prediction log into a flat DataFrame
    (one row per request, inputs + score/decision/latency).
    """
    if not log_path.exists():
        return pd.DataFrame()

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not records:
        return pd.DataFrame()

    inputs = pd.json_normalize([r["input"] for r in records])
    meta = pd.DataFrame(
        [
            {
                "score": r["score"],
                "decision": r["decision"],
                "latency_ms": r["latency_ms"],
                "logged_at": r["logged_at"],
                "model_version": r.get("model_version"),
            }
            for r in records
        ]
    )
    return pd.concat([inputs, meta], axis=1)
