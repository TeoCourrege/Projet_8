"""Send simulated client traffic to a running scoring API instance.

Replays real credit applications from the Home Credit
``application_test.csv`` (new applications the model has never seen), or,
when it is not available, rows of the drift reference sample. Populates
``logs/predictions.jsonl`` so the monitoring dashboard and
``scripts/run_drift_analysis.py`` have production-like data to work with.
``--drift`` shifts the population on purpose to demonstrate drift detection.

Usage:
    uv run python scripts/simulate_traffic.py --n-requests 300
    uv run python scripts/simulate_traffic.py --n-requests 150 --drift
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from pydantic import ValidationError

from scoring_api.features import HC_RAW_COLUMNS, RAW_COLUMNS, from_home_credit_frame
from scoring_api.schemas import ClientData

DEFAULT_SOURCE = Path("data/raw/application_test.csv")
FALLBACK_SOURCE = Path("data/processed/reference_sample.csv")


def load_applications(source: Path, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample ``n`` applications from ``source``, in the API's input format."""
    if source.exists():
        df = pd.read_csv(source, usecols=HC_RAW_COLUMNS)
        df = df.sample(n=min(n, len(df)), random_state=int(rng.integers(1 << 31)))
        return from_home_credit_frame(df)
    if FALLBACK_SOURCE.exists():
        print(f"[i] {source} introuvable — utilisation de {FALLBACK_SOURCE}")
        df = pd.read_csv(FALLBACK_SOURCE)[RAW_COLUMNS]
        return df.sample(n=min(n, len(df)), random_state=int(rng.integers(1 << 31)))
    raise SystemExit(
        f"Ni {source} ni {FALLBACK_SOURCE} n'existent. Exportez un modèle "
        "(scripts/export_mlflow_model.py) ou entraînez le modèle de démo (scripts/train_model.py)."
    )


def apply_drift(df: pd.DataFrame) -> pd.DataFrame:
    # Simulates a population shift: higher incomes (e.g. a new marketing
    # segment) and lower external-bureau scores.
    df = df.copy()
    df["income_total"] = df["income_total"] * 1.6
    for col in ("ext_source_1", "ext_source_2", "ext_source_3"):
        df[col] = (df[col] - 0.15).clip(lower=0)
    return df


def to_payload(row: pd.Series) -> dict:
    payload = {}
    for col in RAW_COLUMNS:
        value = row[col]
        if pd.isna(value):
            payload[col] = None
        elif isinstance(value, (bool, np.bool_)):
            payload[col] = bool(value)
        elif isinstance(value, (int, np.integer)) or col in ("children", "family_members"):
            payload[col] = int(value)
        elif isinstance(value, (float, np.floating)):
            payload[col] = round(float(value), 4)
        else:
            payload[col] = str(value)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n-requests", type=int, default=300)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--drift", action="store_true", help="Simule une dérive de population")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    applications = load_applications(args.source, args.n_requests, rng)
    if args.drift:
        applications = apply_drift(applications)

    payloads: list[dict] = []
    skipped = 0
    for _, row in applications.iterrows():
        payload = to_payload(row)
        try:
            ClientData(**payload)  # skip rows the API would reject (e.g. missing annuity)
        except ValidationError:
            skipped += 1
            continue
        payloads.append(payload)

    latencies: list[float] = []
    errors = 0
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        for i, payload in enumerate(payloads):
            start = time.perf_counter()
            resp = client.post("/predict", json=payload)
            latencies.append((time.perf_counter() - start) * 1000)
            if resp.status_code != 200:
                errors += 1
                print(f"  [!] requête {i} -> {resp.status_code}: {resp.text[:200]}")
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(payloads)} requêtes envoyées...")

    if not latencies:
        raise SystemExit("Aucune requête envoyée.")
    latencies_arr = np.array(latencies)
    print(f"\nRequêtes envoyées : {len(payloads)} (drift={args.drift}, ignorées={skipped})")
    print(f"Erreurs           : {errors}")
    print(
        "Latence p50/p95   : "
        f"{np.percentile(latencies_arr, 50):.1f} / {np.percentile(latencies_arr, 95):.1f} ms"
    )


if __name__ == "__main__":
    main()
