"""Generate a synthetic credit-application dataset.

Fallback for environments without the Home Credit dataset (CI, tests, a
fresh Docker container): same fields and categories as the API, so the
pipeline can be trained and exercised end-to-end. The production model is
trained on the real data by the legacy MLflow runs and exported with
``scripts/export_mlflow_model.py``.

Usage:
    uv run python scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scoring_api.schemas import (
    CONTRACT_TYPES,
    EDUCATION_TYPES,
    FAMILY_STATUSES,
    GENDERS,
    HOUSING_TYPES,
    OCCUPATION_TYPES,
    ORGANIZATION_TYPES,
)

RNG_SEED = 42


def _sample_categoricals(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    # `None` mixed into the occupation/organization pools reproduces the
    # real dataset's missingness for clients without a declared occupation.
    occupation_pool = list(OCCUPATION_TYPES) + [None]
    organization_pool = list(ORGANIZATION_TYPES) + [None]
    return {
        "gender": rng.choice(GENDERS, size=n, p=[0.45, 0.55]),
        "own_car": rng.choice([True, False], size=n, p=[0.35, 0.65]),
        "own_realty": rng.choice([True, False], size=n, p=[0.6, 0.4]),
        "contract_type": rng.choice(CONTRACT_TYPES, size=n, p=[0.9, 0.1]),
        "education_type": rng.choice(
            EDUCATION_TYPES, size=n, p=[0.65, 0.22, 0.08, 0.04, 0.01]
        ),
        "family_status": rng.choice(FAMILY_STATUSES, size=n, p=[0.55, 0.22, 0.1, 0.07, 0.06]),
        "housing_type": rng.choice(
            HOUSING_TYPES, size=n, p=[0.85, 0.06, 0.04, 0.02, 0.02, 0.01]
        ),
        "occupation_type": rng.choice(occupation_pool, size=n),
        "organization_type": rng.choice(organization_pool, size=n),
    }


def generate_dataset(n_samples: int, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    categoricals = _sample_categoricals(n_samples, rng)

    age_years = np.clip(rng.normal(43, 11, n_samples), 19, 69)
    years_employed = np.clip(
        rng.normal(6, 6, n_samples) + (age_years - 30) * 0.05, 0, age_years - 16
    )
    children = rng.poisson(0.4, n_samples).clip(0, 6)
    family_members = (children + rng.choice([1, 2], n_samples, p=[0.35, 0.65])).clip(1, 10)

    income_total = np.clip(rng.lognormal(mean=11.2, sigma=0.45, size=n_samples), 30_000, 1_500_000)
    credit_amount = np.clip(income_total * rng.uniform(1.5, 6.5, n_samples), 45_000, 2_500_000)
    annuity_amount = credit_amount / rng.uniform(8, 30, n_samples)
    goods_price = credit_amount * rng.uniform(0.85, 1.05, n_samples)

    ext_source_1 = np.clip(rng.normal(0.5, 0.18, n_samples), 0, 1)
    ext_source_2 = np.clip(rng.normal(0.52, 0.16, n_samples), 0, 1)
    ext_source_3 = np.clip(rng.normal(0.5, 0.2, n_samples), 0, 1)
    for arr, miss_rate in ((ext_source_1, 0.3), (ext_source_2, 0.02), (ext_source_3, 0.15)):
        arr[rng.random(n_samples) < miss_rate] = np.nan

    df = pd.DataFrame(
        {
            "client_id": np.arange(100_000, 100_000 + n_samples),
            **categoricals,
            "children": children.astype(int),
            "family_members": family_members.astype(int),
            "income_total": income_total.round(2),
            "credit_amount": credit_amount.round(2),
            "annuity_amount": annuity_amount.round(2),
            "goods_price": goods_price.round(2),
            "age_years": age_years.round(1),
            "years_employed": years_employed.round(1),
            "ext_source_1": ext_source_1,
            "ext_source_2": ext_source_2,
            "ext_source_3": ext_source_3,
        }
    )

    # --- Latent risk score -> TARGET (heavier annuity burden, thin credit
    # bureau history and low external scores push default probability up).
    # Coefficients/noise were hand-tuned only to get a plausible ~3-4%
    # default rate and a LightGBM holdout AUC around ~0.75-0.8 for demo
    # purposes — not meant to model real credit risk. Swap in the real
    # Home Credit dataset for anything beyond local pipeline testing.
    payment_rate = df["annuity_amount"] / df["credit_amount"]
    annuity_income_perc = df["annuity_amount"] / df["income_total"]
    income_per_person = df["income_total"] / df["family_members"]
    credit_goods_perc = df["credit_amount"] / df["goods_price"]
    ext_mean = df[["ext_source_1", "ext_source_2", "ext_source_3"]].mean(axis=1)

    risk = (
        -1.4
        + 6.0 * payment_rate
        + 4.0 * annuity_income_perc
        - 8.0 * ext_mean.fillna(ext_mean.mean())
        + 1.5 * (credit_goods_perc - 1)
        - 0.000002 * income_per_person
        + 0.4 * (df["years_employed"] < 0.5).astype(int)
        + 0.3 * (df["age_years"] < 25).astype(int)
        + rng.normal(0, 0.15, n_samples)
    )
    proba_default = 1 / (1 + np.exp(-risk))
    df["TARGET"] = rng.binomial(1, proba_default)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--output", type=Path, default=Path("data/raw/credit_applications.csv"))
    args = parser.parse_args()

    df = generate_dataset(args.n_samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"{len(df)} lignes générées -> {args.output}")
    print(f"Taux de défaut (TARGET=1) : {df['TARGET'].mean():.2%}")


if __name__ == "__main__":
    main()
