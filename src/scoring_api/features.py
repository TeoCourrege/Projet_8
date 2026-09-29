"""Feature engineering shared by the API and the legacy MLflow training runs
(``legacy_mlops_project/utils.py`` imports it), so the model served by the
API sees exactly the features it was trained on.

The API exposes friendly field names (e.g. ``age_years``); the model is
trained on the Home Credit ``application_train.csv`` column names (e.g. a
negative ``DAYS_BIRTH``). ``to_home_credit_frame`` / ``from_home_credit_frame``
convert between the two, and ``engineer_features`` reproduces the previous
project's ``_feat_eng`` ratios on the Home Credit columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scoring_api.schemas import ClientData

# ---------------------------------------------------------------------------
# API-side (friendly) columns
# ---------------------------------------------------------------------------

RAW_NUMERIC_COLUMNS: list[str] = [
    "children",
    "family_members",
    "income_total",
    "credit_amount",
    "annuity_amount",
    "goods_price",
    "age_years",
    "years_employed",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
]
RAW_BOOL_COLUMNS: list[str] = ["own_car", "own_realty"]
RAW_CATEGORICAL_COLUMNS: list[str] = [
    "gender",
    "contract_type",
    "education_type",
    "family_status",
    "housing_type",
    "occupation_type",
    "organization_type",
]
RAW_COLUMNS: list[str] = RAW_NUMERIC_COLUMNS + RAW_BOOL_COLUMNS + RAW_CATEGORICAL_COLUMNS

# ---------------------------------------------------------------------------
# Model-side (Home Credit) columns
# ---------------------------------------------------------------------------

DAYS_PER_YEAR = 365.25
# Sentinel used by Home Credit for "not employed" (pensioners, unemployed).
DAYS_EMPLOYED_SENTINEL = 365243
# ORGANIZATION_TYPE value for the same population; ``None`` on the API side.
ORGANIZATION_NONE = "XNA"

# API fields copied as-is (only renamed).
_DIRECT_MAPPING: dict[str, str] = {
    "gender": "CODE_GENDER",
    "contract_type": "NAME_CONTRACT_TYPE",
    "children": "CNT_CHILDREN",
    "family_members": "CNT_FAM_MEMBERS",
    "income_total": "AMT_INCOME_TOTAL",
    "credit_amount": "AMT_CREDIT",
    "annuity_amount": "AMT_ANNUITY",
    "goods_price": "AMT_GOODS_PRICE",
    "education_type": "NAME_EDUCATION_TYPE",
    "family_status": "NAME_FAMILY_STATUS",
    "housing_type": "NAME_HOUSING_TYPE",
    "occupation_type": "OCCUPATION_TYPE",
    "ext_source_1": "EXT_SOURCE_1",
    "ext_source_2": "EXT_SOURCE_2",
    "ext_source_3": "EXT_SOURCE_3",
}

HC_NUMERIC_COLUMNS: list[str] = [
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]
HC_CATEGORICAL_COLUMNS: list[str] = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_CONTRACT_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]
HC_RAW_COLUMNS: list[str] = HC_NUMERIC_COLUMNS + HC_CATEGORICAL_COLUMNS

# Same ratios as the previous project's ``_feat_eng``.
ENGINEERED_COLUMNS: list[str] = [
    "DAYS_EMPLOYED_PERC",
    "INCOME_CREDIT_PERC",
    "INCOME_PER_PERSON",
    "ANNUITY_INCOME_PERC",
    "PAYMENT_RATE",
]

FEATURE_COLUMNS: list[str] = HC_RAW_COLUMNS + ENGINEERED_COLUMNS


def clients_to_frame(clients: list[ClientData]) -> pd.DataFrame:
    """Convert validated request payloads into a raw (API-side) DataFrame."""
    rows = [c.model_dump(exclude={"client_id"}) for c in clients]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def to_home_credit_frame(df_api: pd.DataFrame) -> pd.DataFrame:
    """API-side columns -> Home Credit ``application_train.csv`` columns."""
    df = pd.DataFrame(index=df_api.index)
    for api_col, hc_col in _DIRECT_MAPPING.items():
        # None -> NaN so the sklearn imputers treat it as missing.
        df[hc_col] = df_api[api_col].where(df_api[api_col].notna(), np.nan)
    df["FLAG_OWN_CAR"] = np.where(df_api["own_car"].astype(bool), "Y", "N")
    df["FLAG_OWN_REALTY"] = np.where(df_api["own_realty"].astype(bool), "Y", "N")
    df["DAYS_BIRTH"] = -(pd.to_numeric(df_api["age_years"]) * DAYS_PER_YEAR).round()
    df["DAYS_EMPLOYED"] = -(pd.to_numeric(df_api["years_employed"]) * DAYS_PER_YEAR).round()
    df["ORGANIZATION_TYPE"] = df_api["organization_type"].where(
        df_api["organization_type"].notna(), ORGANIZATION_NONE
    )
    return df[HC_RAW_COLUMNS]


def from_home_credit_frame(df_hc: pd.DataFrame) -> pd.DataFrame:
    """Home Credit columns -> API-side columns (reference data for drift
    monitoring, replay of real applications by the traffic simulator)."""
    df = pd.DataFrame(index=df_hc.index)
    for api_col, hc_col in _DIRECT_MAPPING.items():
        df[api_col] = df_hc[hc_col]
    df["own_car"] = df_hc["FLAG_OWN_CAR"] == "Y"
    df["own_realty"] = df_hc["FLAG_OWN_REALTY"] == "Y"
    df["age_years"] = -df_hc["DAYS_BIRTH"] / DAYS_PER_YEAR
    days_employed = df_hc["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
    df["years_employed"] = (-days_employed / DAYS_PER_YEAR).clip(lower=0)
    df["organization_type"] = df_hc["ORGANIZATION_TYPE"].where(
        df_hc["ORGANIZATION_TYPE"] != ORGANIZATION_NONE, None
    )
    return df[RAW_COLUMNS]


def engineer_features(df_hc: pd.DataFrame) -> pd.DataFrame:
    """Derive the previous project's ``_feat_eng`` ratios on Home Credit
    columns (income/credit ratios, employment ratio, ...)."""
    df = df_hc.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    return df.replace([np.inf, -np.inf], np.nan)


def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full API-side raw input -> model-ready feature frame."""
    return engineer_features(to_home_credit_frame(df_raw))[FEATURE_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                HC_NUMERIC_COLUMNS + ENGINEERED_COLUMNS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                HC_CATEGORICAL_COLUMNS,
            ),
        ],
        verbose_feature_names_out=False,
    )
