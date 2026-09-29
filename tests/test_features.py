from __future__ import annotations

import pandas as pd
import pytest

from scoring_api.features import (
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    engineer_features,
    from_home_credit_frame,
    prepare_features,
    to_home_credit_frame,
)
from scoring_api.schemas import ClientData


def _client_kwargs(**overrides) -> dict:
    base = {
        "gender": "M",
        "own_car": False,
        "own_realty": True,
        "contract_type": "Cash loans",
        "children": 0,
        "family_members": 1,
        "income_total": 50000.0,
        "credit_amount": 100000.0,
        "annuity_amount": 8000.0,
        "goods_price": 95000.0,
        "age_years": 40.0,
        "years_employed": 5.0,
        "education_type": "Secondary / secondary special",
        "family_status": "Single / not married",
        "housing_type": "House / apartment",
        "occupation_type": None,
        "organization_type": None,
        "ext_source_1": None,
        "ext_source_2": None,
        "ext_source_3": None,
    }
    base.update(overrides)
    return base


def test_client_data_valid() -> None:
    client = ClientData(**_client_kwargs())
    assert client.income_total == 50000.0


def test_client_data_rejects_negative_age() -> None:
    with pytest.raises(ValueError):
        ClientData(**_client_kwargs(age_years=-5))


def test_client_data_rejects_zero_income() -> None:
    with pytest.raises(ValueError):
        ClientData(**_client_kwargs(income_total=0))


def test_client_data_accepts_missing_years_employed() -> None:
    client = ClientData(**_client_kwargs(years_employed=None))
    assert client.years_employed is None


def test_engineer_features_computes_ratios() -> None:
    df = to_home_credit_frame(pd.DataFrame([_client_kwargs()])[RAW_COLUMNS])
    engineered = engineer_features(df)

    assert engineered.loc[0, "PAYMENT_RATE"] == pytest.approx(8000.0 / 100000.0)
    assert engineered.loc[0, "INCOME_CREDIT_PERC"] == pytest.approx(50000.0 / 100000.0)
    assert engineered.loc[0, "INCOME_PER_PERSON"] == pytest.approx(50000.0 / 1)
    assert engineered.loc[0, "DAYS_EMPLOYED_PERC"] == pytest.approx(5.0 / 40.0, rel=1e-3)


def test_engineer_features_handles_division_edge_cases() -> None:
    df = to_home_credit_frame(pd.DataFrame([_client_kwargs(family_members=1)])[RAW_COLUMNS])
    df.loc[0, "AMT_CREDIT"] = 0  # simulate an edge case bypassing API validation
    engineered = engineer_features(df)
    # division by zero must become NaN, never raise or produce inf
    assert pd.isna(engineered.loc[0, "INCOME_CREDIT_PERC"])
    assert pd.isna(engineered.loc[0, "PAYMENT_RATE"])


def test_to_home_credit_frame_maps_fields() -> None:
    df = to_home_credit_frame(
        pd.DataFrame([_client_kwargs(own_car=True, years_employed=None)])[RAW_COLUMNS]
    )
    row = df.iloc[0]
    assert row["CODE_GENDER"] == "M"
    assert row["FLAG_OWN_CAR"] == "Y"
    assert row["FLAG_OWN_REALTY"] == "Y"
    assert row["DAYS_BIRTH"] == pytest.approx(-40 * 365.25, abs=1)
    assert pd.isna(row["DAYS_EMPLOYED"])
    assert row["ORGANIZATION_TYPE"] == "XNA"  # organization_type=None
    assert pd.isna(row["OCCUPATION_TYPE"])


def test_home_credit_round_trip() -> None:
    hc = pd.DataFrame(
        [
            {
                "CNT_CHILDREN": 1, "CNT_FAM_MEMBERS": 3.0, "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5, "AMT_ANNUITY": 24700.5, "AMT_GOODS_PRICE": 351000.0,
                "DAYS_BIRTH": -9461, "DAYS_EMPLOYED": 365243, "EXT_SOURCE_1": 0.083,
                "EXT_SOURCE_2": 0.263, "EXT_SOURCE_3": None, "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "N", "FLAG_OWN_REALTY": "Y", "NAME_CONTRACT_TYPE": "Cash loans",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Single / not married",
                "NAME_HOUSING_TYPE": "House / apartment", "OCCUPATION_TYPE": "Laborers",
                "ORGANIZATION_TYPE": "XNA",
            }
        ]
    )  # fmt: skip
    api = from_home_credit_frame(hc)
    assert api.loc[0, "age_years"] == pytest.approx(9461 / 365.25)
    assert pd.isna(api.loc[0, "years_employed"])  # 365243 sentinel = not employed
    assert api.loc[0, "organization_type"] is None or pd.isna(api.loc[0, "organization_type"])

    back = to_home_credit_frame(api)
    assert back.loc[0, "DAYS_BIRTH"] == -9461
    assert back.loc[0, "ORGANIZATION_TYPE"] == "XNA"
    assert back.loc[0, "FLAG_OWN_REALTY"] == "Y"


def test_prepare_features_returns_model_columns() -> None:
    features = prepare_features(pd.DataFrame([_client_kwargs()])[RAW_COLUMNS])
    assert list(features.columns) == FEATURE_COLUMNS
