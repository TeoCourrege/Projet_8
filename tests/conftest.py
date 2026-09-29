from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scoring_api import main as main_module
from scoring_api.features import RAW_COLUMNS, prepare_features
from scripts.generate_synthetic_data import generate_dataset
from scripts.train_model import build_pipeline, compute_cost_curve, find_best_threshold


@pytest.fixture()
def valid_payload() -> dict:
    return {
        "gender": "F",
        "own_car": True,
        "own_realty": True,
        "contract_type": "Cash loans",
        "children": 1,
        "family_members": 3,
        "income_total": 65000.0,
        "credit_amount": 180000.0,
        "annuity_amount": 12000.0,
        "goods_price": 175000.0,
        "age_years": 35.0,
        "years_employed": 6.0,
        "education_type": "Higher education",
        "family_status": "Married",
        "housing_type": "House / apartment",
        "occupation_type": "Core staff",
        "organization_type": "Business Entity Type 3",
        "ext_source_1": 0.6,
        "ext_source_2": 0.55,
        "ext_source_3": 0.5,
    }


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a small demo model once per test session — keeps tests fast
    and fully self-contained (no need to run the CLI scripts first)."""
    model_dir = tmp_path_factory.mktemp("model")

    df = generate_dataset(n_samples=1200, seed=7)
    X = prepare_features(df[RAW_COLUMNS])
    y = df["TARGET"]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)[:, 1]
    thresholds, costs = compute_cost_curve(y.values, proba)
    threshold, _ = find_best_threshold(thresholds, costs)

    import joblib

    joblib.dump(pipeline, model_dir / "pipeline.joblib")
    metadata = {
        "model_version": "test",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "threshold": threshold,
        "metrics": {"roc_auc_holdout": 0.75},
        "n_features": X.shape[1],
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata))
    return model_dir


@pytest.fixture()
def client(trained_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SCORING_API_MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("SCORING_API_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    main_module.get_settings.cache_clear()  # type: ignore[attr-defined]
    main_module.settings = main_module.get_settings()

    with TestClient(main_module.app) as test_client:
        yield test_client
