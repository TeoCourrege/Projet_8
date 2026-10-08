"""The optimised inference path must give exactly the sklearn pipeline's scores
(no accuracy regression — see docs/optimisation_report.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from scoring_api import main as main_module
from scoring_api.fast_inference import build_fast_scorer
from scoring_api.features import RAW_COLUMNS, build_preprocessor, prepare_features
from scoring_api.model import load_model_bundle, predict, predict_sklearn
from scoring_api.schemas import ClientData
from scripts.generate_synthetic_data import generate_dataset


def _clients(n: int, seed: int) -> list[ClientData]:
    df = generate_dataset(n_samples=n, seed=seed)[RAW_COLUMNS]
    df = df.astype(object).where(df.notna(), None)
    clients = []
    for row in df.to_dict(orient="records"):
        try:
            clients.append(ClientData(**row))
        except ValidationError:  # rows the API itself would reject
            continue
    return clients


@pytest.fixture(scope="module")
def clients() -> list[ClientData]:
    clients = _clients(400, seed=123)
    # Edge cases: unemployed client, all optional fields missing.
    base = clients[0].model_dump()
    clients.append(ClientData(**{**base, "years_employed": None}))
    clients.append(
        ClientData(
            **{
                **base,
                "occupation_type": None,
                "organization_type": None,
                "ext_source_1": None,
                "ext_source_2": None,
                "ext_source_3": None,
            }
        )
    )
    return clients


def test_demo_model_uses_optimised_lightgbm_path(trained_model_dir: Path) -> None:
    bundle = load_model_bundle(trained_model_dir)
    assert bundle.inference_engine == "optimized-lightgbm"


def test_optimised_scores_match_sklearn(trained_model_dir: Path, clients) -> None:
    bundle = load_model_bundle(trained_model_dir)
    fast = np.array([predict(bundle, c) for c in clients])
    reference = np.array([predict_sklearn(bundle, c) for c in clients])
    np.testing.assert_allclose(fast, reference, rtol=0, atol=1e-9)
    assert ((fast >= bundle.threshold) == (reference >= bundle.threshold)).all()


def test_fast_inference_can_be_disabled(trained_model_dir: Path) -> None:
    bundle = load_model_bundle(trained_model_dir, fast_inference=False)
    assert bundle.fast_scorer is None
    assert bundle.inference_engine == "sklearn"


def test_other_models_keep_optimised_preprocessing(clients) -> None:
    df = generate_dataset(n_samples=600, seed=5)
    pipeline = Pipeline(
        [("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=500))]
    )
    pipeline.fit(prepare_features(df[RAW_COLUMNS]), df["TARGET"])

    scorer = build_fast_scorer(pipeline)
    assert scorer is not None and scorer.engine == "optimized-preprocessing"

    frame = pd.DataFrame([c.model_dump(exclude={"client_id"}) for c in clients])[RAW_COLUMNS]
    expected = pipeline.predict_proba(prepare_features(frame))[:, 1]
    actual = np.array([scorer.predict_proba(c) for c in clients])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)


def test_unsupported_pipeline_falls_back_to_sklearn() -> None:
    assert build_fast_scorer(LogisticRegression()) is None


def test_model_info_exposes_inference_engine(client: TestClient) -> None:
    body = client.get("/model/info").json()
    assert body["inference_engine"] == "optimized-lightgbm"


def test_api_without_fast_inference(
    trained_model_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid_payload: dict
) -> None:
    monkeypatch.setenv("SCORING_API_MODEL_DIR", str(trained_model_dir))
    monkeypatch.setenv("SCORING_API_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    monkeypatch.setenv("SCORING_API_FAST_INFERENCE", "false")
    main_module.get_settings.cache_clear()  # type: ignore[attr-defined]
    main_module.settings = main_module.get_settings()

    with TestClient(main_module.app) as test_client:
        assert test_client.get("/model/info").json()["inference_engine"] == "sklearn"
        assert test_client.post("/predict", json=valid_payload).status_code == 200
