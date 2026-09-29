from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client: TestClient) -> None:
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "test"
    assert 0 <= body["threshold"] <= 1


def test_predict_valid_payload_returns_score_and_decision(
    client: TestClient, valid_payload: dict
) -> None:
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in {"ACCEPTE", "REFUSE"}
    assert body["latency_ms"] >= 0
    assert "request_id" in body


def test_predict_missing_required_field(client: TestClient, valid_payload: dict) -> None:
    payload = dict(valid_payload)
    del payload["income_total"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_negative_income(client: TestClient, valid_payload: dict) -> None:
    payload = dict(valid_payload)
    payload["income_total"] = 0
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_negative_age(client: TestClient, valid_payload: dict) -> None:
    payload = dict(valid_payload)
    payload["age_years"] = -5
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_wrong_type(client: TestClient, valid_payload: dict) -> None:
    payload = dict(valid_payload)
    payload["income_total"] = "beaucoup d'argent"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_unknown_category(client: TestClient, valid_payload: dict) -> None:
    payload = dict(valid_payload)
    payload["gender"] = "X"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_annuity_above_credit(client: TestClient, valid_payload: dict) -> None:
    payload = dict(valid_payload)
    payload["annuity_amount"] = payload["credit_amount"] + 1000
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_accepts_missing_optional_ext_sources(
    client: TestClient, valid_payload: dict
) -> None:
    payload = dict(valid_payload)
    payload["ext_source_1"] = None
    payload["ext_source_2"] = None
    payload["ext_source_3"] = None
    payload["occupation_type"] = None
    payload["organization_type"] = None
    payload["years_employed"] = None  # retraité / sans emploi
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_logs_prediction(client: TestClient, valid_payload: dict, tmp_path) -> None:
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200

    from scoring_api.main import settings as live_settings

    assert live_settings.log_path.exists()
    lines = live_settings.log_path.read_text().splitlines()
    assert len(lines) == 1
