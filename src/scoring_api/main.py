from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from scoring_api.config import get_settings
from scoring_api.logging_utils import log_prediction
from scoring_api.model import ModelBundle, load_model_bundle, predict
from scoring_api.schemas import (
    ClientData,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model is loaded exactly once at startup and reused for every request —
    # loading it per-request would tank latency and memory under load.
    try:
        app.state.model_bundle = load_model_bundle(settings.model_dir)
        app.state.model_load_error = None
    except FileNotFoundError as exc:
        app.state.model_bundle = None
        app.state.model_load_error = str(exc)
    yield


app = FastAPI(
    title="Prêt à Dépenser — API de scoring crédit",
    description=(
        "Reçoit les données d'une demande de crédit et retourne un score de "
        "risque de défaut ainsi qu'une décision (ACCEPTE / REFUSE)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_bundle(request: Request) -> ModelBundle:
    bundle: ModelBundle | None = request.app.state.model_bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.model_load_error or "Modèle non chargé",
        )
    return bundle


@app.get("/", tags=["meta"])
def root() -> dict:
    return {"service": "scoring-api", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    bundle: ModelBundle | None = request.app.state.model_bundle
    return HealthResponse(
        status="ok" if bundle else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.version if bundle else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info(request: Request) -> ModelInfoResponse:
    bundle = _get_bundle(request)
    return ModelInfoResponse(
        model_version=bundle.version,
        trained_at=bundle.trained_at,
        threshold=bundle.threshold,
        metrics=bundle.metrics,
        n_features=bundle.n_features,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict_endpoint(client: ClientData, request: Request) -> PredictionResponse:
    bundle = _get_bundle(request)

    start = time.perf_counter()
    try:
        score = predict(bundle, client)
    except Exception as exc:  # malformed feature computation on valid-but-odd input
        raise HTTPException(
            status_code=422, detail=f"Erreur lors du calcul des features : {exc}"
        ) from exc
    latency_ms = (time.perf_counter() - start) * 1000

    decision = "REFUSE" if score >= bundle.threshold else "ACCEPTE"
    request_id = str(uuid.uuid4())

    response = PredictionResponse(
        request_id=request_id,
        score=round(score, 6),
        decision=decision,
        threshold=bundle.threshold,
        model_version=bundle.version,
        latency_ms=round(latency_ms, 3),
        timestamp=datetime.now(UTC).isoformat(),
    )

    log_prediction(
        settings.log_path,
        {
            "request_id": request_id,
            "input": client.model_dump(),
            "score": response.score,
            "decision": decision,
            "latency_ms": response.latency_ms,
            "model_version": bundle.version,
        },
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur."})
