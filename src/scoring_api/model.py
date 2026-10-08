from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from scoring_api.fast_inference import FastScorer, build_fast_scorer
from scoring_api.features import clients_to_frame, prepare_features
from scoring_api.schemas import ClientData

SKLEARN_ENGINE = "sklearn"


@dataclass
class ModelBundle:
    pipeline: object
    version: str
    trained_at: str
    threshold: float
    metrics: dict
    n_features: int
    fast_scorer: FastScorer | None = None

    @property
    def inference_engine(self) -> str:
        return self.fast_scorer.engine if self.fast_scorer else SKLEARN_ENGINE


def load_model_bundle(model_dir: Path, fast_inference: bool = True) -> ModelBundle:
    """Load the trained pipeline once (called at API startup, never per
    request — see README "Chargement du modèle").
    """
    pipeline_path = model_dir / "pipeline.joblib"
    metadata_path = model_dir / "metadata.json"
    if not pipeline_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable dans '{model_dir}'. Exportez le modèle MLflow : "
            "'uv run --extra training python scripts/export_mlflow_model.py' "
            "(ou, modèle de démo : scripts/generate_synthetic_data.py puis "
            "scripts/train_model.py)."
        )
    pipeline = joblib.load(pipeline_path)
    metadata = json.loads(metadata_path.read_text())
    return ModelBundle(
        pipeline=pipeline,
        version=metadata["model_version"],
        trained_at=metadata["trained_at"],
        threshold=metadata["threshold"],
        metrics=metadata.get("metrics", {}),
        n_features=metadata.get("n_features", 0),
        # Compiled once here too (see fast_inference.py / docs/optimisation_report.md).
        fast_scorer=build_fast_scorer(pipeline) if fast_inference else None,
    )


def predict(bundle: ModelBundle, client: ClientData) -> float:
    if bundle.fast_scorer is not None:
        return bundle.fast_scorer.predict_proba(client)
    return predict_sklearn(bundle, client)


def predict_sklearn(bundle: ModelBundle, client: ClientData) -> float:
    """Reference (unoptimised) path: pandas features + full sklearn pipeline."""
    df_raw = clients_to_frame([client])
    df_features = prepare_features(df_raw)
    proba = bundle.pipeline.predict_proba(df_features)[:, 1]
    return float(proba[0])
