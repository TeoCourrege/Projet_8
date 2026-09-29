#!/bin/sh
set -e

MODEL_DIR="${SCORING_API_MODEL_DIR:-models}"

if [ ! -f "${MODEL_DIR}/pipeline.joblib" ]; then
    echo "Aucun modele trouve dans ${MODEL_DIR} -- generation de donnees synthetiques et entrainement d'un modele de demo..."
    python scripts/generate_synthetic_data.py
    python scripts/train_model.py --model-dir "${MODEL_DIR}"
fi

exec uvicorn scoring_api.main:app --host 0.0.0.0 --port 8000
