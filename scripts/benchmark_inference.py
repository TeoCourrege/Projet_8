"""Benchmark & profile the inference: sklearn baseline vs optimised path.

Produces the measurements behind docs/optimisation_report.md:

- accuracy check: optimised scores vs sklearn scores on the reference sample
  (max difference, decision mismatches, ROC AUC of both),
- inference latency (p50/p95/p99) and per-stage breakdown for both paths,
- cProfile of both paths (top functions by cumulative time),
- end-to-end ``POST /predict`` response time through the FastAPI app,
- optionally (``--onnx``) an ONNX Runtime conversion of the model, which needs
  ``uv run --with skl2onnx --with onnxmltools --with onnxruntime``.

Usage:
    uv run python scripts/benchmark_inference.py
    uv run --with skl2onnx --with onnxmltools --with onnxruntime \\
        python scripts/benchmark_inference.py --onnx
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import re
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.metrics import roc_auc_score

from scoring_api.features import RAW_COLUMNS, clients_to_frame, prepare_features
from scoring_api.model import ModelBundle, load_model_bundle, predict, predict_sklearn
from scoring_api.schemas import ClientData

WARMUP = 50


def load_clients(reference: Path) -> tuple[list[ClientData], np.ndarray]:
    df = pd.read_csv(reference)
    target = df["TARGET"].to_numpy() if "TARGET" in df else None
    raw = df[RAW_COLUMNS].astype(object).where(df[RAW_COLUMNS].notna(), None)
    clients, kept = [], []
    for i, row in enumerate(raw.to_dict(orient="records")):
        try:
            clients.append(ClientData(**row))
            kept.append(i)
        except ValidationError:
            continue
    return clients, (target[kept] if target is not None else None)


def latency_stats(fn: Callable[[ClientData], object], clients: list[ClientData]) -> dict:
    for c in clients[:WARMUP]:
        fn(c)
    timings = []
    for c in clients:
        start = time.perf_counter()
        fn(c)
        timings.append((time.perf_counter() - start) * 1000)
    arr = np.array(timings)
    return {
        "n": len(arr),
        "mean_ms": round(float(arr.mean()), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
    }


def profile(fn: Callable[[ClientData], object], clients: list[ClientData], path: Path) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for c in clients:
        fn(c)
    profiler.disable()
    out = io.StringIO()
    pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(25)
    # Strip machine-specific absolute paths (user name, virtualenv location).
    text = re.sub(r"\S*/site-packages/", "", out.getvalue())
    text = re.sub(r"\S*/lib/python3\.\d+/", "", text)
    path.write_text(text.replace(str(Path.cwd()) + "/", ""))


def stage_breakdown(baseline: ModelBundle, optimised: ModelBundle, clients) -> dict:
    pipeline = baseline.pipeline
    preprocess, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    frames = [prepare_features(clients_to_frame([c])) for c in clients]
    rows = [preprocess.transform(f) for f in frames]
    scorer = optimised.fast_scorer
    stages = {
        "baseline": {
            "features_pandas": latency_stats(
                lambda c: prepare_features(clients_to_frame([c])), clients
            )["p50_ms"],
            "preprocess_sklearn": _p50_over(preprocess.transform, frames),
            "model_predict_proba": _p50_over(model.predict_proba, rows),
        }
    }
    if scorer is not None:
        compiled = [scorer.transform(c) for c in clients]
        stages["optimised"] = {
            "features_and_preprocess_numpy": latency_stats(scorer.transform, clients)["p50_ms"],
            "model": _p50_over(
                lambda r: _score_row(scorer, r), compiled
            ),
        }
    return stages


def _p50_over(fn: Callable, items: list) -> float:
    for item in items[:WARMUP]:
        fn(item)
    timings = []
    for item in items:
        start = time.perf_counter()
        fn(item)
        timings.append((time.perf_counter() - start) * 1000)
    return round(float(np.percentile(timings, 50)), 4)


def _score_row(scorer, row: np.ndarray) -> float:
    if not scorer.calibrated_boosters:
        return float(scorer.model.predict_proba(row)[0, 1])
    total = 0.0
    for cb in scorer.calibrated_boosters:
        raw = cb.booster.predict(row, raw_score=True, num_threads=1)
        total += float(np.interp(raw, cb.x_thresholds, cb.y_thresholds)[0])
    return total / len(scorer.calibrated_boosters)


def accuracy_check(baseline: ModelBundle, optimised: ModelBundle, clients, target) -> dict:
    frame = clients_to_frame(clients)
    reference = baseline.pipeline.predict_proba(prepare_features(frame))[:, 1]
    fast = np.array([predict(optimised, c) for c in clients])
    threshold = baseline.threshold
    result = {
        "n_clients": len(clients),
        "max_abs_score_diff": float(np.max(np.abs(reference - fast))),
        "decision_mismatches": int(((reference >= threshold) != (fast >= threshold)).sum()),
        "refusal_rate_baseline": round(float((reference >= threshold).mean()), 5),
        "refusal_rate_optimised": round(float((fast >= threshold).mean()), 5),
    }
    if target is not None and len(np.unique(target)) == 2:
        result["roc_auc_baseline"] = round(float(roc_auc_score(target, reference)), 6)
        result["roc_auc_optimised"] = round(float(roc_auc_score(target, fast)), 6)
    return result


def api_response_time(model_dir: Path, fast: bool, clients: list[ClientData]) -> dict:
    from fastapi.testclient import TestClient

    from scoring_api import main as main_module

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["SCORING_API_MODEL_DIR"] = str(model_dir)
        os.environ["SCORING_API_LOG_PATH"] = str(Path(tmp) / "predictions.jsonl")
        os.environ["SCORING_API_FAST_INFERENCE"] = "true" if fast else "false"
        main_module.get_settings.cache_clear()
        main_module.settings = main_module.get_settings()
        payloads = [c.model_dump(mode="json") for c in clients]
        with TestClient(main_module.app) as client:
            payload_iter = iter(payloads * 2)
            return latency_stats(
                lambda _: client.post("/predict", json=next(payload_iter)), clients
            )


def onnx_benchmark(bundle: ModelBundle, clients: list[ClientData]) -> dict:
    """Convert the model step to ONNX and compare it with sklearn (model step only)."""
    import onnxruntime as ort
    from lightgbm import LGBMClassifier
    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

    update_registered_converter(
        LGBMClassifier,
        "LightGbmLGBMClassifier",
        calculate_linear_classifier_output_shapes,
        convert_lightgbm,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )
    preprocess = bundle.pipeline.named_steps["preprocess"]
    model = bundle.pipeline.named_steps["model"]
    rows = preprocess.transform(prepare_features(clients_to_frame(clients)))
    onx = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, rows.shape[1]]))],
        options={id(model): {"zipmap": False}},
        target_opset={"": 17, "ai.onnx.ml": 3},
    )
    session = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    rows32 = rows.astype(np.float32)
    onnx_scores = session.run(None, {"input": rows32})[1][:, 1]
    reference = model.predict_proba(rows)[:, 1]
    threshold = bundle.threshold
    single = [rows32[i : i + 1] for i in range(len(rows32))]
    return {
        "max_abs_score_diff": float(np.max(np.abs(onnx_scores - reference))),
        "decision_mismatches": int(((onnx_scores >= threshold) != (reference >= threshold)).sum()),
        "model_step_p50_ms": _p50_over(lambda r: session.run(None, {"input": r}), single),
        "onnxruntime_version": version("onnxruntime"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--reference-data", type=Path, default=Path("data/processed/reference_sample.csv")
    )
    parser.add_argument("--n-requests", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/optimisation"))
    parser.add_argument("--skip-api", action="store_true", help="Sans mesure de bout en bout")
    parser.add_argument("--onnx", action="store_true", help="Teste aussi ONNX Runtime")
    args = parser.parse_args()

    baseline = load_model_bundle(args.model_dir, fast_inference=False)
    optimised = load_model_bundle(args.model_dir, fast_inference=True)
    clients, target = load_clients(args.reference_data)
    bench_clients = clients[: args.n_requests]
    print(f"Modèle {baseline.version} — moteur optimisé : {optimised.inference_engine}")
    print(f"{len(clients)} clients de référence, {len(bench_clients)} requêtes chronométrées")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": baseline.version,
        "inference_engine": optimised.inference_engine,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "scikit-learn": version("scikit-learn"),
            "lightgbm": version("lightgbm"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
        },
        "accuracy": accuracy_check(baseline, optimised, clients, target),
        "inference_latency": {
            "baseline_sklearn": latency_stats(lambda c: predict_sklearn(baseline, c), bench_clients),
            "optimised": latency_stats(lambda c: predict(optimised, c), bench_clients),
        },
        "stages_p50_ms": stage_breakdown(baseline, optimised, bench_clients),
    }
    lat = results["inference_latency"]
    lat["speedup_p50"] = round(lat["baseline_sklearn"]["p50_ms"] / lat["optimised"]["p50_ms"], 1)

    profile_clients = bench_clients[:300]
    profile(lambda c: predict_sklearn(baseline, c), profile_clients,
            args.output_dir / "profile_baseline.txt")  # fmt: skip
    profile(lambda c: predict(optimised, c), profile_clients,
            args.output_dir / "profile_optimised.txt")  # fmt: skip

    if not args.skip_api:
        api_clients = bench_clients[:500]
        results["api_response_time"] = {
            "baseline_sklearn": api_response_time(args.model_dir, False, api_clients),
            "optimised": api_response_time(args.model_dir, True, api_clients),
        }
    if args.onnx:
        try:
            results["onnx_runtime"] = onnx_benchmark(baseline, bench_clients)
        except Exception as exc:  # noqa: BLE001 - optional exploratory step
            results["onnx_runtime"] = {"error": f"{type(exc).__name__}: {exc}"}

    out = args.output_dir / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nRésultats -> {out}")


if __name__ == "__main__":
    main()
