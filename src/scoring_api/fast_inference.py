"""Optimised single-request inference path (see docs/optimisation_report.md).

Profiling ``/predict`` showed that ~95% of the inference time was framework
overhead, not the model: building a one-row pandas DataFrame column by column
(``features.to_home_credit_frame``), then the sklearn ``ColumnTransformer``
(imputers, scaler, one-hot encoder) re-validating that single row.

``FastScorer`` "compiles" the fitted pipeline once, at API startup, into plain
NumPy arrays (imputation values, scaler mean/scale, one-hot positions) and
scores a request with:

1. the features computed in pure Python from the validated ``ClientData``,
2. the preprocessing applied with NumPy,
3. for a ``CalibratedClassifierCV`` over LightGBM, a direct call to each
   LightGBM booster (single thread) followed by its isotonic calibrator;
   any other model is called through its own ``predict_proba``.

The result is numerically identical to ``pipeline.predict_proba`` — this is
checked at startup by ``build_fast_scorer`` (and in ``tests/``), which falls
back to the regular sklearn path if the pipeline is not supported or if the
two paths disagree.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scoring_api.features import (
    DAYS_PER_YEAR,
    ORGANIZATION_NONE,
    clients_to_frame,
    prepare_features,
)
from scoring_api.schemas import ClientData

logger = logging.getLogger(__name__)

# Max |fast - sklearn| accepted by the startup self-check.
SELF_CHECK_TOLERANCE = 1e-9


class UnsupportedPipelineError(ValueError):
    """The fitted pipeline does not have the structure FastScorer compiles."""


def _ratio(numerator: float, denominator: float) -> float:
    # Same semantics as features.engineer_features: x/0 -> inf -> NaN.
    if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator


def _optional(value: float | None) -> float:
    return math.nan if value is None else float(value)


def client_to_features(client: ClientData) -> tuple[dict[str, float], dict[str, str | None]]:
    """Pure-Python equivalent of ``features.prepare_features`` for one client:
    returns the numeric and categorical Home Credit features by column name."""
    days_birth = -float(round(client.age_years * DAYS_PER_YEAR))
    days_employed = (
        math.nan
        if client.years_employed is None
        else -float(round(client.years_employed * DAYS_PER_YEAR))
    )
    income = float(client.income_total)
    credit = float(client.credit_amount)
    annuity = float(client.annuity_amount)
    family = float(client.family_members)

    numeric = {
        "CNT_CHILDREN": float(client.children),
        "CNT_FAM_MEMBERS": family,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": float(client.goods_price),
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "EXT_SOURCE_1": _optional(client.ext_source_1),
        "EXT_SOURCE_2": _optional(client.ext_source_2),
        "EXT_SOURCE_3": _optional(client.ext_source_3),
        "DAYS_EMPLOYED_PERC": _ratio(days_employed, days_birth),
        "INCOME_CREDIT_PERC": _ratio(income, credit),
        "INCOME_PER_PERSON": _ratio(income, family),
        "ANNUITY_INCOME_PERC": _ratio(annuity, income),
        "PAYMENT_RATE": _ratio(annuity, credit),
    }
    categorical = {
        "CODE_GENDER": client.gender,
        "FLAG_OWN_CAR": "Y" if client.own_car else "N",
        "FLAG_OWN_REALTY": "Y" if client.own_realty else "N",
        "NAME_CONTRACT_TYPE": client.contract_type,
        "NAME_EDUCATION_TYPE": client.education_type,
        "NAME_FAMILY_STATUS": client.family_status,
        "NAME_HOUSING_TYPE": client.housing_type,
        "OCCUPATION_TYPE": client.occupation_type,
        "ORGANIZATION_TYPE": client.organization_type or ORGANIZATION_NONE,
    }
    return numeric, categorical


@dataclass
class _CalibratedBooster:
    booster: object  # lightgbm.Booster
    x_thresholds: np.ndarray
    y_thresholds: np.ndarray


@dataclass
class FastScorer:
    numeric_columns: list[str]
    numeric_fill: np.ndarray
    numeric_mean: np.ndarray
    numeric_scale: np.ndarray
    categorical_columns: list[str]
    categorical_fill: list[str]
    # One dict per categorical column: category -> output position.
    categorical_positions: list[dict[str, int]]
    n_outputs: int
    model: object
    calibrated_boosters: list[_CalibratedBooster] = field(default_factory=list)

    @property
    def engine(self) -> str:
        return "optimized-lightgbm" if self.calibrated_boosters else "optimized-preprocessing"

    def transform(self, client: ClientData) -> np.ndarray:
        numeric, categorical = client_to_features(client)
        row = np.zeros((1, self.n_outputs))

        values = np.array([numeric[c] for c in self.numeric_columns], dtype=float)
        values = np.where(np.isnan(values), self.numeric_fill, values)
        row[0, : len(values)] = (values - self.numeric_mean) / self.numeric_scale

        for col, fill, positions in zip(
            self.categorical_columns, self.categorical_fill, self.categorical_positions
        ):
            value = categorical[col]
            position = positions.get(fill if value is None else value)
            if position is not None:  # unknown category -> all zeros (handle_unknown="ignore")
                row[0, position] = 1.0
        return row

    def predict_proba(self, client: ClientData) -> float:
        row = self.transform(client)
        if not self.calibrated_boosters:
            return float(self.model.predict_proba(row)[0, 1])
        total = 0.0
        for cb in self.calibrated_boosters:
            raw = cb.booster.predict(row, raw_score=True, num_threads=1)
            total += float(np.interp(raw, cb.x_thresholds, cb.y_thresholds)[0])
        return total / len(self.calibrated_boosters)


def _compile_preprocessor(preprocessor: ColumnTransformer) -> dict:
    if not isinstance(preprocessor, ColumnTransformer) or preprocessor.remainder != "drop":
        raise UnsupportedPipelineError("preprocessing is not the expected ColumnTransformer")
    transformers = {
        name: (steps, list(columns))
        for name, steps, columns in preprocessor.transformers_
        if name != "remainder"
    }
    if [name for name, _, _ in preprocessor.transformers_ if name != "remainder"] != [
        "num",
        "cat",
    ]:
        raise UnsupportedPipelineError("expected exactly the 'num' then 'cat' transformers")

    num, numeric_columns = transformers["num"]
    cat, categorical_columns = transformers["cat"]
    if not (
        isinstance(num, Pipeline)
        and [type(s) for _, s in num.steps] == [SimpleImputer, StandardScaler]
        and isinstance(cat, Pipeline)
        and [type(s) for _, s in cat.steps] == [SimpleImputer, OneHotEncoder]
    ):
        raise UnsupportedPipelineError("unexpected numeric/categorical sub-pipelines")

    imputer, scaler = num.steps[0][1], num.steps[1][1]
    cat_imputer, onehot = cat.steps[0][1], cat.steps[1][1]
    if onehot.drop is not None or onehot.handle_unknown != "ignore" or getattr(
        onehot, "infrequent_categories_", None
    ):
        raise UnsupportedPipelineError("unsupported OneHotEncoder options")

    mean = scaler.mean_ if scaler.with_mean else np.zeros(len(numeric_columns))
    scale = scaler.scale_ if scaler.with_std else np.ones(len(numeric_columns))

    positions: list[dict[str, int]] = []
    offset = len(numeric_columns)
    for categories in onehot.categories_:
        positions.append({str(v): offset + i for i, v in enumerate(categories)})
        offset += len(categories)

    return {
        "numeric_columns": numeric_columns,
        "numeric_fill": np.asarray(imputer.statistics_, dtype=float),
        "numeric_mean": np.asarray(mean, dtype=float),
        "numeric_scale": np.asarray(scale, dtype=float),
        "categorical_columns": categorical_columns,
        "categorical_fill": [str(v) for v in cat_imputer.statistics_],
        "categorical_positions": positions,
        "n_outputs": offset,
    }


def _compile_calibrated_lightgbm(model: object) -> list[_CalibratedBooster]:
    """Direct booster calls for CalibratedClassifierCV(LGBMClassifier, isotonic);
    empty list (generic ``predict_proba``) for any other model."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:  # pragma: no cover - lightgbm is a core dependency
        return []
    if not isinstance(model, CalibratedClassifierCV) or model.method != "isotonic":
        return []
    compiled = []
    for cc in model.calibrated_classifiers_:
        estimator = cc.estimator
        calibrator = cc.calibrators[0] if len(cc.calibrators) == 1 else None
        if (
            not isinstance(estimator, LGBMClassifier)
            or not isinstance(calibrator, IsotonicRegression)
            or calibrator.out_of_bounds != "clip"
        ):
            return []
        # sklearn calibrates on decision_function (LightGBM raw score) when the
        # estimator exposes it, on predict_proba otherwise: mirror that choice.
        if not hasattr(estimator, "decision_function"):
            return []
        compiled.append(
            _CalibratedBooster(
                booster=estimator.booster_,
                x_thresholds=np.asarray(calibrator.X_thresholds_, dtype=float),
                y_thresholds=np.asarray(calibrator.y_thresholds_, dtype=float),
            )
        )
    return compiled


def compile_pipeline(pipeline: object) -> FastScorer:
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["preprocess", "model"]:
        raise UnsupportedPipelineError("expected Pipeline([('preprocess', ...), ('model', ...)])")
    model = pipeline.named_steps["model"]
    return FastScorer(
        **_compile_preprocessor(pipeline.named_steps["preprocess"]),
        model=model,
        calibrated_boosters=_compile_calibrated_lightgbm(model),
    )


# Representative requests for the startup self-check: missing optional
# fields, unemployed client (years_employed=None), unknown-to-imputer values.
_SELF_CHECK_CLIENTS: list[dict] = [
    {
        "gender": "F", "own_car": True, "own_realty": True, "contract_type": "Cash loans",
        "children": 1, "family_members": 3, "income_total": 65000, "credit_amount": 180000,
        "annuity_amount": 12000, "goods_price": 175000, "age_years": 35, "years_employed": 6,
        "education_type": "Higher education", "family_status": "Married",
        "housing_type": "House / apartment", "occupation_type": "Core staff",
        "organization_type": "Business Entity Type 3",
        "ext_source_1": 0.6, "ext_source_2": 0.55, "ext_source_3": 0.5,
    },
    {
        "gender": "M", "own_car": False, "own_realty": False,
        "contract_type": "Revolving loans", "children": 0, "family_members": 1,
        "income_total": 202500, "credit_amount": 406597.5, "annuity_amount": 24700.5,
        "goods_price": 351000, "age_years": 64.2, "years_employed": None,
        "education_type": "Secondary / secondary special",
        "family_status": "Single / not married", "housing_type": "With parents",
    },
    {
        "gender": "F", "own_car": False, "own_realty": True, "contract_type": "Cash loans",
        "children": 3, "family_members": 5, "income_total": 31500, "credit_amount": 1350000,
        "annuity_amount": 39604.5, "goods_price": 1350000, "age_years": 23.5,
        "years_employed": 0.5, "education_type": "Academic degree", "family_status": "Widow",
        "housing_type": "Co-op apartment", "occupation_type": "IT staff",
        "organization_type": "Industry: type 13", "ext_source_2": 0.01, "ext_source_3": 0.99,
    },
]  # fmt: skip


def build_fast_scorer(pipeline: object) -> FastScorer | None:
    """Compile ``pipeline`` and verify it against the sklearn path; ``None``
    (-> regular sklearn inference) if unsupported or not identical."""
    try:
        scorer = compile_pipeline(pipeline)
        clients = [ClientData(**c) for c in _SELF_CHECK_CLIENTS]
        expected = pipeline.predict_proba(prepare_features(clients_to_frame(clients)))[:, 1]
        actual = np.array([scorer.predict_proba(c) for c in clients])
        max_diff = float(np.max(np.abs(expected - actual)))
    except Exception as exc:  # noqa: BLE001 - never prevent the API from starting
        logger.warning("Inférence optimisée désactivée (%s) — chemin sklearn utilisé.", exc)
        return None
    if max_diff > SELF_CHECK_TOLERANCE:
        logger.warning(
            "Inférence optimisée désactivée : écart %.2e avec le pipeline sklearn.", max_diff
        )
        return None
    return scorer
