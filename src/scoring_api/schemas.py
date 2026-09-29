from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Domain constants — single source of truth, reused by the data-generation
# and traffic-simulation scripts so their sampling stays in sync with the
# API's validation rules.
# ---------------------------------------------------------------------------

# Category values are the ones found in the Home Credit ``application_train.csv``
# the legacy MLflow models are trained on (see ``features.to_home_credit_frame``),
# so every value accepted here is one the model has seen during training.

GENDERS: tuple[str, ...] = ("M", "F")
Gender = Literal[GENDERS]

CONTRACT_TYPES: tuple[str, ...] = ("Cash loans", "Revolving loans")
ContractType = Literal[CONTRACT_TYPES]

EDUCATION_TYPES: tuple[str, ...] = (
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
)
EducationType = Literal[EDUCATION_TYPES]

FAMILY_STATUSES: tuple[str, ...] = (
    "Married",
    "Single / not married",
    "Civil marriage",
    "Separated",
    "Widow",
)
FamilyStatus = Literal[FAMILY_STATUSES]

HOUSING_TYPES: tuple[str, ...] = (
    "House / apartment",
    "With parents",
    "Municipal apartment",
    "Rented apartment",
    "Office apartment",
    "Co-op apartment",
)
HousingType = Literal[HOUSING_TYPES]

OCCUPATION_TYPES: tuple[str, ...] = (
    "Laborers",
    "Sales staff",
    "Core staff",
    "Managers",
    "Drivers",
    "High skill tech staff",
    "Accountants",
    "Medicine staff",
    "Security staff",
    "Cooking staff",
    "Cleaning staff",
    "Private service staff",
    "Low-skill Laborers",
    "Waiters/barmen staff",
    "Secretaries",
    "Realty agents",
    "HR staff",
    "IT staff",
)
OccupationType = Literal[OCCUPATION_TYPES]

# "XNA" in the raw data (pensioners / unemployed) is represented by
# ``organization_type=None`` in the API.
ORGANIZATION_TYPES: tuple[str, ...] = (
    "Advertising", "Agriculture", "Bank",
    "Business Entity Type 1", "Business Entity Type 2", "Business Entity Type 3",
    "Cleaning", "Construction", "Culture", "Electricity", "Emergency", "Government",
    "Hotel", "Housing",
    "Industry: type 1", "Industry: type 2", "Industry: type 3", "Industry: type 4",
    "Industry: type 5", "Industry: type 6", "Industry: type 7", "Industry: type 8",
    "Industry: type 9", "Industry: type 10", "Industry: type 11", "Industry: type 12",
    "Industry: type 13",
    "Insurance", "Kindergarten", "Legal Services", "Medicine", "Military", "Mobile",
    "Other", "Police", "Postal", "Realtor", "Religion", "Restaurant", "School",
    "Security", "Security Ministries", "Self-employed", "Services", "Telecom",
    "Trade: type 1", "Trade: type 2", "Trade: type 3", "Trade: type 4",
    "Trade: type 5", "Trade: type 6", "Trade: type 7",
    "Transport: type 1", "Transport: type 2", "Transport: type 3", "Transport: type 4",
    "University",
)  # fmt: skip
OrganizationType = Literal[ORGANIZATION_TYPES]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ClientData(BaseModel):
    """Raw application data for a single credit request.

    Field constraints double as the API's first line of defence against
    malformed input (missing required fields, out-of-range values, wrong
    types) — FastAPI turns any violation into an HTTP 422 automatically.
    """

    model_config = {"extra": "forbid"}

    client_id: int | None = Field(default=None, description="Identifiant client (optionnel)")

    gender: Gender
    own_car: bool
    own_realty: bool
    contract_type: ContractType = "Cash loans"

    children: int = Field(ge=0, le=20, description="Nombre d'enfants à charge")
    family_members: int = Field(ge=1, le=25, description="Taille du foyer")

    income_total: float = Field(gt=0, le=200_000_000, description="Revenu annuel total")
    credit_amount: float = Field(gt=0, le=10_000_000, description="Montant du crédit demandé")
    annuity_amount: float = Field(gt=0, le=1_000_000, description="Montant de l'annuité")
    goods_price: float = Field(gt=0, le=10_000_000, description="Prix du bien financé")

    age_years: float = Field(gt=17, le=100, description="Âge du client en années")
    years_employed: float | None = Field(
        default=None,
        ge=0,
        le=60,
        description="Ancienneté professionnelle en années (null si sans emploi / retraité)",
    )

    education_type: EducationType
    family_status: FamilyStatus
    housing_type: HousingType
    occupation_type: OccupationType | None = None
    organization_type: OrganizationType | None = None

    ext_source_1: float | None = Field(default=None, ge=0, le=1)
    ext_source_2: float | None = Field(default=None, ge=0, le=1)
    ext_source_3: float | None = Field(default=None, ge=0, le=1)

    @field_validator("annuity_amount")
    @classmethod
    def _annuity_not_above_credit(cls, v: float, info):
        credit = info.data.get("credit_amount")
        if credit is not None and v > credit:
            raise ValueError("annuity_amount ne peut pas dépasser credit_amount")
        return v

    @field_validator("years_employed")
    @classmethod
    def _employment_not_above_age(cls, v: float, info):
        age = info.data.get("age_years")
        if v is not None and age is not None and v > age - 16:
            raise ValueError("years_employed incohérent avec age_years")
        return v


class PredictionResponse(BaseModel):
    request_id: str
    score: float = Field(description="Probabilité de défaut estimée (0-1)")
    decision: Literal["ACCEPTE", "REFUSE"]
    threshold: float
    model_version: str
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str
    threshold: float
    metrics: dict
    n_features: int
