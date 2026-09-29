from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via environment variables
    (prefix ``SCORING_API_``) or a local ``.env`` file — see ``.env.example``.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SCORING_API_", extra="ignore")

    model_dir: Path = Path("models")
    log_path: Path = Path("logs/predictions.jsonl")
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
