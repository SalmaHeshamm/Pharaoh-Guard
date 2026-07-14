"""
Central application configuration.
Single source of truth for every path, threshold, and external-service
credential used across the system. Follows the same BaseSettings +
computed_field pattern used in Muqla_AI/Offline/src/config/settings.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RRS_",  # RRS = Risk Response System
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Project layout
    # ------------------------------------------------------------------ #
    project_root: Path = Path(__file__).resolve().parent.parent

    # ------------------------------------------------------------------ #
    # LLM provider — Groq API (OpenAI-compatible chat completions)
    # ------------------------------------------------------------------ #
    groq_api_key: str = Field(
        default="",
        description=(
            "Groq API key. Required — set via RRS_GROQ_API_KEY env var or "
            "in .env. Get one at https://console.groq.com/keys."
        ),
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL of the Groq OpenAI-compatible API.",
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description=(
            "Groq model id used by the Reasoning Agent. Override via "
            "RRS_LLM_MODEL. Alternatives: `llama-3.1-8b-instant` (faster/cheaper), "
            "`gemma2-9b-it`. See https://console.groq.com/docs/models for the "
            "current list."
        ),
    )
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 60.0

    # ------------------------------------------------------------------ #
    # Risk model integration — trained sklearn Pipeline from Notebook 04/05.
    # Verified: identical artifact (md5) to the working pharaoh-guard-min
    # reference implementation, which is where core/risk_model.py's
    # feature-engineering logic was ported from.
    # ------------------------------------------------------------------ #
    risk_model_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "datascience" / "risk_model.joblib",
        description="Path to the trained sklearn Pipeline (preprocessing + classifier).",
    )
    feature_manifest_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "datascience" / "feature_manifest.json",
        description="Path to feature_manifest.json (Notebook 04) — defines the exact 39-column feature order.",
    )

    # ------------------------------------------------------------------ #
    # Risk thresholds
    # ------------------------------------------------------------------ #
    # The model is a classifier and predicts Risk_Level directly — these
    # score_* buckets are kept only for legacy/analytics display, they are
    # NOT used to derive risk_level anymore.
    risk_score_low_max: float = 30.0
    risk_score_medium_max: float = 55.0
    risk_score_high_max: float = 75.0
    # anything above risk_score_high_max => Critical

    # P(Critical) above this arms the emergency protocol automatically
    # (Notebook 06 threshold sweep — matches the pitch deck's use-case slide).
    critical_alert_threshold: float = Field(default=0.7, description="P(Critical) alert threshold.")

    monitoring_poll_interval_seconds: int = 30
    monitoring_trigger_level: Literal["Medium", "High", "Critical"] = "High"

    # ------------------------------------------------------------------ #
    # RAG / knowledge base
    # ------------------------------------------------------------------ #
    vector_db_backend: Literal["chroma", "faiss"] = "chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 4

    # ------------------------------------------------------------------ #
    # Notification / dispatch integrations
    # ------------------------------------------------------------------ #
    slack_webhook_url: str = ""
    sms_webhook_url: str = ""
    dispatch_dry_run: bool = Field(
        default=True,
        description="If True, dispatch tools log the action instead of calling real webhooks.",
    )

    # ------------------------------------------------------------------ #
    # API server
    # ------------------------------------------------------------------ #
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ------------------------------------------------------------------ #
    # Derived paths (computed, never set directly)
    # ------------------------------------------------------------------ #
    @computed_field
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @computed_field
    @property
    def protocols_dir(self) -> Path:
        return self.data_dir / "protocols"

    @computed_field
    @property
    def vector_store_dir(self) -> Path:
        return self.data_dir / "vector_store"

    @computed_field
    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings instance. Import this, not Settings()."""
    return Settings()
