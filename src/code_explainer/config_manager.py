"""Secure configuration management with environment variables and secrets.

Compatible with Pydantic v2 (pydantic-settings) with v1 fallback.
"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for pydantic v1 installations
    from pydantic import BaseSettings  # type: ignore[no-redef]

# Frozen set for O(1) lookups against insecure defaults
_INSECURE_KEYS = frozenset({"dev-key-12345", "changeme", "default"})


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    # API Configuration
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    api_debug: bool = Field(False, alias="API_DEBUG")
    api_key: str = Field("development-key-min16chars", alias="API_KEY")

    # Database Configuration
    database_url: str = Field(
        "sqlite:///./code_explainer.db", alias="DATABASE_URL"
    )
    database_pool_size: int = Field(5, alias="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(10, alias="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(30, alias="DATABASE_POOL_TIMEOUT")

    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    redis_ttl: int = Field(3600, alias="REDIS_TTL")

    # Model Configuration
    model_name: str = Field("codet5-base", alias="MODEL_NAME")
    model_path: str = Field("./results", alias="CODE_EXPLAINER_MODEL_PATH")
    config_path: str = Field(
        "./configs/default.yaml", alias="CODE_EXPLAINER_CONFIG_PATH"
    )

    # Logging — safe default that works without root privileges
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_dir: str = Field("logs", alias="LOG_DIR")

    # Feature Flags
    enable_cache: bool = Field(True, alias="ENABLE_CACHE")
    enable_monitoring: bool = Field(True, alias="ENABLE_MONITORING")
    enable_audit_logging: bool = Field(True, alias="ENABLE_AUDIT_LOGGING")

    # Security Configuration
    cors_allowed_origins: List[str] = Field(
        default=["http://localhost:8000", "http://localhost:8501", "http://localhost:7860"],
        alias="CORS_ALLOWED_ORIGINS",
    )
    cors_allow_credentials: bool = Field(True, alias="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    cors_allow_headers: List[str] = Field(
        default=["Content-Type", "Authorization"]
    )

    # Rate Limiting
    rate_limit_enabled: bool = Field(True, alias="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(100, alias="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(60, alias="RATE_LIMIT_PERIOD")

    # SSL/TLS Configuration
    ssl_enabled: bool = Field(False, alias="SSL_ENABLED")
    ssl_cert_path: Optional[str] = Field(None, alias="SSL_CERT_PATH")
    ssl_key_path: Optional[str] = Field(None, alias="SSL_KEY_PATH")

    # Monitoring
    prometheus_enabled: bool = Field(True, alias="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(8001, alias="PROMETHEUS_PORT")

    model_config = {
        "env_file": ".env",
        "populate_by_name": True,
        "extra": "ignore",
    }

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not default/insecure."""
        if v in _INSECURE_KEYS:
            warnings.warn(
                "API_KEY is set to an insecure default value. "
                "Update API_KEY environment variable in production.",
                UserWarning,
                stacklevel=2,
            )
        if len(v) < 16:
            raise ValueError("API_KEY must be at least 16 characters long")
        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL has no embedded passwords."""
        if v.startswith("postgresql://") and "@" in v:
            if ":" in v.split("://")[1]:
                warnings.warn(
                    "Password found in DATABASE_URL. "
                    "Consider using K8s secrets or environment variables.",
                    UserWarning,
                    stacklevel=2,
                )
        return v

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: object) -> list:
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application settings (cached singleton).

    Returns:
        Settings instance loaded from environment.

    Raises:
        ValueError: If required environment variables are missing.
    """
    try:
        return Settings()  # type: ignore[call-arg]
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}") from e


# Lazy property — only instantiated on first access, not at import time
class _SettingsProxy:
    """Lazy proxy so `from .config_manager import settings` doesn't
    trigger validation at import time."""

    __slots__ = ("_obj",)

    def __init__(self) -> None:
        object.__setattr__(self, "_obj", None)

    def _load(self) -> Settings:
        obj = object.__getattribute__(self, "_obj")
        if obj is None:
            obj = get_settings()
            object.__setattr__(self, "_obj", obj)
        return obj

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


settings: Settings = _SettingsProxy()  # type: ignore[assignment]
