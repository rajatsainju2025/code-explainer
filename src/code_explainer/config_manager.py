"""Secure configuration management with environment variables and secrets."""

import os
from typing import Optional
from pydantic import BaseSettings, Field, validator
import warnings


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_debug: bool = Field(False, env="API_DEBUG")
    api_key: str = Field(..., env="API_KEY")  # Required, from K8s secret

    # Database Configuration
    database_url: str = Field(
        "sqlite:///./code_explainer.db",
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(5, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(10, env="DATABASE_MAX_OVERFLOW")
    database_pool_timeout: int = Field(30, env="DATABASE_POOL_TIMEOUT")

    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    redis_ttl: int = Field(3600, env="REDIS_TTL")

    # Model Configuration
    model_name: str = Field("codet5-base", env="MODEL_NAME")
    model_path: str = Field("./results", env="CODE_EXPLAINER_MODEL_PATH")
    config_path: str = Field("./configs/default.yaml", env="CODE_EXPLAINER_CONFIG_PATH")

    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_dir: str = Field("/var/log/code-explainer", env="LOG_DIR")

    # Feature Flags
    enable_cache: bool = Field(True, env="ENABLE_CACHE")
    enable_monitoring: bool = Field(True, env="ENABLE_MONITORING")
    enable_audit_logging: bool = Field(True, env="ENABLE_AUDIT_LOGGING")

    # Security Configuration
    cors_allowed_origins: str = Field(
        "http://localhost:8000,http://localhost:8501,http://localhost:7860",
        env="CORS_ALLOWED_ORIGINS"
    )
    cors_allow_credentials: bool = Field(True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: list = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: list = ["Content-Type", "Authorization"]

    # Rate Limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(60, env="RATE_LIMIT_PERIOD")

    # SSL/TLS Configuration
    ssl_enabled: bool = Field(False, env="SSL_ENABLED")
    ssl_cert_path: Optional[str] = Field(None, env="SSL_CERT_PATH")
    ssl_key_path: Optional[str] = Field(None, env="SSL_KEY_PATH")

    # Monitoring
    prometheus_enabled: bool = Field(True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(8001, env="PROMETHEUS_PORT")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False

    @validator("api_key")
    def validate_api_key(cls, v):
        """Validate API key is not default/insecure."""
        if v in ["dev-key-12345", "changeme", "default"]:
            warnings.warn(
                "API_KEY is set to an insecure default value. "
                "Update API_KEY environment variable in production."
            )
        if len(v) < 16:
            raise ValueError("API_KEY must be at least 16 characters long")
        return v

    @validator("database_url")
    def validate_database_url(cls, v):
        """Validate database URL."""
        if v.startswith("postgresql://") and "@" in v:
            # Warn if password is in URL
            if ":" in v.split("://")[1]:
                warnings.warn(
                    "Password found in DATABASE_URL. "
                    "Consider using K8s secrets or environment variables instead."
                )
        return v

    @validator("cors_allowed_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


def get_settings() -> Settings:
    """Get application settings.

    Returns:
        Settings instance loaded from environment

    Raises:
        ValueError: If required environment variables are missing
    """
    try:
        return Settings()
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {str(e)}")


# Global settings instance
settings = get_settings()
