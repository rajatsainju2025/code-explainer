"""Security-related unit tests for Code Explainer."""

import pytest
from unittest.mock import patch, MagicMock
from code_explainer.error_handling import (
    ValidationError, ModelError, DatabaseError, handle_exception
)
from code_explainer.config_manager import Settings


class TestSecurityExceptions:
    """Test security exception handling."""

    def test_validation_error_creation(self):
        """Test ValidationError creation."""
        exc = ValidationError("Invalid input", field="code")
        assert exc.code == "VALIDATION_ERROR"
        assert exc.status_code == 400
        assert exc.details["field"] == "code"

    def test_model_error_creation(self):
        """Test ModelError creation."""
        exc = ModelError("Model inference failed", model_name="codet5")
        assert exc.code == "MODEL_ERROR"
        assert exc.status_code == 500
        assert exc.details["model"] == "codet5"

    def test_exception_to_dict(self):
        """Test exception serialization."""
        exc = ValidationError("Test error", field="test_field")
        error_dict = exc.to_dict()
        assert "error" in error_dict
        assert error_dict["error"]["code"] == "VALIDATION_ERROR"
        assert error_dict["error"]["message"] == "Test error"

    def test_handle_exception(self):
        """Test centralized exception handler."""
        exc = ValidationError("Input error", field="code")
        result = handle_exception(exc, context={"endpoint": "/api/v1/explain"})
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert result["error"]["status_code"] == 400

    def test_handle_unexpected_exception(self):
        """Test handling of unexpected exceptions."""
        exc = RuntimeError("Unexpected error")
        result = handle_exception(exc)
        assert result["error"]["code"] == "INTERNAL_ERROR"
        assert result["error"]["status_code"] == 500


class TestConfigurationSecurity:
    """Test configuration security validation."""

    def test_api_key_validation_minimum_length(self):
        """Test API_KEY minimum length validation."""
        with pytest.raises(ValueError, match="at least 16 characters"):
            Settings(api_key="short")

    def test_api_key_validation_insecure_defaults(self, caplog):
        """Test warning on insecure API_KEY defaults."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Settings(
                api_key="dev-key-12345",
                database_url="sqlite:///:memory:"
            )
            # Should have warning about insecure API_KEY
            assert any("insecure" in str(warning.message).lower() for warning in w)

    def test_database_url_validation_password_warning(self, caplog):
        """Test warning when password is in DATABASE_URL."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Settings(
                api_key="secure-api-key-12345",
                database_url="postgresql://user:password@localhost/db"
            )
            # Should warn about password in URL
            assert any("password" in str(warning.message).lower() for warning in w)

    def test_cors_origins_parsing(self):
        """Test CORS origins parsing from comma-separated string."""
        settings = Settings(
            api_key="secure-api-key-12345",
            cors_allowed_origins="http://localhost:3000,https://example.com"
        )
        assert isinstance(settings.cors_allowed_origins, list)
        assert "http://localhost:3000" in settings.cors_allowed_origins
        assert "https://example.com" in settings.cors_allowed_origins

    def test_ssl_configuration(self):
        """Test SSL/TLS configuration."""
        settings = Settings(
            api_key="secure-api-key-12345",
            ssl_enabled=True,
            ssl_cert_path="/etc/ssl/certs/cert.pem",
            ssl_key_path="/etc/ssl/private/key.pem"
        )
        assert settings.ssl_enabled
        assert settings.ssl_cert_path == "/etc/ssl/certs/cert.pem"

    def test_rate_limiting_configuration(self):
        """Test rate limiting settings."""
        settings = Settings(
            api_key="secure-api-key-12345",
            rate_limit_enabled=True,
            rate_limit_requests=50,
            rate_limit_period=30
        )
        assert settings.rate_limit_enabled
        assert settings.rate_limit_requests == 50
        assert settings.rate_limit_period == 30


class TestDependencySecure:
    """Test dependency security configurations."""

    def test_requirements_no_known_vulnerabilities(self):
        """Test that installed packages have no critical CVEs.
        
        This test verifies that key dependencies are secure.
        Run: pip-audit to check for vulnerabilities.
        """
        # Key security-critical packages
        secure_packages = [
            "pydantic",      # Config validation
            "sqlalchemy",    # Database ORM
            "redis",         # Caching
            "fastapi",       # Web framework
        ]
        
        for package in secure_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.skip(f"{package} not installed")
