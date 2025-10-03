"""API endpoint tests for new v2 endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from code_explainer.api.server import app


class TestAPIV2Endpoints:
    """Test new v2 API endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_health_v2_endpoint(self):
        """Test enhanced health check endpoint."""
        response = self.client.get("/api/v2/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "performance" in data
        assert "security" in data
        assert "version" in data

    def test_performance_endpoint(self):
        """Test performance metrics endpoint."""
        response = self.client.get("/api/v2/performance")
        assert response.status_code == 200

        data = response.json()
        assert "performance_report" in data
        assert "timestamp" in data
        assert isinstance(data["performance_report"], str)

    def test_security_validation_endpoint(self):
        """Test security validation endpoint."""
        test_code = {"code": "def safe_function(): return 1"}

        response = self.client.post("/api/v2/validate-security", json=test_code)
        assert response.status_code == 200

        data = response.json()
        assert "safe" in data
        assert "warnings" in data
        assert "code_length" in data
        assert "validation_timestamp" in data
        assert data["safe"] is True
        assert isinstance(data["warnings"], list)

    def test_security_validation_unsafe_code(self):
        """Test security validation with potentially unsafe code."""
        unsafe_code = {"code": "import os; os.system('ls')"}

        response = self.client.post("/api/v2/validate-security", json=unsafe_code)
        assert response.status_code == 200

        data = response.json()
        assert "safe" in data
        assert "warnings" in data
        assert isinstance(data["warnings"], list)

    def test_setup_info_endpoint(self):
        """Test setup info endpoint."""
        response = self.client.get("/api/v2/setup-info")
        assert response.status_code == 200

        data = response.json()
        assert "setup_info" in data
        assert "api_version" in data
        assert data["api_version"] == "v2"
        assert isinstance(data["setup_info"], str)

    def test_batch_explain_endpoint(self):
        """Test batch explanation endpoint."""
        batch_request = {
            "codes": ["def add(a, b): return a + b", "print('hello')"],
            "strategy": "vanilla"
        }

        response = self.client.post("/api/v2/batch-explain", json=batch_request)
        # May return 503 if explainer not initialized, but should not crash
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "explanations" in data
            assert "batch_size" in data
            assert "batch_id" in data
            assert data["batch_size"] == 2
            assert len(data["explanations"]) == 2

    def test_model_optimization_endpoint(self):
        """Test model optimization endpoint."""
        optimization_request = {
            "enable_quantization": False,
            "enable_gradient_checkpointing": True,
            "optimize_for_inference": True,
            "optimize_tokenizer": True
        }

        response = self.client.post("/api/v2/optimize-model", json=optimization_request)
        # May return 503 if explainer not initialized, but should not crash
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "optimizations_applied" in data
            assert "optimization_id" in data
            assert isinstance(data["optimizations_applied"], dict)

    def test_secure_explain_endpoint(self):
        """Test secure explain endpoint."""
        explain_request = {
            "code": "def test(): return 42",
            "strategy": "vanilla"
        }

        response = self.client.post("/api/v2/secure-explain", json=explain_request)
        # May return 503 if explainer not initialized, but should not crash
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "explanation" in data
            assert "code_length" in data
            assert "request_id" in data

    def test_invalid_requests(self):
        """Test handling of invalid requests."""
        # Empty code for security validation
        response = self.client.post("/api/v2/validate-security", json={"code": ""})
        assert response.status_code == 200  # Should handle gracefully

        # Missing codes for batch explain
        response = self.client.post("/api/v2/batch-explain", json={})
        assert response.status_code == 422  # Validation error

        # Invalid optimization parameters
        response = self.client.post("/api/v2/optimize-model", json={"invalid_param": True})
        assert response.status_code in [200, 422]  # Should handle or validate


class TestAPIErrorHandling:
    """Test error handling in API endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_endpoint_not_found(self):
        """Test 404 for non-existent endpoints."""
        response = self.client.get("/api/v2/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self):
        """Test 405 for wrong HTTP methods."""
        response = self.client.put("/api/v2/health")
        assert response.status_code == 405

    @patch('code_explainer.api.server.explainer')
    def test_service_unavailable(self, mock_explainer):
        """Test 503 when explainer is not available."""
        mock_explainer.__bool__ = lambda: False
        mock_explainer.__nonzero__ = lambda: False

        response = self.client.get("/api/v2/performance")
        assert response.status_code == 503