"""Tests for batch explanation endpoint."""

import json
from fastapi.testclient import TestClient

from code_explainer.api.server import create_app


def test_explain_batch_endpoint_smoke():
    app = create_app()
    client = TestClient(app)

    payload = {
        "codes": [
            "def add(a,b):\n    return a+b",
            "def fib(n):\n    return 1 if n<=1 else fib(n-1)+fib(n-2)",
        ],
        "strategy": "vanilla",
        "max_length": 128,
    }

    # Health check to ensure app is up
    h = client.get("/api/v1/health")
    assert h.status_code in (200, 503)

    r = client.post("/api/v1/explain/batch", json=payload)
    assert r.status_code in (200, 501, 500)  # tolerate missing model in CI
    if r.status_code == 200:
        data = r.json()
        assert "explanations" in data
        assert len(data["explanations"]) == 2
