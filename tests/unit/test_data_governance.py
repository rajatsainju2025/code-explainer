"""Tests for data governance and lineage tracking."""

import os
import time
from datetime import datetime, timedelta

import pytest

from code_explainer.data_governance import (
    DataGovernanceConfig,
    log_data_access,
    calculate_expiration,
    is_data_expired,
    DataProvenance,
)


def test_data_governance_config_defaults():
    config = DataGovernanceConfig()
    assert config.retention_days == 30
    assert config.cleanup_enabled is True
    assert config.storage_disabled is False


def test_data_governance_config_from_env(monkeypatch):
    monkeypatch.setenv("CODE_EXPLAINER_DATA_RETENTION_DAYS", "60")
    monkeypatch.setenv("CODE_EXPLAINER_CLEANUP_ENABLED", "0")
    
    config = DataGovernanceConfig.from_env()
    assert config.retention_days == 60
    assert config.cleanup_enabled is False


def test_data_governance_disable_storage(monkeypatch):
    monkeypatch.setenv("CODE_EXPLAINER_DATA_STORAGE_DISABLED", "1")
    
    config = DataGovernanceConfig.from_env()
    assert config.storage_disabled is True


def test_calculate_expiration():
    now = datetime.utcnow()
    expiration = calculate_expiration(retention_days=30)
    
    # Should be approximately 30 days from now
    delta = expiration - now
    assert 29 <= delta.days <= 31


def test_is_data_expired():
    # Create timestamp from 40 days ago
    old_timestamp = time.time() - (40 * 24 * 3600)
    
    assert is_data_expired(old_timestamp, retention_days=30) is True


def test_is_data_not_expired():
    # Create timestamp from 20 days ago
    recent_timestamp = time.time() - (20 * 24 * 3600)
    
    assert is_data_expired(recent_timestamp, retention_days=30) is False


def test_log_data_access(caplog):
    import logging
    caplog.clear()
    
    with caplog.at_level(logging.INFO):
        log_data_access("req-123", "STORE", "code_snippet", 1024)
    
    assert "req-123" in caplog.text
    assert "STORE" in caplog.text
    assert "code_snippet" in caplog.text


def test_data_provenance_card_creation():
    provenance = DataProvenance()
    
    card = provenance.create_provenance_card(
        dataset_name="test-dataset",
        description="Test dataset",
        source="https://example.com",
        composition={"total_samples": 100, "languages": ["python"]},
        license_id="CC-BY-4.0"
    )
    
    assert card["name"] == "test-dataset"
    assert card["description"] == "Test dataset"
    assert card["source"] == "https://example.com"
    assert card["license"] == "CC-BY-4.0"
    assert "created_at" in card


def test_data_provenance_save_card(tmp_path):
    provenance_dir = str(tmp_path / "provenance")
    provenance = DataProvenance(provenance_dir=provenance_dir)
    
    card = {
        "name": "test-dataset",
        "description": "Test",
        "source": "https://example.com",
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    path = provenance.save_provenance_card("test-dataset", card)
    
    assert os.path.exists(path)
    
    # Verify saved content
    import json
    with open(path) as f:
        loaded = json.load(f)
    
    assert loaded["name"] == "test-dataset"
