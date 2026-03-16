"""Tests for database models and persistence layer."""

import os
import tempfile
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from code_explainer.database import (
    DatabaseConfig,
    DatabaseManager,
    AuditLog,
    RequestHistory,
    CacheEntry,
    ModelMetrics,
    get_database_manager,
    init_database,
    close_database
)


@pytest.fixture
def temp_db_path():
    """Create temporary database path."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def db_config(temp_db_path):
    """Create database config for testing."""
    return DatabaseConfig(url=f"sqlite:///{temp_db_path}", echo=False)


@pytest.fixture
def db_manager(db_config):
    """Create database manager for testing."""
    manager = DatabaseManager(db_config)
    manager.create_tables()
    yield manager
    manager.drop_tables()


def test_database_config_defaults():
    """Test default database configuration."""
    config = DatabaseConfig()
    assert config.url.startswith("sqlite:///")
    assert config.echo is False
    assert config.pool_size == 10
    assert config.enable_migrations is True


def test_database_config_from_env(monkeypatch, temp_db_path):
    """Test database config from environment variables."""
    monkeypatch.setenv("CODE_EXPLAINER_DATABASE_URL", f"sqlite:///{temp_db_path}")
    monkeypatch.setenv("CODE_EXPLAINER_DATABASE_ECHO", "1")
    monkeypatch.setenv("CODE_EXPLAINER_DATABASE_POOL_SIZE", "20")
    monkeypatch.setenv("CODE_EXPLAINER_DATABASE_MIGRATIONS", "0")

    config = DatabaseConfig.from_env()
    assert config.url == f"sqlite:///{temp_db_path}"
    assert config.echo is True
    assert config.pool_size == 20
    assert config.enable_migrations is False


def test_database_manager_create_tables(db_manager):
    """Test table creation."""
    # Tables should be created by fixture
    with db_manager.get_session() as session:
        # Check if tables exist by trying to query them
        result = session.query(AuditLog).count()
        assert result == 0  # Should be empty


def test_audit_log_creation(db_manager):
    """Test audit log creation and retrieval."""
    with db_manager.get_session() as session:
        audit = AuditLog(
            request_id="req-123",
            operation="STORE",
            data_type="code_snippet",
            size_bytes=1024,
            metadata={"model": "codet5-base"},
            user_id="user-456",
            ip_address="192.168.1.1"
        )
        session.add(audit)
        session.commit()

        # Retrieve and verify
        retrieved = session.query(AuditLog).filter_by(request_id="req-123").first()
        assert retrieved is not None
        assert retrieved.operation == "STORE"
        assert retrieved.data_type == "code_snippet"
        assert retrieved.size_bytes == 1024
        assert retrieved.metadata == {"model": "codet5-base"}


def test_request_history_creation(db_manager):
    """Test request history creation."""
    with db_manager.get_session() as session:
        request = RequestHistory(
            request_id="req-456",
            method="POST",
            endpoint="/api/v1/explain",
            status_code=200,
            processing_time_ms=125.5,
            request_size_bytes=512,
            response_size_bytes=1024,
            user_agent="test-agent",
            ip_address="127.0.0.1",
            metadata={"strategy": "vanilla"}
        )
        session.add(request)
        session.commit()

        retrieved = session.query(RequestHistory).filter_by(request_id="req-456").first()
        assert retrieved is not None
        assert retrieved.method == "POST"
        assert retrieved.endpoint == "/api/v1/explain"
        assert retrieved.status_code == 200
        assert retrieved.processing_time_ms == 125.5


def test_cache_entry_operations(db_manager):
    """Test cache entry CRUD operations."""
    # Test set
    db_manager.set_cache_entry(
        cache_key="test_key",
        cache_type="explanation",
        content='{"explanation": "test"}',
        ttl_seconds=3600,
        metadata={"model": "codet5-base"}
    )

    # Test get
    entry = db_manager.get_cache_entry("test_key")
    assert entry is not None
    assert entry["content"] == '{"explanation": "test"}'
    assert entry["metadata"] == {"model": "codet5-base"}
    assert entry["access_count"] == 1

    # Test get non-existent
    entry = db_manager.get_cache_entry("non_existent")
    assert entry is None


def test_cache_entry_update(db_manager):
    """Test cache entry update."""
    # Initial set
    db_manager.set_cache_entry("update_key", "explanation", "old_content")

    # Update
    db_manager.set_cache_entry("update_key", "explanation", "new_content")

    # Verify update
    entry = db_manager.get_cache_entry("update_key")
    assert entry["content"] == "new_content"


def test_cleanup_expired_cache(db_manager):
    """Test expired cache cleanup."""
    # Add entry with short TTL
    with db_manager.get_session() as session:
        expired_entry = CacheEntry(
            cache_key="expired_key",
            cache_type="explanation",
            content="expired",
            ttl_seconds=1,
            created_at=datetime.utcnow() - timedelta(seconds=10)  # Already expired
        )
        session.add(expired_entry)

        valid_entry = CacheEntry(
            cache_key="valid_key",
            cache_type="explanation",
            content="valid",
            ttl_seconds=3600,
            created_at=datetime.utcnow()
        )
        session.add(valid_entry)

    # Cleanup
    removed = db_manager.cleanup_expired_cache()
    assert removed == 1

    # Verify expired entry removed, valid entry remains
    assert db_manager.get_cache_entry("expired_key") is None
    assert db_manager.get_cache_entry("valid_key") is not None


def test_log_audit_event(db_manager):
    """Test audit event logging."""
    db_manager.log_audit_event(
        request_id="audit-123",
        operation="RETRIEVE",
        data_type="explanation",
        size_bytes=256,
        metadata={"cached": True}
    )

    with db_manager.get_session() as session:
        audit = session.query(AuditLog).filter_by(request_id="audit-123").first()
        assert audit is not None
        assert audit.operation == "RETRIEVE"
        assert audit.data_type == "explanation"


def test_log_request(db_manager):
    """Test request logging."""
    db_manager.log_request(
        request_id="req-log-123",
        method="GET",
        endpoint="/api/v1/health",
        status_code=200,
        processing_time_ms=5.2,
        request_size_bytes=0,
        response_size_bytes=45,
        user_agent="pytest",
        ip_address="127.0.0.1"
    )

    with db_manager.get_session() as session:
        request = session.query(RequestHistory).filter_by(request_id="req-log-123").first()
        assert request is not None
        assert request.method == "GET"
        assert request.endpoint == "/api/v1/health"
        assert request.status_code == 200


def test_get_request_stats(db_manager):
    """Test request statistics retrieval."""
    # Add some test data
    with db_manager.get_session() as session:
        for i in range(5):
            request = RequestHistory(
                request_id=f"stats-{i}",
                method="POST",
                endpoint="/api/v1/explain",
                status_code=200 if i < 4 else 500,
                processing_time_ms=100.0 + i * 10,
                timestamp=datetime.utcnow()
            )
            session.add(request)

    stats = db_manager.get_request_stats(hours=1)
    assert stats["total_requests"] == 5
    assert stats["status_distribution"][200] == 4
    assert stats["status_distribution"][500] == 1
    assert stats["endpoint_usage"]["/api/v1/explain"] == 5
    assert stats["avg_processing_time_ms"] > 100


def test_get_database_manager_singleton():
    """Test database manager singleton pattern."""
    # Reset global state
    global _db_manager
    import code_explainer.database
    code_explainer.database._db_manager = None

    manager1 = get_database_manager()
    manager2 = get_database_manager()

    assert manager1 is manager2


def test_init_database():
    """Test database initialization."""
    # Reset global state
    global _db_manager
    import code_explainer.database
    code_explainer.database._db_manager = None

    init_database()
    manager = get_database_manager()
    assert manager is not None

    # Cleanup
    close_database()
