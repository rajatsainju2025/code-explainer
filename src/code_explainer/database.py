"""Database models and persistence layer for Code Explainer.

Optimized for:
- __slots__ on DatabaseConfig for memory efficiency
- Connection pooling with configurable limits
- Indexed lookups on frequently queried columns
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, Float,
    Boolean, JSON, ForeignKey, Index, func, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager

Base = declarative_base()


class AuditLog(Base):
    """Audit log for all data access and operations."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    data_type = Column(String(32), nullable=False)  # code_snippet, explanation, etc.
    size_bytes = Column(Integer, default=0)
    metadata_json = Column(JSON, nullable=True)
    user_id = Column(String(128), nullable=True)  # For future auth integration
    ip_address = Column(String(45), nullable=True)  # IPv4/IPv6 support

    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_operation', 'operation'),
        Index('idx_audit_request', 'request_id'),
    )


class RequestHistory(Base):
    """Historical record of all API requests."""
    __tablename__ = "request_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    method = Column(String(8), nullable=False)  # GET, POST, etc.
    endpoint = Column(String(256), nullable=False)
    status_code = Column(Integer, nullable=False)
    processing_time_ms = Column(Float, nullable=True)
    request_size_bytes = Column(Integer, default=0)
    response_size_bytes = Column(Integer, default=0)
    user_agent = Column(String(512), nullable=True)
    ip_address = Column(String(45), nullable=True)
    error_message = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=True)  # Additional request context

    __table_args__ = (
        Index('idx_request_timestamp', 'timestamp'),
        Index('idx_request_endpoint', 'endpoint'),
        Index('idx_request_status', 'status_code'),
    )


class CacheEntry(Base):
    """Persistent cache entries for explanations and embeddings."""
    __tablename__ = "cache_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cache_key = Column(String(256), unique=True, nullable=False, index=True)
    cache_type = Column(String(32), nullable=False)  # explanation, embedding, etc.
    content = Column(Text, nullable=False)  # JSON-serialized content
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    last_accessed = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    access_count = Column(Integer, default=0)
    ttl_seconds = Column(Integer, nullable=True)
    metadata_json = Column(JSON, nullable=True)  # Model version, strategy, etc.

    __table_args__ = (
        Index('idx_cache_type', 'cache_type'),
        Index('idx_cache_accessed', 'last_accessed'),
        Index('idx_cache_created', 'created_at'),
    )


class ModelMetrics(Base):
    """Performance metrics for model inference."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    model_name = Column(String(128), nullable=False)
    operation = Column(String(32), nullable=False)  # inference, embedding, etc.
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    processing_time_ms = Column(Float, nullable=False)
    memory_usage_mb = Column(Float, nullable=True)
    device = Column(String(32), nullable=True)  # cpu, cuda, mps
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

    __table_args__ = (
        Index('idx_metrics_timestamp', 'timestamp'),
        Index('idx_metrics_model', 'model_name'),
        Index('idx_metrics_operation', 'operation'),
    )


class DatabaseConfig:
    """Database configuration with environment variable support."""
    
    __slots__ = ('url', 'echo', 'pool_size', 'max_overflow', 'pool_timeout', 
                 'pool_recycle', 'enable_migrations')

    def __init__(
        self,
        url: Optional[str] = None,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        enable_migrations: bool = True
    ):
        """Initialize database configuration.

        Args:
            url: Database URL (postgresql:// or sqlite:///path)
            echo: Enable SQLAlchemy echo logging
            pool_size: Connection pool size
            max_overflow: Max overflow connections
            pool_timeout: Connection timeout
            pool_recycle: Connection recycle time
            enable_migrations: Enable Alembic migrations
        """
        self.url = url or self._get_default_url()
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.enable_migrations = enable_migrations

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables.

        Environment variables:
            CODE_EXPLAINER_DATABASE_URL: Database connection URL
            CODE_EXPLAINER_DATABASE_ECHO: Enable SQL logging (0 or 1)
            CODE_EXPLAINER_DATABASE_POOL_SIZE: Connection pool size
            CODE_EXPLAINER_DATABASE_MIGRATIONS: Enable migrations (0 or 1)
        """
        return cls(
            url=os.getenv("CODE_EXPLAINER_DATABASE_URL"),
            echo=os.getenv("CODE_EXPLAINER_DATABASE_ECHO", "0") == "1",
            pool_size=int(os.getenv("CODE_EXPLAINER_DATABASE_POOL_SIZE", "10")),
            enable_migrations=os.getenv("CODE_EXPLAINER_DATABASE_MIGRATIONS", "1") == "1",
        )

    def _get_default_url(self) -> str:
        """Get default database URL (SQLite for development)."""
        db_path = os.path.join(os.getcwd(), "data", "code_explainer.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return f"sqlite:///{db_path}"

    def create_engine(self):
        """Create SQLAlchemy engine with appropriate settings."""
        if self.url.startswith("sqlite"):
            # SQLite-specific settings
            return create_engine(
                self.url,
                echo=self.echo,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
            )
        else:
            # PostgreSQL/MySQL settings
            return create_engine(
                self.url,
                echo=self.echo,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
            )

    def __repr__(self) -> str:
        # Strip credentials from the URL for safe logging / debugging
        safe_url = self.url.split("@")[-1] if "@" in self.url else self.url
        return (
            f"DatabaseConfig(url=...{safe_url}, "
            f"pool_size={self.pool_size}, "
            f"echo={self.echo})"
        )


class DatabaseManager:
    """Database session and operation manager."""

    __slots__ = ("config", "engine", "SessionLocal")

    def __init__(self, config: DatabaseConfig):
        """Initialize database manager.

        Args:
            config: Database configuration
        """
        self.config = config
        self.engine = config.create_engine()
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self) -> None:
        """Drop all database tables (for testing)."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def log_audit_event(
        self,
        request_id: str,
        operation: str,
        data_type: str,
        size_bytes: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """Log an audit event to the database.

        Args:
            request_id: Unique request identifier
            operation: Operation type (STORE, RETRIEVE, etc.)
            data_type: Type of data accessed
            size_bytes: Size of data in bytes
            metadata: Additional metadata
            user_id: User identifier (future use)
            ip_address: Client IP address
        """
        with self.get_session() as session:
            audit_log = AuditLog(
                request_id=request_id,
                operation=operation,
                data_type=data_type,
                size_bytes=size_bytes,
                metadata=metadata,
                user_id=user_id,
                ip_address=ip_address
            )
            session.add(audit_log)

    def log_request(
        self,
        request_id: str,
        method: str,
        endpoint: str,
        status_code: int,
        processing_time_ms: Optional[float] = None,
        request_size_bytes: int = 0,
        response_size_bytes: int = 0,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a request to the database.

        Args:
            request_id: Unique request identifier
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            processing_time_ms: Processing time in milliseconds
            request_size_bytes: Request body size
            response_size_bytes: Response body size
            user_agent: User agent string
            ip_address: Client IP address
            error_message: Error message if any
            metadata: Additional request metadata
        """
        with self.get_session() as session:
            request_log = RequestHistory(
                request_id=request_id,
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                processing_time_ms=processing_time_ms,
                request_size_bytes=request_size_bytes,
                response_size_bytes=response_size_bytes,
                user_agent=user_agent,
                ip_address=ip_address,
                error_message=error_message,
                metadata=metadata
            )
            session.add(request_log)

    def get_cache_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cache entry by key.

        Args:
            cache_key: Cache key

        Returns:
            Cache entry data or None if not found
        """
        with self.get_session() as session:
            entry = session.query(CacheEntry).filter_by(cache_key=cache_key).first()
            if entry:
                # Update access statistics
                entry.last_accessed = datetime.now(timezone.utc)
                entry.access_count += 1
                return {
                    "content": entry.content,
                    "metadata": entry.metadata_json,
                    "created_at": entry.created_at,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count
                }
        return None

    def set_cache_entry(
        self,
        cache_key: str,
        cache_type: str,
        content: str,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store cache entry.

        Args:
            cache_key: Unique cache key
            cache_type: Type of cached content
            content: JSON-serialized content
            ttl_seconds: Time-to-live in seconds
            metadata: Additional metadata
        """
        with self.get_session() as session:
            entry = session.query(CacheEntry).filter_by(cache_key=cache_key).first()
            if entry:
                # Update existing
                entry.content = content
                entry.last_accessed = datetime.now(timezone.utc)
                entry.metadata_json = metadata
            else:
                # Create new
                entry = CacheEntry(
                    cache_key=cache_key,
                    cache_type=cache_type,
                    content=content,
                    ttl_seconds=ttl_seconds,
                    metadata_json=metadata
                )
                session.add(entry)

    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        cutoff = datetime.now(timezone.utc)
        with self.get_session() as session:
            # For SQLite, use datetime function to calculate expiration
            result = session.query(CacheEntry).filter(
                CacheEntry.ttl_seconds.isnot(None),
                text("datetime(created_at, '+' || ttl_seconds || ' seconds') < :cutoff").params(cutoff=cutoff)
            ).delete()
            return result

    def get_request_stats(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get request statistics for the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            Statistics dictionary
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        with self.get_session() as session:
            # Total requests
            total_requests = session.query(func.count(RequestHistory.id)).filter(
                RequestHistory.timestamp >= cutoff
            ).scalar()

            # Status code distribution
            status_counts = session.query(
                RequestHistory.status_code,
                func.count(RequestHistory.id)
            ).filter(
                RequestHistory.timestamp >= cutoff
            ).group_by(RequestHistory.status_code).all()

            # Average processing time
            avg_time = session.query(func.avg(RequestHistory.processing_time_ms)).filter(
                RequestHistory.timestamp >= cutoff,
                RequestHistory.processing_time_ms.isnot(None)
            ).scalar()

            # Endpoint usage
            endpoint_counts = session.query(
                RequestHistory.endpoint,
                func.count(RequestHistory.id)
            ).filter(
                RequestHistory.timestamp >= cutoff
            ).group_by(RequestHistory.endpoint).all()

            return {
                "total_requests": total_requests or 0,
                "status_distribution": dict(status_counts),
                "avg_processing_time_ms": avg_time or 0,
                "endpoint_usage": dict(endpoint_counts),
                "time_range_hours": hours
            }


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance."""
    global _db_manager
    if _db_manager is None:
        config = DatabaseConfig.from_env()
        _db_manager = DatabaseManager(config)
        if config.enable_migrations:
            _db_manager.create_tables()
    return _db_manager


def init_database() -> None:
    """Initialize database and create tables."""
    manager = get_database_manager()
    manager.create_tables()


def close_database() -> None:
    """Close database connections."""
    global _db_manager
    if _db_manager:
        _db_manager.engine.dispose()
        _db_manager = None
