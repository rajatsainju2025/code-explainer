"""HTTP connection pooling for efficient external service communication."""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

try:
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConnectionPool:
    """HTTP connection pool with connection reuse and retry logic."""
    
    __slots__ = ('pool_connections', 'pool_maxsize', 'max_retries', 'backoff_factor', 'session')
    
    def __init__(self, pool_connections: int = 10, pool_maxsize: int = 10,
                 max_retries: int = 3, backoff_factor: float = 0.3):
        """Initialize connection pool.
        
        Args:
            pool_connections: Number of pools to cache
            pool_maxsize: Maximum connections per pool
            max_retries: Maximum number of retries
            backoff_factor: Backoff factor for retries
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for connection pooling")
        
        self.pool_connections = pool_connections
        self.pool_maxsize = pool_maxsize
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Create session with connection pooling
        self.session = requests.Session()
        self._setup_retries()
    
    def _setup_retries(self):
        """Setup retry strategy for the session."""
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["POST", "GET", "PUT"])  # Use frozenset for immutability
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.pool_connections,
            pool_maxsize=self.pool_maxsize
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    @contextmanager
    def get_session(self):
        """Get session as context manager."""
        try:
            yield self.session
        except Exception as e:
            logger.error(f"Session error: {e}")
            raise
    
    def close(self):
        """Close all connections in the pool."""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global connection pool instance
_global_pool: Optional[ConnectionPool] = None


def get_connection_pool(
    pool_connections: int = 10,
    pool_maxsize: int = 10,
    max_retries: int = 3,
    backoff_factor: float = 0.3
) -> Optional[ConnectionPool]:
    """Get or create global connection pool.
    
    Args:
        pool_connections: Number of pools to cache
        pool_maxsize: Maximum connections per pool
        max_retries: Maximum retries
        backoff_factor: Backoff factor for retries
    
    Returns:
        ConnectionPool instance or None if requests not available
    """
    global _global_pool
    
    if not REQUESTS_AVAILABLE:
        logger.warning("requests library not available, connection pooling disabled")
        return None
    
    if _global_pool is None:
        _global_pool = ConnectionPool(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=max_retries,
            backoff_factor=backoff_factor
        )
    
    return _global_pool


def close_global_pool():
    """Close global connection pool."""
    global _global_pool
    if _global_pool is not None:
        _global_pool.close()
        _global_pool = None
