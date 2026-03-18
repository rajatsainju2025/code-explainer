"""Redis caching and queue infrastructure for scalable deployments."""

import redis
from typing import Optional, Any, Dict

from .utils.hashing import json_loads, json_dumps
from datetime import timedelta
import logging
from config_manager import settings


logger = logging.getLogger("code-explainer.redis")


class RedisClient:
    """Singleton Redis client with connection pooling."""

    _instance = None
    _redis_conn = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize Redis connection pool."""
        if self._redis_conn is None:
            try:
                # Parse Redis URL and create connection pool
                self._redis_conn = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    health_check_interval=30
                )
                # Test connection
                self._redis_conn.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                raise

    @property
    def conn(self) -> redis.Redis:
        """Get Redis connection."""
        if self._redis_conn is None:
            self.__init__()
        return self._redis_conn

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        try:
            value = self.conn.get(key)
            if value:
                return json_loads(value)
            return None
        except Exception as e:
            logger.error(f"Error reading from cache: {str(e)}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default: settings.redis_ttl)

        Returns:
            Success status
        """
        try:
            ttl = ttl or settings.redis_ttl
            serialized = json_dumps(value)
            self.conn.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key

        Returns:
            Success status
        """
        try:
            self.conn.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists, False otherwise
        """
        try:
            return self.conn.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache: {str(e)}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., 'inference:*')

        Returns:
            Number of keys deleted
        """
        try:
            keys = self.conn.keys(pattern)
            if keys:
                return self.conn.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing pattern: {str(e)}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis server statistics.

        Returns:
            Dictionary with stats
        """
        try:
            info = self.conn.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "used_memory_percent": info.get("used_memory_percent"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "uptime_seconds": info.get("uptime_in_seconds"),
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}


# Global singleton instance
redis_client = RedisClient()
