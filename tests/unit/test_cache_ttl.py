"""Tests for cache TTL configuration and TTL-aware caching."""

import time
import os

import pytest

from code_explainer.cache_ttl import CacheTTLConfig, TTLCache


def test_cache_ttl_config_defaults():
    config = CacheTTLConfig()
    assert config.embedding_cache_ttl_seconds == 3600
    assert config.explanation_cache_ttl_seconds == 7200
    assert config.enable_ttl_enforcement is True


def test_cache_ttl_config_from_env(monkeypatch):
    monkeypatch.setenv("CODE_EXPLAINER_CACHE_EMBEDDING_TTL", "1800")
    monkeypatch.setenv("CODE_EXPLAINER_CACHE_EXPLANATION_TTL", "3600")
    
    config = CacheTTLConfig.from_env()
    assert config.embedding_cache_ttl_seconds == 1800
    assert config.explanation_cache_ttl_seconds == 3600


def test_cache_ttl_config_disable_ttl_enforcement(monkeypatch):
    monkeypatch.setenv("CODE_EXPLAINER_CACHE_DISABLE_TTL", "1")
    
    config = CacheTTLConfig.from_env()
    assert config.enable_ttl_enforcement is False


def test_ttl_cache_set_get():
    cache = TTLCache(ttl_seconds=10)
    cache.set("key1", "value1")
    
    assert cache.get("key1") == "value1"


def test_ttl_cache_expired_entry():
    cache = TTLCache(ttl_seconds=1)
    cache.set("key1", "value1")
    
    # Value is available immediately
    assert cache.get("key1") == "value1"
    
    # After TTL expires, None is returned
    time.sleep(1.1)
    assert cache.get("key1") is None


def test_ttl_cache_missing_key():
    cache = TTLCache()
    assert cache.get("nonexistent") is None


def test_ttl_cache_overwrite():
    cache = TTLCache()
    cache.set("key1", "value1")
    cache.set("key1", "value2")
    
    assert cache.get("key1") == "value2"


def test_ttl_cache_clear():
    cache = TTLCache()
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    
    assert cache.size() == 2
    cache.clear()
    assert cache.size() == 0


def test_ttl_cache_cleanup_expired():
    cache = TTLCache(ttl_seconds=1)
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    
    assert cache.size() == 2
    
    time.sleep(1.1)
    
    removed = cache.cleanup_expired()
    assert removed == 2
    assert cache.size() == 0


def test_ttl_cache_cleanup_partial():
    cache = TTLCache(ttl_seconds=1)
    cache.set("key1", "value1")
    
    time.sleep(0.5)
    
    cache.set("key2", "value2")
    
    time.sleep(0.7)  # key1 expired, key2 not yet
    
    removed = cache.cleanup_expired()
    assert removed == 1
    assert cache.get("key2") == "value2"
