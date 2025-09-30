#!/usr/bin/env python3
"""Test script for advanced caching functionality."""

import time
import json
from pathlib import Path

from src.code_explainer.advanced_cache import (
    CacheManager,
    CacheStrategy,
    AdvancedCacheManager,
    CacheInvalidator
)


def test_basic_caching():
    """Test basic cache operations."""
    print("🧪 Testing basic caching operations...")

    cache = AdvancedCacheManager(
        cache_dir=".cache/test_basic",
        max_memory_entries=100,
        strategy=CacheStrategy.LRU
    )

    # Test put and get
    cache.put("test_key_1", "test_value_1", tags={"test", "basic"})
    cache.put("test_key_2", "test_value_2", tags={"test", "basic"})

    assert cache.get("test_key_1") == "test_value_1"
    assert cache.get("test_key_2") == "test_value_2"

    # Test cache stats
    stats = cache.get_metrics()
    assert stats['memory_entries'] == 2

    print("✅ Basic caching operations work correctly")


def test_cache_invalidation():
    """Test cache invalidation strategies."""
    print("🧪 Testing cache invalidation...")

    cache = AdvancedCacheManager(
        cache_dir=".cache/test_invalidation",
        max_memory_entries=100
    )

    # Test version-based invalidation
    cache.put("versioned_key", "value_v1", version="1.0")
    assert cache.get("versioned_key", version_check="1.0") == "value_v1"
    assert cache.get("versioned_key", version_check="2.0") is None  # Should be invalidated

    # Test tag-based invalidation
    cache.put("tagged_key_1", "value1", tags={"group_a"})
    cache.put("tagged_key_2", "value2", tags={"group_a"})
    cache.put("tagged_key_3", "value3", tags={"group_b"})

    invalidated = cache.invalidate_by_tag("group_a")
    assert invalidated == 2
    assert cache.get("tagged_key_1") is None
    assert cache.get("tagged_key_2") is None
    assert cache.get("tagged_key_3") == "value3"

    print("✅ Cache invalidation works correctly")


def test_cache_strategies():
    """Test different cache eviction strategies."""
    print("🧪 Testing cache eviction strategies...")

    # Test LRU strategy
    lru_cache = AdvancedCacheManager(
        cache_dir=".cache/test_lru",
        max_memory_entries=3,
        strategy=CacheStrategy.LRU
    )

    lru_cache.put("key1", "value1")
    lru_cache.put("key2", "value2")
    lru_cache.put("key3", "value3")

    # Access key1 to make it most recently used
    lru_cache.get("key1")

    # Add one more to trigger eviction
    lru_cache.put("key4", "value4")

    # key2 should be evicted (least recently used)
    assert lru_cache.get("key1") == "value1"
    assert lru_cache.get("key2") is None
    assert lru_cache.get("key3") == "value3"
    assert lru_cache.get("key4") == "value4"

    print("✅ Cache eviction strategies work correctly")


def test_cache_persistence():
    """Test cache persistence across sessions."""
    print("🧪 Testing cache persistence...")

    cache_dir = ".cache/test_persistence"

    # First session
    cache1 = AdvancedCacheManager(
        cache_dir=cache_dir,
        max_memory_entries=100
    )
    cache1.put("persistent_key", "persistent_value", tags={"persistent"})
    cache1.put("session_key", "session_value", tags={"session"})

    # Simulate session end by creating new instance
    cache2 = AdvancedCacheManager(
        cache_dir=cache_dir,
        max_memory_entries=100
    )

    # Persistent data should be available
    assert cache2.get("persistent_key") == "persistent_value"

    print("✅ Cache persistence works correctly")


def test_cache_performance():
    """Test cache performance monitoring."""
    print("🧪 Testing cache performance monitoring...")

    cache = AdvancedCacheManager(
        cache_dir=".cache/test_performance",
        enable_monitoring=True
    )

    # Perform some operations
    for i in range(10):
        cache.put(f"perf_key_{i}", f"perf_value_{i}")

    for i in range(10):
        cache.get(f"perf_key_{i}")

    # Check some misses
    for i in range(10, 15):
        cache.get(f"missing_key_{i}")

    stats = cache.get_metrics()
    assert stats['total_requests'] == 25  # 10 hits + 15 misses
    assert stats['hits'] == 10
    assert stats['misses'] == 15
    assert 0.0 <= stats['hit_rate'] <= 1.0

    print("✅ Cache performance monitoring works correctly")


def test_unified_cache_manager():
    """Test the unified cache manager."""
    print("🧪 Testing unified cache manager...")

    # Clean up any existing test caches
    import shutil
    test_cache_dir = Path(".cache/test_unified")
    if test_cache_dir.exists():
        shutil.rmtree(test_cache_dir)

    manager = CacheManager(base_cache_dir=str(test_cache_dir))

    # Test all cache types
    explanation_cache = manager.get_explanation_cache()
    embedding_cache = manager.get_embedding_cache()
    advanced_cache = manager.get_advanced_cache()

    # Test explanation cache
    explanation_cache.put("test_code", "vanilla", "codet5-base", "test explanation")
    assert explanation_cache.get("test_code", "vanilla", "codet5-base") == "test explanation"

    # Test advanced cache
    advanced_cache.put("advanced_key", "advanced_value")
    assert advanced_cache.get("advanced_key") == "advanced_value"

    # Test stats
    stats = manager.get_cache_stats()
    assert 'explanation_cache' in stats
    assert 'advanced_cache' in stats

    print("✅ Unified cache manager works correctly")


def main():
    """Run all cache tests."""
    print("🚀 Running advanced caching tests...\n")

    try:
        test_basic_caching()
        test_cache_invalidation()
        test_cache_strategies()
        test_cache_persistence()
        test_cache_performance()
        test_unified_cache_manager()

        print("\n🎉 All advanced caching tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

    finally:
        # Clean up test caches
        import shutil
        for test_dir in Path(".cache").glob("test_*"):
            if test_dir.is_dir():
                shutil.rmtree(test_dir)


if __name__ == "__main__":
    main()