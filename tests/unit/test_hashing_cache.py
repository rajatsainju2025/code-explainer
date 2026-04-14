"""Tests for hashing cache behavior and performance."""

from typing import Any, Callable

import pytest
from code_explainer.utils.hashing import fast_hash_str, fast_hash_bytes, json_loads, json_dumps


def _cache_method(func: Callable[..., str], name: str) -> Callable[..., Any]:
    return getattr(func, name)


class TestHashingFunctions:
    """Tests for hashing functions."""

    def setup_method(self):
        _cache_method(fast_hash_str, "cache_clear")()
        _cache_method(fast_hash_bytes, "cache_clear")()
    
    def test_fast_hash_str_cache(self):
        """Test that fast_hash_str returns consistent results."""
        a = "repeated-key"
        h1 = fast_hash_str(a)
        h2 = fast_hash_str(a)
        assert h1 == h2
    
    def test_fast_hash_str_different_inputs(self):
        """Test that different inputs produce different hashes."""
        h1 = fast_hash_str("input1")
        h2 = fast_hash_str("input2")
        assert h1 != h2
    
    def test_fast_hash_bytes_consistent(self):
        """Test that fast_hash_bytes returns consistent results."""
        data = b"test data"
        h1 = fast_hash_bytes(data)
        h2 = fast_hash_bytes(data)
        assert h1 == h2
    
    def test_fast_hash_bytes_different_inputs(self):
        """Test that different byte inputs produce different hashes."""
        h1 = fast_hash_bytes(b"data1")
        h2 = fast_hash_bytes(b"data2")
        assert h1 != h2

    def test_fast_hash_str_skips_cache_for_large_inputs(self):
        """Large unique strings should not consume LRU slots."""
        large_value = "x" * 4096
        before = _cache_method(fast_hash_str, "cache_info")()
        fast_hash_str(large_value)
        after = _cache_method(fast_hash_str, "cache_info")()

        assert after.currsize == before.currsize

    def test_fast_hash_bytes_skips_cache_for_large_inputs(self):
        """Large unique byte payloads should not consume LRU slots."""
        large_value = b"x" * 4096
        before = _cache_method(fast_hash_bytes, "cache_info")()
        fast_hash_bytes(large_value)
        after = _cache_method(fast_hash_bytes, "cache_info")()

        assert after.currsize == before.currsize


class TestJsonUtilities:
    """Tests for JSON serialization utilities."""
    
    def test_json_roundtrip_dict(self):
        """Test JSON serialization roundtrip for dicts."""
        data = {"key": "value", "number": 42}
        serialized = json_dumps(data)
        deserialized = json_loads(serialized)
        assert deserialized == data
    
    def test_json_roundtrip_list(self):
        """Test JSON serialization roundtrip for lists."""
        data = [1, 2, 3, "test"]
        serialized = json_dumps(data)
        deserialized = json_loads(serialized)
        assert deserialized == data
    
    def test_json_roundtrip_nested(self):
        """Test JSON serialization for nested structures."""
        data = {"nested": {"list": [1, 2, 3]}, "value": True}
        serialized = json_dumps(data)
        deserialized = json_loads(serialized)
        assert deserialized == data
    
    def test_json_loads_bytes(self):
        """Test that json_loads accepts bytes input."""
        data = b'{"key": "value"}'
        result = json_loads(data)
        assert result == {"key": "value"}
    
    def test_json_loads_string(self):
        """Test that json_loads accepts string input."""
        data = '{"key": "value"}'
        result = json_loads(data)
        assert result == {"key": "value"}


@pytest.mark.benchmark
class TestHashingPerformance:
    """Benchmark tests for hashing performance."""
    
    def test_hash_str_performance(self, benchmark):
        """Benchmark hash string performance."""
        def hash_many():
            for i in range(1000):
                fast_hash_str(f"input-{i}")
        benchmark(hash_many)
    
    def test_json_dumps_performance(self, benchmark):
        """Benchmark JSON serialization performance."""
        data = {"key": "value", "numbers": list(range(100))}
        def serialize_many():
            for _ in range(1000):
                json_dumps(data)
        benchmark(serialize_many)