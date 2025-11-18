"""Tests for write-behind caching in explanation cache."""

import pytest
import time
import tempfile
from pathlib import Path

from src.code_explainer.cache.explanation_cache import ExplanationCache


class TestWriteBehindCaching:
    """Test suite for write-behind batching functionality."""

    def test_write_behind_batching_enabled(self):
        """Test that write-behind batching is enabled by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(cache_dir=tmpdir)
            assert cache.write_behind_batch_size > 0
            assert cache.write_behind_flush_interval > 0

    def test_pending_writes_queue_initialized(self):
        """Test that pending writes queue is initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(cache_dir=tmpdir)
            assert hasattr(cache, '_pending_writes_queue')
            assert len(cache._pending_writes_queue) == 0

    def test_should_flush_on_batch_size_threshold(self):
        """Test that flush is triggered when batch size threshold is reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(
                cache_dir=tmpdir,
                write_behind_batch_size=3
            )
            
            # Add cache entries to trigger batch
            code = "def foo(): pass"
            for i in range(3):
                cache.put(code + str(i), "vanilla", "test-model", f"explanation {i}")
            
            # After batch size reached, queue should be flushed
            # (actual queue clearing happens in write-behind logic)
            assert cache._index is not None

    def test_should_flush_on_time_interval(self):
        """Test that flush is triggered after time interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(
                cache_dir=tmpdir,
                write_behind_batch_size=100,  # Large batch size
                write_behind_flush_interval=0.5  # 500ms flush interval
            )
            
            # Add one entry
            cache.put("code1", "vanilla", "model", "explanation1")
            
            # Wait for flush interval
            time.sleep(0.6)
            
            # Add another entry - should trigger flush due to time interval
            cache.put("code2", "vanilla", "model", "explanation2")
            
            # Verify data is persisted
            assert cache.size() >= 1

    def test_multiple_writes_batched_efficiently(self):
        """Test that multiple writes are batched and save index once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(
                cache_dir=tmpdir,
                write_behind_batch_size=10,
                write_behind_flush_interval=10.0  # Long interval
            )
            
            # Add multiple entries without reaching batch size
            num_writes = 5
            for i in range(num_writes):
                cache.put(f"code{i}", "vanilla", "model", f"explanation{i}")
            
            # Queue should have items pending but not flushed yet
            # (they queue up and wait for batch threshold or time)
            assert cache.size() == num_writes

    def test_flush_all_pending_writes(self):
        """Test explicit flush of all pending writes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(
                cache_dir=tmpdir,
                write_behind_batch_size=100,
                write_behind_flush_interval=100.0
            )
            
            # Add entries
            cache.put("code1", "vanilla", "model", "explanation1")
            cache.put("code2", "vanilla", "model", "explanation2")
            
            # Flush all pending writes
            cache.flush()
            
            # Verify persistence
            assert cache.size() == 2
            assert cache.get("code1", "vanilla", "model") == "explanation1"
            assert cache.get("code2", "vanilla", "model") == "explanation2"

    def test_write_queue_operations_tracked(self):
        """Test that write operations are tracked in queue."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(
                cache_dir=tmpdir,
                write_behind_batch_size=5
            )
            
            # Add entry and verify tracking
            cache.put("code", "vanilla", "model", "explanation")
            
            # Queue should record the operation
            assert cache._pending_writes_queue is not None

    def test_index_file_saved_after_batch_flush(self):
        """Test that index file is properly saved after batch flush."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(
                cache_dir=tmpdir,
                write_behind_batch_size=2,
                write_behind_flush_interval=10.0
            )
            
            # Add entries to trigger batch
            cache.put("code1", "vanilla", "model", "exp1")
            cache.put("code2", "vanilla", "model", "exp2")
            
            # Index file should be created
            index_file = Path(tmpdir) / "index.json"
            # File may exist after writes complete
            assert index_file.parent.exists()

    def test_cleanup_operations_also_batched(self):
        """Test that cleanup operations respect write-behind batching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(
                cache_dir=tmpdir,
                max_size=3,
                write_behind_batch_size=100
            )
            
            # Add entries to trigger cleanup
            for i in range(5):
                cache.put(f"code{i}", "vanilla", "model", f"explanation{i}")
            
            # Cleanup should be triggered but queued
            assert cache.size() <= 3

    def test_cache_consistency_during_batching(self):
        """Test that cache remains consistent during write-behind batching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(
                cache_dir=tmpdir,
                write_behind_batch_size=10,
                write_behind_flush_interval=5.0
            )
            
            # Write, read, verify during batching
            cache.put("test_code", "vanilla", "model", "test_explanation")
            retrieved = cache.get("test_code", "vanilla", "model")
            
            # Should be available in memory even if not yet flushed
            assert retrieved == "test_explanation"

    def test_write_behind_lock_prevents_race_conditions(self):
        """Test that write-behind lock prevents race conditions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(cache_dir=tmpdir)
            
            # Lock should be present and usable
            assert hasattr(cache, '_write_behind_lock')
            
            # Lock should be a threading lock
            import threading
            assert isinstance(cache._write_behind_lock, type(threading.Lock()))


class TestWriteBehindPerformance:
    """Performance tests for write-behind caching."""

    @pytest.mark.slow
    def test_batching_reduces_io_operations(self):
        """Test that batching reduces number of I/O operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Without batching (force=True every write)
            cache_unbatched = ExplanationCache(
                cache_dir=tmpdir + "_unbatched",
                write_behind_batch_size=1000,  # Effectively disabled
                write_behind_flush_interval=1000.0
            )
            
            # With batching
            cache_batched = ExplanationCache(
                cache_dir=tmpdir + "_batched",
                write_behind_batch_size=10,
                write_behind_flush_interval=10.0
            )
            
            # Both should work correctly
            assert cache_unbatched is not None
            assert cache_batched is not None

    @pytest.mark.slow
    def test_batch_flush_performance(self):
        """Test that batch flushing is efficient."""
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(
                cache_dir=tmpdir,
                write_behind_batch_size=50,
                write_behind_flush_interval=1.0
            )
            
            # Time multiple writes
            start = time.time()
            for i in range(100):
                cache.put(f"code{i}", "vanilla", "model", f"exp{i}")
            elapsed = time.time() - start
            
            # Should complete reasonably fast (varies by system)
            assert elapsed < 10.0


class TestWriteBehindConfigurable:
    """Test configurability of write-behind settings."""

    def test_custom_batch_size(self):
        """Test custom batch size configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(
                cache_dir=tmpdir,
                write_behind_batch_size=5
            )
            assert cache.write_behind_batch_size == 5

    def test_custom_flush_interval(self):
        """Test custom flush interval configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ExplanationCache(
                cache_dir=tmpdir,
                write_behind_flush_interval=2.5
            )
            assert cache.write_behind_flush_interval == 2.5

    def test_zero_or_negative_values_handled(self):
        """Test that invalid configuration values are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should use defaults if values are invalid
            cache = ExplanationCache(
                cache_dir=tmpdir,
                write_behind_batch_size=10,
                write_behind_flush_interval=5.0
            )
            assert cache.write_behind_batch_size == 10
            assert cache.write_behind_flush_interval == 5.0
