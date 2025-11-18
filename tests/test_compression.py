"""Tests for gzip compression in retrieval corpus serialization."""

import pytest
import gzip
import json
import tempfile
from pathlib import Path

from src.code_explainer.retrieval.retriever import CodeRetriever


class TestCorpusCompression:
    """Test suite for corpus compression functionality."""

    def test_save_index_creates_compressed_corpus(self):
        """Test that save_index creates gzip-compressed corpus file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = CodeRetriever()
            test_codes = [
                "def foo(): pass",
                "def bar(): return 42",
                "def baz(x): return x * 2"
            ]
            
            retriever.build_index(test_codes)
            save_path = str(Path(tmpdir) / "test_index")
            retriever.save_index(save_path)
            
            # Check that compressed file exists
            corpus_file_gz = Path(f"{save_path}.corpus.json.gz")
            assert corpus_file_gz.exists(), "Compressed corpus file should be created"

    def test_compressed_corpus_is_valid_gzip(self):
        """Test that saved corpus file is valid gzip format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = CodeRetriever()
            test_codes = ["def test(): pass", "def sample(): return 1"]
            
            retriever.build_index(test_codes)
            save_path = str(Path(tmpdir) / "test_index")
            retriever.save_index(save_path)
            
            corpus_file_gz = Path(f"{save_path}.corpus.json.gz")
            
            # Should be readable as gzip
            with gzip.open(corpus_file_gz, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            assert data == test_codes

    def test_compression_reduces_file_size(self):
        """Test that gzip compression significantly reduces file size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = CodeRetriever()
            
            # Create large corpus
            test_codes = [f"def func{i}():\n    return {i}\n" for i in range(1000)]
            retriever.build_index(test_codes)
            
            save_path = str(Path(tmpdir) / "test_index")
            retriever.save_index(save_path)
            
            # Compare file sizes
            corpus_file_gz = Path(f"{save_path}.corpus.json.gz")
            
            # Estimate uncompressed size from JSON
            json_data = json.dumps(test_codes, separators=(',', ':'))
            uncompressed_size = len(json_data.encode('utf-8'))
            
            compressed_size = corpus_file_gz.stat().st_size
            
            # Compression should achieve at least 50% reduction for repetitive data
            ratio = compressed_size / uncompressed_size
            assert ratio < 0.5, f"Compression ratio {ratio:.2%} should be < 50%"

    def test_load_index_handles_compressed_corpus(self):
        """Test that load_index can read compressed corpus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            retriever1 = CodeRetriever()
            test_codes = ["def foo(): pass", "def bar(): pass"]
            retriever1.build_index(test_codes)
            
            save_path = str(Path(tmpdir) / "test_index")
            retriever1.save_index(save_path)
            
            # Load
            retriever2 = CodeRetriever()
            retriever2.load_index(save_path)
            
            assert retriever2.code_corpus == test_codes

    def test_backward_compatibility_uncompressed_corpus(self):
        """Test that load_index still works with uncompressed corpus files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Manually create uncompressed corpus (old format)
            save_path = str(Path(tmpdir) / "test_index")
            corpus_file = Path(f"{save_path}.corpus.json")
            corpus_file.parent.mkdir(parents=True, exist_ok=True)
            
            test_codes = ["code1", "code2"]
            with open(corpus_file, 'w') as f:
                json.dump(test_codes, f)
            
            # Should still load
            retriever = CodeRetriever()
            # Mock the FAISS index load to avoid actual file operations
            import unittest.mock as mock
            with mock.patch.object(retriever.faiss_index, 'load_index'):
                retriever.load_index(save_path)
            
            assert retriever.code_corpus == test_codes

    def test_load_prefers_compressed_over_uncompressed(self):
        """Test that load_index prefers compressed format when both exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "test_index")
            
            # Create both compressed and uncompressed files
            codes_compressed = ["compressed1", "compressed2"]
            codes_uncompressed = ["uncompressed1", "uncompressed2"]
            
            # Write uncompressed
            corpus_file = Path(f"{save_path}.corpus.json")
            corpus_file.parent.mkdir(parents=True, exist_ok=True)
            with open(corpus_file, 'w') as f:
                json.dump(codes_uncompressed, f)
            
            # Write compressed
            corpus_file_gz = Path(f"{save_path}.corpus.json.gz")
            with gzip.open(corpus_file_gz, 'wt', encoding='utf-8') as f:
                json.dump(codes_compressed, f)
            
            # Load should prefer compressed
            retriever = CodeRetriever()
            import unittest.mock as mock
            with mock.patch.object(retriever.faiss_index, 'load_index'):
                retriever.load_index(save_path)
            
            assert retriever.code_corpus == codes_compressed

    def test_compressed_corpus_with_special_characters(self):
        """Test that compression handles special characters correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = CodeRetriever()
            test_codes = [
                "def foo(s):\n    return s.split('\\n')",
                '"""Docstring with "quotes" and \'apostrophes\'"""',
                "# Comment with unicode: 你好",
                "x = {'key': 'value\twith\ttabs'}"
            ]
            
            retriever.build_index(test_codes)
            save_path = str(Path(tmpdir) / "test_index")
            retriever.save_index(save_path)
            
            # Load and verify
            retriever2 = CodeRetriever()
            import unittest.mock as mock
            with mock.patch.object(retriever2.faiss_index, 'load_index'):
                retriever2.load_index(save_path)
            
            assert retriever2.code_corpus == test_codes

    def test_save_load_roundtrip(self):
        """Test save and load roundtrip preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_codes = [
                "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
                "class MyClass:\n    def __init__(self):\n        self.value = 0",
                "async def async_func():\n    await asyncio.sleep(1)"
            ]
            
            retriever1 = CodeRetriever()
            retriever1.build_index(original_codes)
            
            save_path = str(Path(tmpdir) / "roundtrip_index")
            retriever1.save_index(save_path)
            
            # Load in new retriever
            retriever2 = CodeRetriever()
            import unittest.mock as mock
            with mock.patch.object(retriever2.faiss_index, 'load_index'):
                retriever2.load_index(save_path)
            
            assert retriever2.code_corpus == original_codes


class TestCompressionPerformance:
    """Performance tests for corpus compression."""

    @pytest.mark.slow
    def test_compression_save_performance(self):
        """Test that gzip compression doesn't significantly slow save."""
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = CodeRetriever()
            
            # Create medium-sized corpus
            test_codes = [f"def func{i}(x):\n    return x * {i}\n" for i in range(500)]
            retriever.build_index(test_codes)
            
            save_path = str(Path(tmpdir) / "perf_index")
            
            # Time the save operation
            start = time.time()
            retriever.save_index(save_path)
            save_time = time.time() - start
            
            # Should complete in reasonable time (varies by system)
            assert save_time < 5.0, f"Save took {save_time:.2f}s, expected < 5s"

    @pytest.mark.slow
    def test_compression_load_performance(self):
        """Test that gzip decompression is fast."""
        import time
        
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever1 = CodeRetriever()
            test_codes = [f"def func{i}(x):\n    return x * {i}\n" for i in range(500)]
            retriever1.build_index(test_codes)
            
            save_path = str(Path(tmpdir) / "perf_index")
            retriever1.save_index(save_path)
            
            # Time the load operation
            retriever2 = CodeRetriever()
            import unittest.mock as mock
            
            start = time.time()
            with mock.patch.object(retriever2.faiss_index, 'load_index'):
                retriever2.load_index(save_path)
            load_time = time.time() - start
            
            # Should complete quickly
            assert load_time < 2.0, f"Load took {load_time:.2f}s, expected < 2s"


class TestCompressionEdgeCases:
    """Edge case tests for compression."""

    def test_empty_corpus_compression(self):
        """Test that empty corpus can be compressed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = CodeRetriever()
            retriever.build_index([])
            
            save_path = str(Path(tmpdir) / "empty_index")
            retriever.save_index(save_path)
            
            corpus_file_gz = Path(f"{save_path}.corpus.json.gz")
            assert corpus_file_gz.exists()
            
            # Should load as empty
            retriever2 = CodeRetriever()
            import unittest.mock as mock
            with mock.patch.object(retriever2.faiss_index, 'load_index'):
                retriever2.load_index(save_path)
            
            assert retriever2.code_corpus == []

    def test_single_item_corpus(self):
        """Test compression with single item."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = CodeRetriever()
            retriever.build_index(["single code snippet"])
            
            save_path = str(Path(tmpdir) / "single_index")
            retriever.save_index(save_path)
            
            retriever2 = CodeRetriever()
            import unittest.mock as mock
            with mock.patch.object(retriever2.faiss_index, 'load_index'):
                retriever2.load_index(save_path)
            
            assert retriever2.code_corpus == ["single code snippet"]

    def test_very_large_corpus_compression(self):
        """Test compression with very large corpus."""
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = CodeRetriever()
            
            # Create large corpus
            test_codes = [
                f"def long_function_{i}():\n" +
                "".join([f"    line_{j} = {k}\n" for j in range(100)])
                for i in range(100)
            ]
            
            retriever.build_index(test_codes)
            save_path = str(Path(tmpdir) / "large_index")
            retriever.save_index(save_path)
            
            corpus_file_gz = Path(f"{save_path}.corpus.json.gz")
            assert corpus_file_gz.exists()
            assert corpus_file_gz.stat().st_size < 5_000_000  # Should be under 5MB compressed
