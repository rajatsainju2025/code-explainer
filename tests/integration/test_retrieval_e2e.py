"""End-to-end FAISS index tests with mocked embeddings.

Tests the full lifecycle: build index → save → load → search.
"""

from unittest.mock import MagicMock
from pathlib import Path
import tempfile

import numpy as np

from code_explainer.retrieval.faiss_index import FAISSIndex


def test_faiss_build_and_search():
    """Test building and searching a FAISS index with mock embeddings."""
    mock_model = MagicMock()
    # Return small embeddings for 2 codes
    mock_model.encode.side_effect = [
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        np.array([[0.15, 0.25, 0.35]], dtype=np.float32),
    ]

    faiss_idx = FAISSIndex(mock_model, batch_size=2)
    codes = ["def add(a, b): return a + b", "def sub(a, b): return a - b"]
    faiss_idx.build_index(codes)

    assert faiss_idx.index is not None
    assert faiss_idx.index.ntotal == 2

    # Now search
    distances, indices = faiss_idx.search(codes[0], k=1)
    assert indices.shape[0] == 1


def test_faiss_save_and_load():
    """Test saving and loading a FAISS index from disk."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = [
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        np.array([[0.15, 0.25]], dtype=np.float32),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build and save
        faiss_idx = FAISSIndex(mock_model, batch_size=2)
        codes = ["code1", "code2"]
        faiss_idx.build_index(codes)

        save_path = Path(tmpdir) / "index.faiss"
        faiss_idx.save_index(str(save_path))
        assert save_path.exists()

        # Load in new instance
        faiss_idx2 = FAISSIndex(mock_model, batch_size=2)
        faiss_idx2.load_index(str(save_path))
        assert faiss_idx2.index is not None
        assert faiss_idx2.index.ntotal == 2


def test_faiss_load_nonexistent_raises():
    """Test that loading a nonexistent index raises FileNotFoundError."""
    mock_model = MagicMock()
    faiss_idx = FAISSIndex(mock_model)

    try:
        faiss_idx.load_index("/nonexistent/path.faiss")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
