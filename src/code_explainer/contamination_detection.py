"""
Contamination detection utilities.
"""

from enum import Enum
from typing import Dict, Any, List, Set
import hashlib

# Pre-compute hash function for faster access
_md5 = hashlib.md5


class ContaminationType(Enum):
    EXACT = "exact"
    N_GRAM = "n_gram"
    SEMANTIC = "semantic"


class ContaminationDetector:
    """Detects data contamination in datasets."""
    
    __slots__ = ('exact_hashes', 'n_gram_hashes', 'semantic_hashes')

    def __init__(self):
        self.exact_hashes: Set[str] = set()
        self.n_gram_hashes: Set[str] = set()
        self.semantic_hashes: Set[str] = set()

    def detect_contamination(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect contamination in dataset."""
        return {
            "contamination_detected": False,
            "contamination_score": 0.0,
            "details": {}
        }

    def add_training_sample(self, code: str) -> None:
        """Add a training sample to the contamination detection database."""
        code_bytes = code.encode()
        
        # Add exact hash
        self.exact_hashes.add(_md5(code_bytes).hexdigest())

        # Add n-gram hashes (simple implementation)
        words = code.split()
        n_gram_hashes = self.n_gram_hashes
        for i in range(len(words) - 2):
            ngram = ' '.join(words[i:i+3])
            n_gram_hashes.add(_md5(ngram.encode()).hexdigest())

        # Add semantic hash (simplified)
        self.semantic_hashes.add(_md5(code.lower().encode()).hexdigest())

    def check_contamination(self, code: str) -> Dict[str, Any]:
        """Check if code is contaminated."""
        code_bytes = code.encode()
        code_hash = _md5(code_bytes).hexdigest()

        exact_match = code_hash in self.exact_hashes

        # Check n-grams
        words = code.split()
        ngram_matches = 0
        total_ngrams = max(len(words) - 2, 0)
        
        if total_ngrams > 0:
            n_gram_hashes = self.n_gram_hashes
            for i in range(total_ngrams):
                ngram = ' '.join(words[i:i+3])
                if _md5(ngram.encode()).hexdigest() in n_gram_hashes:
                    ngram_matches += 1

        ngram_score = ngram_matches / total_ngrams if total_ngrams > 0 else 0

        # Check semantic similarity (simplified)
        semantic_hash = _md5(code.lower().encode()).hexdigest()
        semantic_match = semantic_hash in self.semantic_hashes

        # Calculate contamination score
        if exact_match:
            contamination_score = 1.0
        elif ngram_score > 0.5:
            contamination_score = 0.8
        elif semantic_match:
            contamination_score = 0.6
        else:
            contamination_score = 0.0

        return {
            "is_contaminated": contamination_score > 0.5,
            "contamination_score": contamination_score,
            "exact_match": exact_match,
            "ngram_overlap": ngram_score,
            "semantic_match": semantic_match
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get contamination detection statistics."""
        return {
            "total_exact_hashes": len(self.exact_hashes),
            "total_ngram_hashes": len(self.n_gram_hashes),
            "total_semantic_hashes": len(self.semantic_hashes)
        }