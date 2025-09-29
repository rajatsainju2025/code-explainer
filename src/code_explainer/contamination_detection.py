"""Adaptive contamination detection using multiple detection strategies."""

import hashlib
import difflib
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle

logger = logging.getLogger(__name__)


class ContaminationType(Enum):
    """Types of contamination detection."""
    EXACT_MATCH = "exact_match"
    NEAR_DUPLICATE = "near_duplicate"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CODE_TEMPLATE = "code_template"
    FUNCTION_SIGNATURE = "function_signature"
    VARIABLE_RENAMING = "variable_renaming"
    STRUCTURAL_SIMILARITY = "structural_similarity"


@dataclass
class ContaminationResult:
    """Result of contamination detection."""
    is_contaminated: bool
    contamination_type: ContaminationType
    confidence_score: float
    source_match: Optional[str] = None
    similarity_metrics: Dict[str, float] = field(default_factory=dict)
    detection_metadata: Dict[str, Any] = field(default_factory=dict)


class ContaminationDetector:
    """Advanced contamination detection system."""

    def __init__(self, training_corpus_path: Optional[str] = None):
        """Initialize contamination detector.

        Args:
            training_corpus_path: Path to training corpus for comparison
        """
        self.training_corpus_path = training_corpus_path
        self.exact_hashes: Set[str] = set()
        self.fuzzy_hashes: Dict[str, str] = {}
        self.ast_patterns: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Load training corpus if available
        if training_corpus_path and Path(training_corpus_path).exists():
            self._load_training_corpus()

    def _load_training_corpus(self) -> None:
        """Load and index training corpus for contamination detection."""
        try:
            logger.info(f"Loading training corpus from {self.training_corpus_path}")

            if not self.training_corpus_path:
                return

            corpus_path = Path(self.training_corpus_path)
            if corpus_path.suffix == '.json':
                with open(corpus_path) as f:
                    corpus_data = json.load(f)
            elif corpus_path.suffix == '.jsonl':
                corpus_data = []
                with open(corpus_path) as f:
                    for line in f:
                        corpus_data.append(json.loads(line))
            else:
                logger.warning(f"Unsupported corpus format: {corpus_path.suffix}")
                return

            # Index the corpus
            for item in corpus_data:
                if isinstance(item, dict) and 'code' in item:
                    self._index_code_sample(item['code'], item.get('id', ''))
                elif isinstance(item, str):
                    self._index_code_sample(item, '')

            logger.info(f"Indexed {len(self.exact_hashes)} exact hashes and {len(self.fuzzy_hashes)} fuzzy hashes")

        except Exception as e:
            logger.error(f"Failed to load training corpus: {e}")

    def _index_code_sample(self, code: str, sample_id: str) -> None:
        """Index a code sample for contamination detection.

        Args:
            code: Code to index
            sample_id: Identifier for the sample
        """
        # Exact hash
        exact_hash = hashlib.sha256(code.encode()).hexdigest()
        self.exact_hashes.add(exact_hash)

        # Normalized hash (remove whitespace, comments)
        normalized_code = self._normalize_code(code)
        fuzzy_hash = hashlib.sha256(normalized_code.encode()).hexdigest()
        self.fuzzy_hashes[fuzzy_hash] = sample_id

        # AST pattern (simplified)
        try:
            ast_pattern = self._extract_ast_pattern(code)
            if ast_pattern:
                pattern_hash = hashlib.sha256(str(ast_pattern).encode()).hexdigest()
                self.ast_patterns[pattern_hash] = sample_id
        except:
            pass  # Skip if AST parsing fails

    def _normalize_code(self, code: str) -> str:
        """Normalize code for fuzzy matching.

        Args:
            code: Input code

        Returns:
            Normalized code
        """
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)

        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)

        # Remove leading/trailing whitespace
        code = code.strip()

        return code

    def _extract_ast_pattern(self, code: str) -> Optional[Dict[str, Any]]:
        """Extract AST pattern from code.

        Args:
            code: Input code

        Returns:
            AST pattern dictionary
        """
        try:
            import ast

            tree = ast.parse(code)
            pattern = {
                'functions': [],
                'classes': [],
                'imports': [],
                'control_flow': []
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    pattern['functions'].append({
                        'name': node.name,
                        'args': len(node.args.args),
                        'returns': isinstance(node.returns, ast.Name) if node.returns else False
                    })
                elif isinstance(node, ast.ClassDef):
                    pattern['classes'].append({
                        'name': node.name,
                        'bases': len(node.bases),
                        'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            pattern['imports'].append(alias.name)
                    else:
                        pattern['imports'].append(node.module or '')
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    pattern['control_flow'].append(type(node).__name__)

            return pattern

        except Exception as e:
            logger.debug(f"AST extraction failed: {e}")
            return None

    def detect_exact_match(self, code: str) -> ContaminationResult:
        """Detect exact string matches.

        Args:
            code: Code to check

        Returns:
            Contamination result
        """
        exact_hash = hashlib.sha256(code.encode()).hexdigest()
        is_contaminated = exact_hash in self.exact_hashes

        return ContaminationResult(
            is_contaminated=is_contaminated,
            contamination_type=ContaminationType.EXACT_MATCH,
            confidence_score=1.0 if is_contaminated else 0.0,
            similarity_metrics={'exact_hash_match': 1.0 if is_contaminated else 0.0}
        )

    def detect_near_duplicate(self, code: str, threshold: float = 0.85) -> ContaminationResult:
        """Detect near-duplicate matches using fuzzy hashing.

        Args:
            code: Code to check
            threshold: Similarity threshold

        Returns:
            Contamination result
        """
        normalized_code = self._normalize_code(code)
        fuzzy_hash = hashlib.sha256(normalized_code.encode()).hexdigest()

        # Check exact normalized match
        if fuzzy_hash in self.fuzzy_hashes:
            return ContaminationResult(
                is_contaminated=True,
                contamination_type=ContaminationType.NEAR_DUPLICATE,
                confidence_score=1.0,
                source_match=self.fuzzy_hashes[fuzzy_hash],
                similarity_metrics={'normalized_match': 1.0}
            )

        # Check fuzzy similarity with all normalized codes
        max_similarity = 0.0
        best_match = None

        for stored_hash, sample_id in list(self.fuzzy_hashes.items())[:100]:  # Limit for performance
            # Use a simplified similarity metric
            similarity = self._calculate_string_similarity(fuzzy_hash, stored_hash)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = sample_id

        is_contaminated = max_similarity >= threshold

        return ContaminationResult(
            is_contaminated=is_contaminated,
            contamination_type=ContaminationType.NEAR_DUPLICATE,
            confidence_score=max_similarity,
            source_match=best_match if is_contaminated else None,
            similarity_metrics={'fuzzy_similarity': max_similarity}
        )

    def detect_structural_similarity(self, code: str, threshold: float = 0.8) -> ContaminationResult:
        """Detect structural similarity using AST patterns.

        Args:
            code: Code to check
            threshold: Similarity threshold

        Returns:
            Contamination result
        """
        ast_pattern = self._extract_ast_pattern(code)
        if not ast_pattern:
            return ContaminationResult(
                is_contaminated=False,
                contamination_type=ContaminationType.STRUCTURAL_SIMILARITY,
                confidence_score=0.0,
                similarity_metrics={'ast_extraction_failed': True}
            )

        pattern_hash = hashlib.sha256(str(ast_pattern).encode()).hexdigest()

        # Check exact pattern match
        if pattern_hash in self.ast_patterns:
            return ContaminationResult(
                is_contaminated=True,
                contamination_type=ContaminationType.STRUCTURAL_SIMILARITY,
                confidence_score=1.0,
                source_match=self.ast_patterns[pattern_hash],
                similarity_metrics={'ast_pattern_match': 1.0}
            )

        # Check pattern similarity (simplified)
        max_similarity = 0.0
        best_match = None

        # For performance, we'd implement a more sophisticated AST similarity metric
        # For now, using a simplified approach

        return ContaminationResult(
            is_contaminated=max_similarity >= threshold,
            contamination_type=ContaminationType.STRUCTURAL_SIMILARITY,
            confidence_score=max_similarity,
            source_match=best_match if max_similarity >= threshold else None,
            similarity_metrics={'structural_similarity': max_similarity}
        )

    def detect_variable_renaming(self, code: str, threshold: float = 0.9) -> ContaminationResult:
        """Detect contamination via variable renaming.

        Args:
            code: Code to check
            threshold: Similarity threshold

        Returns:
            Contamination result
        """
        # Extract variable names and replace with placeholders
        variable_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        variables = set(re.findall(variable_pattern, code))

        # Create template by replacing variables with placeholders
        template_code = code
        for i, var in enumerate(sorted(variables)):
            if not var.isupper() and var not in ['def', 'class', 'if', 'else', 'for', 'while', 'try', 'except']:
                template_code = re.sub(rf'\b{re.escape(var)}\b', f'VAR_{i}', template_code)

        template_hash = hashlib.sha256(template_code.encode()).hexdigest()

        # This would require a more sophisticated implementation with a template database
        # For now, return a basic result

        return ContaminationResult(
            is_contaminated=False,  # Simplified for now
            contamination_type=ContaminationType.VARIABLE_RENAMING,
            confidence_score=0.0,
            detection_metadata={'template_hash': template_hash}
        )

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0 and 1
        """
        if str1 == str2:
            return 1.0

        # Use SequenceMatcher for basic similarity
        matcher = difflib.SequenceMatcher(None, str1, str2)
        return matcher.ratio()

    async def detect_comprehensive(self, code: str) -> List[ContaminationResult]:
        """Run comprehensive contamination detection.

        Args:
            code: Code to check

        Returns:
            List of contamination results from different detectors
        """
        start_time = time.time()

        # Run all detection methods
        tasks = [
            asyncio.create_task(asyncio.to_thread(self.detect_exact_match, code)),
            asyncio.create_task(asyncio.to_thread(self.detect_near_duplicate, code)),
            asyncio.create_task(asyncio.to_thread(self.detect_structural_similarity, code)),
            asyncio.create_task(asyncio.to_thread(self.detect_variable_renaming, code))
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and add timing metadata
        valid_results = []
        for result in results:
            if isinstance(result, ContaminationResult):
                result.detection_metadata['detection_time_ms'] = (time.time() - start_time) * 1000
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Contamination detection failed: {result}")

        return valid_results

    def get_contamination_summary(self, results: List[ContaminationResult]) -> Dict[str, Any]:
        """Summarize contamination detection results.

        Args:
            results: List of contamination results

        Returns:
            Summary dictionary
        """
        summary = {
            'is_contaminated': False,
            'contamination_types': [],
            'max_confidence': 0.0,
            'detection_count': len(results),
            'positive_detections': 0,
            'detection_breakdown': {}
        }

        for result in results:
            summary['detection_breakdown'][result.contamination_type.value] = {
                'is_contaminated': result.is_contaminated,
                'confidence': result.confidence_score,
                'source_match': result.source_match
            }

            if result.is_contaminated:
                summary['is_contaminated'] = True
                summary['contamination_types'].append(result.contamination_type.value)
                summary['positive_detections'] += 1
                summary['max_confidence'] = max(summary['max_confidence'], result.confidence_score)

        return summary

    def save_detector_state(self, filepath: str) -> None:
        """Save detector state to file.

        Args:
            filepath: Path to save state
        """
        try:
            state = {
                'exact_hashes': list(self.exact_hashes),
                'fuzzy_hashes': self.fuzzy_hashes,
                'ast_patterns': self.ast_patterns,
                'training_corpus_path': self.training_corpus_path
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            logger.info(f"Detector state saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save detector state: {e}")

    def load_detector_state(self, filepath: str) -> None:
        """Load detector state from file.

        Args:
            filepath: Path to load state from
        """
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.exact_hashes = set(state.get('exact_hashes', []))
            self.fuzzy_hashes = state.get('fuzzy_hashes', {})
            self.ast_patterns = state.get('ast_patterns', {})
            self.training_corpus_path = state.get('training_corpus_path')

            logger.info(f"Detector state loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load detector state: {e}")


def create_contamination_detector(training_corpus_path: Optional[str] = None) -> ContaminationDetector:
    """Create a contamination detector instance.

    Args:
        training_corpus_path: Path to training corpus

    Returns:
        Contamination detector instance
    """
    return ContaminationDetector(training_corpus_path)


# Example usage for evaluation
async def evaluate_contamination_detection():
    """Example evaluation of contamination detection."""
    detector = create_contamination_detector()

    # Test cases
    test_codes = [
        "def add(a, b): return a + b",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "import numpy as np\ndef matrix_multiply(A, B):\n    return np.dot(A, B)"
    ]

    for i, code in enumerate(test_codes):
        print(f"\nTest Case {i+1}:")
        print(f"Code: {code}")

        results = await detector.detect_comprehensive(code)
        summary = detector.get_contamination_summary(results)

        print(f"Contaminated: {summary['is_contaminated']}")
        print(f"Max Confidence: {summary['max_confidence']:.3f}")
        print(f"Types: {summary['contamination_types']}")


if __name__ == "__main__":
    asyncio.run(evaluate_contamination_detection())
