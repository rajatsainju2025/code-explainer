"""Data contamination detection for evaluation integrity.

This module implements state-of-the-art contamination detection methods to ensure
evaluation integrity and prevent data leakage between training and test sets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class ContaminationMatch:
    """A potential contamination match between train and test examples."""
    
    train_id: str
    test_id: str
    similarity_score: float
    match_type: str  # "exact", "substring", "ngram", "semantic"
    evidence: Dict[str, Any]
    confidence: float


@dataclass
class ContaminationReport:
    """Comprehensive contamination detection report."""
    
    total_test_examples: int
    total_train_examples: int
    contaminated_examples: List[ContaminationMatch]
    contamination_rate: float
    detection_methods: List[str]
    summary_stats: Dict[str, Any]


class ExactMatchDetector:
    """Detect exact matches between train and test data."""
    
    def __init__(self, normalize_whitespace: bool = True, case_sensitive: bool = False):
        self.normalize_whitespace = normalize_whitespace
        self.case_sensitive = case_sensitive
    
    def detect(
        self,
        train_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        fields_to_check: Optional[List[str]] = None
    ) -> List[ContaminationMatch]:
        """Detect exact matches between train and test data."""
        fields_to_check = fields_to_check or ["code", "explanation"]
        matches = []
        
        # Build hash index for train data
        train_hashes = {}
        for i, train_item in enumerate(train_data):
            for field in fields_to_check:
                if field in train_item:
                    content = self._normalize_content(train_item[field])
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    train_hashes[content_hash] = (i, field, content)
        
        # Check test data against train hashes
        for j, test_item in enumerate(test_data):
            test_id = test_item.get("id", f"test_{j}")
            
            for field in fields_to_check:
                if field in test_item:
                    content = self._normalize_content(test_item[field])
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    
                    if content_hash in train_hashes:
                        train_idx, train_field, train_content = train_hashes[content_hash]
                        train_id = train_data[train_idx].get("id", f"train_{train_idx}")
                        
                        matches.append(ContaminationMatch(
                            train_id=train_id,
                            test_id=test_id,
                            similarity_score=1.0,
                            match_type="exact",
                            evidence={
                                "field": field,
                                "train_field": train_field,
                                "content_hash": content_hash,
                                "content_length": len(content)
                            },
                            confidence=1.0
                        ))
        
        return matches
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison."""
        if not self.case_sensitive:
            content = content.lower()
        
        if self.normalize_whitespace:
            # Normalize whitespace and remove extra spaces
            content = re.sub(r'\s+', ' ', content.strip())
        
        return content


class NGramDetector:
    """Detect contamination using n-gram overlap analysis."""
    
    def __init__(
        self,
        ngram_sizes: Optional[List[int]] = None,
        min_overlap_ratio: float = 0.8,
        min_ngram_count: int = 5
    ):
        self.ngram_sizes = ngram_sizes or [4, 8, 16, 32]
        self.min_overlap_ratio = min_overlap_ratio
        self.min_ngram_count = min_ngram_count
    
    def detect(
        self,
        train_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        fields_to_check: Optional[List[str]] = None
    ) -> List[ContaminationMatch]:
        """Detect contamination using n-gram overlap."""
        fields_to_check = fields_to_check or ["code", "explanation"]
        matches = []
        
        for ngram_size in self.ngram_sizes:
            logger.info(f"Checking {ngram_size}-gram contamination...")
            
            # Build n-gram index for train data
            train_ngrams = defaultdict(list)
            for i, train_item in enumerate(train_data):
                for field in fields_to_check:
                    if field in train_item:
                        ngrams = self._extract_ngrams(train_item[field], ngram_size)
                        for ngram in ngrams:
                            train_ngrams[ngram].append((i, field))
            
            # Check test data for n-gram overlap
            for j, test_item in enumerate(test_data):
                test_id = test_item.get("id", f"test_{j}")
                
                for field in fields_to_check:
                    if field in test_item:
                        test_ngrams = self._extract_ngrams(test_item[field], ngram_size)
                        
                        if len(test_ngrams) < self.min_ngram_count:
                            continue
                        
                        # Find overlapping n-grams
                        overlapping_ngrams = []
                        for ngram in test_ngrams:
                            if ngram in train_ngrams:
                                overlapping_ngrams.extend(train_ngrams[ngram])
                        
                        if overlapping_ngrams:
                            # Group by train example
                            train_overlap_counts = defaultdict(int)
                            for train_idx, train_field in overlapping_ngrams:
                                train_overlap_counts[(train_idx, train_field)] += 1
                            
                            # Check if overlap ratio exceeds threshold
                            for (train_idx, train_field), overlap_count in train_overlap_counts.items():
                                overlap_ratio = overlap_count / len(test_ngrams)
                                
                                if overlap_ratio >= self.min_overlap_ratio:
                                    train_id = train_data[train_idx].get("id", f"train_{train_idx}")
                                    
                                    matches.append(ContaminationMatch(
                                        train_id=train_id,
                                        test_id=test_id,
                                        similarity_score=overlap_ratio,
                                        match_type="ngram",
                                        evidence={
                                            "ngram_size": ngram_size,
                                            "field": field,
                                            "train_field": train_field,
                                            "overlap_count": overlap_count,
                                            "total_ngrams": len(test_ngrams),
                                            "overlap_ratio": overlap_ratio
                                        },
                                        confidence=min(overlap_ratio, 1.0)
                                    ))
        
        return matches
    
    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """Extract character-level n-grams from text."""
        # Normalize text
        text = re.sub(r'\s+', ' ', text.strip().lower())
        
        # Extract n-grams
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i + n])
        
        return ngrams


class SubstringDetector:
    """Detect contamination using longest common substring analysis."""
    
    def __init__(self, min_substring_length: int = 50, min_similarity_ratio: float = 0.7):
        self.min_substring_length = min_substring_length
        self.min_similarity_ratio = min_similarity_ratio
    
    def detect(
        self,
        train_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        fields_to_check: Optional[List[str]] = None
    ) -> List[ContaminationMatch]:
        """Detect contamination using substring matching."""
        fields_to_check = fields_to_check or ["code", "explanation"]
        matches = []
        
        for j, test_item in enumerate(test_data):
            test_id = test_item.get("id", f"test_{j}")
            
            for field in fields_to_check:
                if field not in test_item:
                    continue
                
                test_content = test_item[field]
                
                for i, train_item in enumerate(train_data):
                    train_id = train_item.get("id", f"train_{i}")
                    
                    for train_field in fields_to_check:
                        if train_field not in train_item:
                            continue
                        
                        train_content = train_item[train_field]
                        
                        # Find longest common substring
                        similarity_ratio, longest_match = self._find_longest_common_substring(
                            test_content, train_content
                        )
                        
                        if (len(longest_match) >= self.min_substring_length and
                            similarity_ratio >= self.min_similarity_ratio):
                            
                            matches.append(ContaminationMatch(
                                train_id=train_id,
                                test_id=test_id,
                                similarity_score=similarity_ratio,
                                match_type="substring",
                                evidence={
                                    "field": field,
                                    "train_field": train_field,
                                    "longest_match": longest_match[:200] + "..." if len(longest_match) > 200 else longest_match,
                                    "match_length": len(longest_match),
                                    "test_length": len(test_content),
                                    "train_length": len(train_content),
                                    "similarity_ratio": similarity_ratio
                                },
                                confidence=similarity_ratio
                            ))
        
        return matches
    
    def _find_longest_common_substring(self, text1: str, text2: str) -> Tuple[float, str]:
        """Find longest common substring and similarity ratio."""
        matcher = SequenceMatcher(None, text1, text2)
        match = matcher.find_longest_match(0, len(text1), 0, len(text2))
        
        longest_match = text1[match.a:match.a + match.size]
        similarity_ratio = matcher.ratio()
        
        return similarity_ratio, longest_match


class SemanticSimilarityDetector:
    """Detect contamination using semantic similarity."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.9,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self._model = None
    
    def _get_model(self):
        """Lazy load sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers package required for semantic similarity detection")
        return self._model
    
    def detect(
        self,
        train_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        fields_to_check: Optional[List[str]] = None
    ) -> List[ContaminationMatch]:
        """Detect contamination using semantic similarity."""
        fields_to_check = fields_to_check or ["code", "explanation"]
        matches = []
        
        model = self._get_model()
        
        # Extract and encode train texts
        train_texts = []
        train_metadata = []
        
        for i, train_item in enumerate(train_data):
            for field in fields_to_check:
                if field in train_item and train_item[field]:
                    train_texts.append(train_item[field])
                    train_metadata.append((i, field))
        
        if not train_texts:
            return matches
        
        logger.info(f"Encoding {len(train_texts)} training examples...")
        train_embeddings = model.encode(train_texts, batch_size=self.batch_size, show_progress_bar=True)
        
        # Process test data in batches
        for batch_start in range(0, len(test_data), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(test_data))
            test_batch = test_data[batch_start:batch_end]
            
            test_texts = []
            test_metadata = []
            
            for j, test_item in enumerate(test_batch):
                for field in fields_to_check:
                    if field in test_item and test_item[field]:
                        test_texts.append(test_item[field])
                        test_metadata.append((batch_start + j, field))
            
            if not test_texts:
                continue
            
            # Encode test texts
            test_embeddings = model.encode(test_texts, batch_size=self.batch_size)
            
            # Compute similarities
            similarities = np.dot(test_embeddings, train_embeddings.T)
            
            # Find high similarity matches
            high_sim_indices = np.where(similarities >= self.similarity_threshold)
            
            for test_idx, train_idx in zip(high_sim_indices[0], high_sim_indices[1]):
                similarity_score = float(similarities[test_idx, train_idx])
                
                test_data_idx, test_field = test_metadata[test_idx]
                train_data_idx, train_field = train_metadata[train_idx]
                
                test_id = test_data[test_data_idx].get("id", f"test_{test_data_idx}")
                train_id = train_data[train_data_idx].get("id", f"train_{train_data_idx}")
                
                matches.append(ContaminationMatch(
                    train_id=train_id,
                    test_id=test_id,
                    similarity_score=similarity_score,
                    match_type="semantic",
                    evidence={
                        "field": test_field,
                        "train_field": train_field,
                        "cosine_similarity": similarity_score,
                        "model_name": self.model_name
                    },
                    confidence=similarity_score
                ))
        
        return matches


class ContaminationDetector:
    """Main contamination detection class combining multiple methods."""
    
    def __init__(self, detection_methods: Optional[List[str]] = None):
        self.detection_methods = detection_methods or ["exact", "ngram", "substring", "semantic"]
        
        # Initialize detectors
        self.detectors = {}
        if "exact" in self.detection_methods:
            self.detectors["exact"] = ExactMatchDetector()
        if "ngram" in self.detection_methods:
            self.detectors["ngram"] = NGramDetector()
        if "substring" in self.detection_methods:
            self.detectors["substring"] = SubstringDetector()
        if "semantic" in self.detection_methods:
            self.detectors["semantic"] = SemanticSimilarityDetector()
    
    def detect_contamination(
        self,
        train_file: Union[str, Path],
        test_file: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        fields_to_check: Optional[List[str]] = None
    ) -> ContaminationReport:
        """Run comprehensive contamination detection."""
        # Load data
        train_data = self._load_data(train_file)
        test_data = self._load_data(test_file)
        
        logger.info(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples")
        
        # Run all detection methods
        all_matches = []
        
        for method_name, detector in self.detectors.items():
            logger.info(f"Running {method_name} contamination detection...")
            matches = detector.detect(train_data, test_data, fields_to_check)
            logger.info(f"Found {len(matches)} potential contamination matches with {method_name}")
            
            all_matches.extend(matches)
        
        # Remove duplicates and merge overlapping matches
        unique_matches = self._deduplicate_matches(all_matches)
        
        # Calculate contamination rate
        contaminated_test_ids = set(match.test_id for match in unique_matches)
        contamination_rate = len(contaminated_test_ids) / len(test_data) if test_data else 0.0
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats(unique_matches, test_data, train_data)
        
        # Create report
        report = ContaminationReport(
            total_test_examples=len(test_data),
            total_train_examples=len(train_data),
            contaminated_examples=unique_matches,
            contamination_rate=contamination_rate,
            detection_methods=self.detection_methods,
            summary_stats=summary_stats
        )
        
        # Save report if output file specified
        if output_file:
            self._save_report(report, output_file)
        
        return report
    
    def _load_data(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load data from JSON or JSONL file."""
        data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            if content.startswith('['):
                # JSON array format
                data = json.loads(content)
            else:
                # JSONL format
                for line in content.split('\n'):
                    if line.strip():
                        data.append(json.loads(line))
        
        return data
    
    def _deduplicate_matches(self, matches: List[ContaminationMatch]) -> List[ContaminationMatch]:
        """Remove duplicate matches and merge overlapping ones."""
        # Group by test_id and train_id
        grouped_matches = defaultdict(list)
        for match in matches:
            key = (match.test_id, match.train_id)
            grouped_matches[key].append(match)
        
        # Keep the match with highest confidence for each pair
        unique_matches = []
        for match_group in grouped_matches.values():
            best_match = max(match_group, key=lambda m: m.confidence)
            
            # Merge evidence from all matches
            merged_evidence = {}
            for match in match_group:
                merged_evidence.update(match.evidence)
            
            best_match.evidence = merged_evidence
            unique_matches.append(best_match)
        
        return unique_matches
    
    def _generate_summary_stats(
        self,
        matches: List[ContaminationMatch],
        test_data: List[Dict[str, Any]],
        train_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not matches:
            return {
                "contamination_by_method": {},
                "contamination_by_field": {},
                "confidence_distribution": {},
                "similarity_distribution": {}
            }
        
        # Contamination by method
        contamination_by_method = defaultdict(int)
        for match in matches:
            contamination_by_method[match.match_type] += 1
        
        # Contamination by field
        contamination_by_field = defaultdict(int)
        for match in matches:
            field = match.evidence.get("field", "unknown")
            contamination_by_field[field] += 1
        
        # Confidence distribution
        confidences = [match.confidence for match in matches]
        confidence_dist = {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "median": float(np.median(confidences))
        }
        
        # Similarity distribution
        similarities = [match.similarity_score for match in matches]
        similarity_dist = {
            "mean": float(np.mean(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "median": float(np.median(similarities))
        }
        
        return {
            "contamination_by_method": dict(contamination_by_method),
            "contamination_by_field": dict(contamination_by_field),
            "confidence_distribution": confidence_dist,
            "similarity_distribution": similarity_dist
        }
    
    def _save_report(self, report: ContaminationReport, output_file: Union[str, Path]):
        """Save contamination report to file."""
        report_dict = {
            "total_test_examples": report.total_test_examples,
            "total_train_examples": report.total_train_examples,
            "contamination_rate": report.contamination_rate,
            "detection_methods": report.detection_methods,
            "summary_stats": report.summary_stats,
            "contaminated_examples": [
                {
                    "train_id": match.train_id,
                    "test_id": match.test_id,
                    "similarity_score": match.similarity_score,
                    "match_type": match.match_type,
                    "evidence": match.evidence,
                    "confidence": match.confidence
                }
                for match in report.contaminated_examples
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Contamination report saved to {output_file}")


def run_contamination_detection(
    train_file: Union[str, Path],
    test_file: Union[str, Path],
    output_file: Union[str, Path],
    methods: Optional[List[str]] = None,
    fields: Optional[List[str]] = None
) -> ContaminationReport:
    """Run contamination detection with specified methods."""
    detector = ContaminationDetector(methods)
    return detector.detect_contamination(train_file, test_file, output_file, fields)
