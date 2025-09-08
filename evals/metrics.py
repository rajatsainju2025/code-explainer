"""
Comprehensive metrics calculation for code explanation evaluation.

Supports accuracy, retrieval, latency, cost, and language-specific metrics
with statistical analysis and confidence intervals.
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Optional imports with fallbacks
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer
    import sacrebleu
    HAS_ML_METRICS = True
except ImportError:
    HAS_ML_METRICS = False
    sentence_bleu = None
    rouge_scorer = None
    logging.warning("ML metrics libraries not available. Install with: pip install scikit-learn nltk rouge-score sacrebleu")


@dataclass
class EvalResults:
    """Container for evaluation results with metadata."""
    
    # Core metrics
    accuracy: float = 0.0
    bleu_score: float = 0.0
    rouge_l: float = 0.0
    
    # Performance metrics
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    total_cost: float = 0.0
    
    # Retrieval metrics (if applicable)
    retrieval_precision: float = 0.0
    retrieval_recall: float = 0.0
    retrieval_ndcg: float = 0.0
    
    # Statistical metrics
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    effect_sizes: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    num_samples: int = 0
    config_hash: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            'accuracy': self.accuracy,
            'bleu_score': self.bleu_score,
            'rouge_l': self.rouge_l,
            'avg_latency': self.avg_latency,
            'p95_latency': self.p95_latency,
            'total_cost': self.total_cost,
            'retrieval_precision': self.retrieval_precision,
            'retrieval_recall': self.retrieval_recall,
            'retrieval_ndcg': self.retrieval_ndcg,
            'confidence_intervals': self.confidence_intervals,
            'p_values': self.p_values,
            'effect_sizes': self.effect_sizes,
            'num_samples': self.num_samples,
            'config_hash': self.config_hash,
            'timestamp': self.timestamp
        }
    
    def save(self, output_path: Path) -> None:
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for code explanation evaluation.
    
    Features:
    - Accuracy metrics for classification tasks
    - Text similarity metrics (BLEU, ROUGE)
    - Performance metrics (latency, cost)
    - Retrieval metrics (precision, recall, NDCG)
    - Statistical analysis with confidence intervals
    """
    
    def __init__(self, bootstrap_samples: int = 1000, confidence_level: float = 0.95):
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.rouge_scorer = None
        
        if HAS_ML_METRICS and rouge_scorer:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        latencies: List[float],
        costs: List[float],
        retrieval_results: Optional[List[Dict]] = None
    ) -> EvalResults:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: List of predicted explanations
            references: List of reference explanations
            latencies: List of response latencies in seconds
            costs: List of inference costs
            retrieval_results: Optional retrieval evaluation data
            
        Returns:
            EvalResults object with all computed metrics
        """
        results = EvalResults()
        results.num_samples = len(predictions)
        results.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Text similarity metrics
        if HAS_ML_METRICS:
            results.bleu_score = self._calculate_bleu(predictions, references)
            results.rouge_l = self._calculate_rouge_l(predictions, references)
        
        # Performance metrics
        results.avg_latency = float(np.mean(latencies))
        results.p95_latency = float(np.percentile(latencies, 95))
        results.total_cost = float(np.sum(costs))
        
        # Retrieval metrics
        if retrieval_results:
            retrieval_metrics = self._calculate_retrieval_metrics(retrieval_results)
            results.retrieval_precision = retrieval_metrics['precision']
            results.retrieval_recall = retrieval_metrics['recall']
            results.retrieval_ndcg = retrieval_metrics['ndcg']
        
        # Statistical analysis
        if self.bootstrap_samples > 0:
            results.confidence_intervals = self._calculate_confidence_intervals(
                predictions, references, latencies
            )
        
        return results
    
    def _calculate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score for text similarity."""
        if not HAS_ML_METRICS or sentence_bleu is None:
            return 0.0
        
        scores = []
        for pred, ref in zip(predictions, references):
            try:
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]  # BLEU expects list of reference lists
                score = sentence_bleu(ref_tokens, pred_tokens)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        return float(np.mean(scores))
    
    def _calculate_rouge_l(self, predictions: List[str], references: List[str]) -> float:
        """Calculate ROUGE-L score for text similarity."""
        if not self.rouge_scorer:
            return 0.0
        
        scores = []
        for pred, ref in zip(predictions, references):
            try:
                score = self.rouge_scorer.score(ref, pred)
                scores.append(score['rougeL'].fmeasure)
            except Exception:
                scores.append(0.0)
        
        return float(np.mean(scores))
    
    def _calculate_retrieval_metrics(self, retrieval_results: List[Dict]) -> Dict[str, float]:
        """
        Calculate retrieval metrics (precision@k, recall@k, NDCG@k).
        
        Expected format for retrieval_results:
        [
            {
                'retrieved_docs': [{'doc_id': str, 'score': float, 'relevant': bool}],
                'total_relevant': int
            }
        ]
        """
        precisions = []
        recalls = []
        ndcgs = []
        
        for result in retrieval_results:
            docs = result['retrieved_docs']
            total_relevant = result['total_relevant']
            
            # Calculate precision@k and recall@k
            relevant_retrieved = sum(1 for doc in docs if doc.get('relevant', False))
            precision = relevant_retrieved / len(docs) if docs else 0.0
            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            
            # Calculate NDCG@k
            ndcg = self._calculate_ndcg(docs)
            ndcgs.append(ndcg)
        
        return {
            'precision': float(np.mean(precisions)),
            'recall': float(np.mean(recalls)),
            'ndcg': float(np.mean(ndcgs))
        }
    
    def _calculate_ndcg(self, docs: List[Dict]) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not docs:
            return 0.0
        
        # DCG calculation
        dcg = 0.0
        for i, doc in enumerate(docs):
            relevance = 1.0 if doc.get('relevant', False) else 0.0
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Ideal DCG (assume all relevant docs at top)
        num_relevant = sum(1 for doc in docs if doc.get('relevant', False))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_confidence_intervals(
        self,
        predictions: List[str],
        references: List[str],
        latencies: List[float]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for key metrics."""
        n_samples = len(predictions)
        
        # Bootstrap sampling
        bleu_scores = []
        rouge_scores = []
        latency_means = []
        
        for _ in range(self.bootstrap_samples):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            sample_preds = [predictions[i] for i in indices]
            sample_refs = [references[i] for i in indices]
            sample_latencies = [latencies[i] for i in indices]
            
            # Calculate metrics for this sample
            if HAS_ML_METRICS:
                bleu = self._calculate_bleu(sample_preds, sample_refs)
                rouge = self._calculate_rouge_l(sample_preds, sample_refs)
                bleu_scores.append(bleu)
                rouge_scores.append(rouge)
            
            latency_means.append(np.mean(sample_latencies))
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        intervals = {}
        
        if bleu_scores:
            intervals['bleu_score'] = (
                np.percentile(bleu_scores, 100 * alpha / 2),
                np.percentile(bleu_scores, 100 * (1 - alpha / 2))
            )
        
        if rouge_scores:
            intervals['rouge_l'] = (
                np.percentile(rouge_scores, 100 * alpha / 2),
                np.percentile(rouge_scores, 100 * (1 - alpha / 2))
            )
        
        intervals['avg_latency'] = (
            np.percentile(latency_means, 100 * alpha / 2),
            np.percentile(latency_means, 100 * (1 - alpha / 2))
        )
        
        return intervals
    
    def compare_results(
        self,
        results_a: EvalResults,
        results_b: EvalResults,
        metric: str = 'bleu_score'
    ) -> Dict[str, float]:
        """
        Compare two evaluation results with statistical significance testing.
        
        Args:
            results_a: First evaluation results
            results_b: Second evaluation results
            metric: Metric to compare
            
        Returns:
            Dictionary with comparison statistics
        """
        # This would implement proper statistical tests
        # For now, returning placeholder
        return {
            'difference': getattr(results_b, metric) - getattr(results_a, metric),
            'p_value': 0.05,  # Placeholder
            'effect_size': 0.2,  # Placeholder
            'significant': True  # Placeholder
        }
