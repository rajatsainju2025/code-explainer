"""Comprehensive evaluation framework for ICML-standard experiments.

This module implements rigorous experimental protocols including:
- Statistical significance testing
- Cross-validation with proper splits  
- Human evaluation interfaces
- Comprehensive metric computation
- Error analysis and categorization
"""

import json
import logging
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from .metrics.evaluate import (
    BERTScoreMetric,
    BLEUMetric,
    CodeBLEUMetric,
    ROUGEMetric,
)

logger = logging.getLogger(__name__)


class ICMLEvaluationFramework:
    """Comprehensive evaluation framework for ICML submission."""
    
    def __init__(self, results_dir: str = "evaluation_results"):
        """Initialize the evaluation framework.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "bleu": BLEUMetric(),
            "rouge": ROUGEMetric(),
            "bertscore": BERTScoreMetric(),
            "codebleu": CodeBLEUMetric(),
        }
        
        # Results storage
        self.results = defaultdict(list)
        self.human_evaluations = []
        
    def evaluate_system(
        self,
        system_name: str,
        predictions: List[str],
        references: List[str],
        codes: List[str],
        metadata: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Evaluate a system with comprehensive metrics.
        
        Args:
            system_name: Name of the system being evaluated
            predictions: Generated explanations
            references: Reference explanations  
            codes: Source code snippets
            metadata: Additional metadata for analysis
            
        Returns:
            Dictionary of evaluation scores
        """
        logger.info(f"Evaluating system: {system_name}")
        
        # Compute automatic metrics
        scores = {}
        for metric_name, metric in self.metrics.items():
            try:
                if metric_name == "codebleu":
                    score = metric.compute(predictions, references, codes)
                else:
                    score = metric.compute(predictions, references)
                
                if isinstance(score, dict):
                    # Handle metrics that return multiple scores
                    for sub_metric, value in score.items():
                        scores[f"{metric_name}_{sub_metric}"] = value
                else:
                    scores[metric_name] = score
                    
            except Exception as e:
                logger.warning(f"Failed to compute {metric_name}: {e}")
                scores[metric_name] = 0.0
        
        # Store results
        result_entry = {
            "system": system_name,
            "scores": scores,
            "metadata": metadata or {},
            "n_examples": len(predictions)
        }
        self.results[system_name].append(result_entry)
        
        # Save intermediate results
        self._save_results()
        
        return scores
    
    def compare_systems(
        self,
        system_a: str,
        system_b: str,
        metric: str = "bleu"
    ) -> Dict[str, Any]:
        """Perform statistical significance testing between two systems.
        
        Args:
            system_a: Name of first system
            system_b: Name of second system  
            metric: Metric to compare
            
        Returns:
            Statistical comparison results
        """
        if system_a not in self.results or system_b not in self.results:
            raise ValueError(f"Systems {system_a} and/or {system_b} not found in results")
        
        scores_a = [r["scores"].get(metric, 0.0) for r in self.results[system_a]]
        scores_b = [r["scores"].get(metric, 0.0) for r in self.results[system_b]]
        
        # Paired t-test
        if len(scores_a) == len(scores_b) and len(scores_a) > 1:
            t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        else:
            t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((np.std(scores_a) ** 2) + (np.std(scores_b) ** 2)) / 2)
        cohens_d = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std if pooled_std > 0 else 0
        
        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            sample_a = np.random.choice(scores_a, len(scores_a), replace=True)
            sample_b = np.random.choice(scores_b, len(scores_b), replace=True)
            bootstrap_diffs.append(np.mean(sample_a) - np.mean(sample_b))
        
        ci_lower, ci_upper = np.percentile(bootstrap_diffs, [2.5, 97.5])
        
        return {
            "system_a": system_a,
            "system_b": system_b,
            "metric": metric,
            "mean_a": np.mean(scores_a),
            "mean_b": np.mean(scores_b),
            "difference": np.mean(scores_a) - np.mean(scores_b),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "cohens_d": cohens_d,
            "confidence_interval": (ci_lower, ci_upper)
        }
    
    def run_human_evaluation(
        self,
        predictions: List[str],
        references: List[str],
        codes: List[str],
        sample_size: int = 100,
        evaluator_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run human evaluation protocol.
        
        Args:
            predictions: System predictions to evaluate
            references: Reference explanations
            codes: Source codes
            sample_size: Number of examples to evaluate
            evaluator_ids: List of evaluator identifiers
            
        Returns:
            Human evaluation results
        """
        # Sample examples for evaluation
        indices = random.sample(range(len(predictions)), min(sample_size, len(predictions)))
        
        evaluation_data = []
        for i, idx in enumerate(indices):
            evaluation_data.append({
                "id": i,
                "code": codes[idx],
                "prediction": predictions[idx],
                "reference": references[idx]
            })
        
        # Save evaluation data for human annotators
        eval_file = self.results_dir / "human_evaluation_data.json"
        with open(eval_file, "w") as f:
            json.dump(evaluation_data, f, indent=2)
        
        logger.info(f"Human evaluation data saved to {eval_file}")
        logger.info(f"Please have evaluators rate explanations on:")
        logger.info("- Accuracy (1-5): Technical correctness")
        logger.info("- Clarity (1-5): Understandability")  
        logger.info("- Completeness (1-5): Coverage of important aspects")
        logger.info("- Overall (1-5): Overall quality")
        
        return {
            "evaluation_file": str(eval_file),
            "n_examples": len(evaluation_data),
            "instructions": {
                "accuracy": "Rate how technically accurate the explanation is (1=incorrect, 5=perfectly accurate)",
                "clarity": "Rate how clear and understandable the explanation is (1=confusing, 5=very clear)",
                "completeness": "Rate how complete the explanation is (1=missing key info, 5=comprehensive)",
                "overall": "Rate the overall quality of the explanation (1=poor, 5=excellent)"
            }
        }
    
    def analyze_human_evaluation(self, human_scores_file: str) -> Dict[str, Any]:
        """Analyze human evaluation results.
        
        Args:
            human_scores_file: Path to file with human evaluation scores
            
        Returns:
            Analysis of human evaluation results
        """
        with open(human_scores_file) as f:
            human_data = json.load(f)
        
        # Extract scores by dimension
        dimensions = ["accuracy", "clarity", "completeness", "overall"]
        scores_by_dim = {dim: [] for dim in dimensions}
        evaluator_scores = defaultdict(lambda: defaultdict(list))
        
        for entry in human_data:
            for dim in dimensions:
                if dim in entry:
                    scores_by_dim[dim].append(entry[dim])
                    if "evaluator_id" in entry:
                        evaluator_scores[entry["evaluator_id"]][dim].append(entry[dim])
        
        # Calculate statistics
        results = {}
        for dim in dimensions:
            scores = scores_by_dim[dim]
            if scores:
                results[f"{dim}_mean"] = np.mean(scores)
                results[f"{dim}_std"] = np.std(scores)
                results[f"{dim}_median"] = np.median(scores)
                results[f"{dim}_min"] = np.min(scores)
                results[f"{dim}_max"] = np.max(scores)
        
        # Inter-annotator agreement (if multiple evaluators)
        if len(evaluator_scores) > 1:
            kappa_scores = {}
            for dim in dimensions:
                evaluator_ratings = []
                for evaluator in evaluator_scores:
                    if dim in evaluator_scores[evaluator]:
                        evaluator_ratings.append(evaluator_scores[evaluator][dim])
                
                if len(evaluator_ratings) >= 2 and len(evaluator_ratings[0]) > 0:
                    # Calculate pairwise kappa
                    kappas = []
                    for i in range(len(evaluator_ratings)):
                        for j in range(i + 1, len(evaluator_ratings)):
                            if len(evaluator_ratings[i]) == len(evaluator_ratings[j]):
                                kappa = cohen_kappa_score(evaluator_ratings[i], evaluator_ratings[j])
                                kappas.append(kappa)
                    
                    if kappas:
                        kappa_scores[f"{dim}_kappa"] = np.mean(kappas)
            
            results.update(kappa_scores)
        
        # Save analysis
        analysis_file = self.results_dir / "human_evaluation_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_error_analysis(
        self,
        system_name: str,
        predictions: List[str],
        references: List[str],
        codes: List[str],
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform systematic error analysis.
        
        Args:
            system_name: Name of system to analyze
            predictions: System predictions
            references: Reference explanations
            codes: Source code snippets
            categories: Error categories to analyze
            
        Returns:
            Error analysis results
        """
        if categories is None:
            categories = [
                "context_misinterpretation",
                "complexity_underestimation", 
                "domain_knowledge_gaps",
                "syntax_errors",
                "semantic_errors"
            ]
        
        # Compute quality scores for error correlation
        bleu_scores = self.metrics["bleu"].compute_per_example(predictions, references)
        
        # Categorize errors (this would be enhanced with ML-based classification)
        error_analysis = {
            "total_examples": len(predictions),
            "low_quality_threshold": 0.2,  # BLEU threshold for "errors"
            "error_categories": {},
            "examples_by_category": defaultdict(list)
        }
        
        # Identify low-quality examples
        low_quality_indices = [i for i, score in enumerate(bleu_scores) if score < 0.2]
        error_analysis["n_errors"] = len(low_quality_indices)
        error_analysis["error_rate"] = len(low_quality_indices) / len(predictions)
        
        # Sample errors for manual analysis
        sample_size = min(50, len(low_quality_indices))
        if low_quality_indices:
            sampled_errors = random.sample(low_quality_indices, sample_size)
            
            error_examples = []
            for idx in sampled_errors:
                error_examples.append({
                    "index": idx,
                    "code": codes[idx],
                    "prediction": predictions[idx],
                    "reference": references[idx],
                    "bleu_score": bleu_scores[idx]
                })
            
            # Save for manual categorization
            error_file = self.results_dir / f"{system_name}_error_analysis.json"
            with open(error_file, "w") as f:
                json.dump({
                    "error_examples": error_examples,
                    "categories": categories,
                    "instructions": "Manually categorize each error example"
                }, f, indent=2)
            
            error_analysis["error_examples_file"] = str(error_file)
        
        return error_analysis
    
    def generate_icml_results_table(self) -> str:
        """Generate LaTeX table of results for ICML paper.
        
        Returns:
            LaTeX table string
        """
        if not self.results:
            return "No results available"
        
        # Aggregate results by system
        system_scores = {}
        for system_name, results_list in self.results.items():
            # Average across all evaluations
            all_scores = defaultdict(list)
            for result in results_list:
                for metric, score in result["scores"].items():
                    all_scores[metric].append(score)
            
            system_scores[system_name] = {
                metric: np.mean(scores) for metric, scores in all_scores.items()
            }
        
        # Generate LaTeX table
        latex = "\\begin{table}[t]\n"
        latex += "\\centering\n"
        latex += "\\caption{Experimental Results on Code Explanation Benchmarks}\n"
        latex += "\\label{tab:results}\n"
        latex += "\\begin{tabular}{l|cccc|c}\n"
        latex += "\\toprule\n"
        latex += "Method & BLEU-4 & ROUGE-L & BERTScore & CodeBLEU & Human Pref \\\\\n"
        latex += "\\midrule\n"
        
        # Sort systems by overall performance
        sorted_systems = sorted(
            system_scores.items(),
            key=lambda x: x[1].get("bleu", 0),
            reverse=True
        )
        
        for system_name, scores in sorted_systems:
            bleu = scores.get("bleu", 0.0)
            rouge = scores.get("rouge_l", scores.get("rouge", 0.0))
            bert = scores.get("bertscore_f1", scores.get("bertscore", 0.0))
            codebleu = scores.get("codebleu", 0.0)
            human = scores.get("human_preference", 0.0)
            
            # Bold the best scores
            if system_name == sorted_systems[0][0]:
                latex += f"\\textbf{{{system_name}}} & \\textbf{{{bleu:.1f}}} & \\textbf{{{rouge:.1f}}} & \\textbf{{{bert:.1f}}} & \\textbf{{{codebleu:.1f}}} & \\textbf{{{human:.1f}\\%}} \\\\\n"
            else:
                latex += f"{system_name} & {bleu:.1f} & {rouge:.1f} & {bert:.1f} & {codebleu:.1f} & {human:.1f}\\% \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        # Save to file
        table_file = self.results_dir / "icml_results_table.tex"
        with open(table_file, "w") as f:
            f.write(latex)
        
        return latex
    
    def _save_results(self):
        """Save current results to disk."""
        results_file = self.results_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(dict(self.results), f, indent=2)
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export results to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with all evaluation results
        """
        rows = []
        for system_name, results_list in self.results.items():
            for result in results_list:
                row = {"system": system_name}
                row.update(result["scores"])
                row.update(result["metadata"])
                rows.append(row)
        
        return pd.DataFrame(rows)


class CrossValidationEvaluator:
    """K-fold cross-validation for robust evaluation."""
    
    def __init__(self, k: int = 5, random_seed: int = 42):
        """Initialize cross-validation evaluator.
        
        Args:
            k: Number of folds
            random_seed: Random seed for reproducibility
        """
        self.k = k
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def create_folds(self, data: List[Any]) -> List[Tuple[List[int], List[int]]]:
        """Create k-fold splits.
        
        Args:
            data: List of data items
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n = len(data)
        indices = list(range(n))
        random.shuffle(indices)
        
        fold_size = n // self.k
        folds = []
        
        for i in range(self.k):
            start = i * fold_size
            end = start + fold_size if i < self.k - 1 else n
            
            test_indices = indices[start:end]
            train_indices = indices[:start] + indices[end:]
            
            folds.append((train_indices, test_indices))
        
        return folds
    
    def evaluate_with_cv(
        self,
        model_func,
        data: List[Dict[str, Any]],
        evaluation_func
    ) -> Dict[str, Any]:
        """Run cross-validation evaluation.
        
        Args:
            model_func: Function that trains and returns a model
            data: List of data examples
            evaluation_func: Function that evaluates model performance
            
        Returns:
            Cross-validation results
        """
        folds = self.create_folds(data)
        fold_results = []
        
        for fold_idx, (train_indices, test_indices) in enumerate(folds):
            logger.info(f"Running fold {fold_idx + 1}/{self.k}")
            
            # Split data
            train_data = [data[i] for i in train_indices]
            test_data = [data[i] for i in test_indices]
            
            # Train model
            model = model_func(train_data)
            
            # Evaluate
            results = evaluation_func(model, test_data)
            results["fold"] = fold_idx
            fold_results.append(results)
        
        # Aggregate results
        metrics = set()
        for result in fold_results:
            metrics.update(result.keys())
        metrics.discard("fold")
        
        cv_results = {}
        for metric in metrics:
            scores = [result[metric] for result in fold_results if metric in result]
            if scores:
                cv_results[f"{metric}_mean"] = np.mean(scores)
                cv_results[f"{metric}_std"] = np.std(scores)
                cv_results[f"{metric}_scores"] = scores
        
        return cv_results
