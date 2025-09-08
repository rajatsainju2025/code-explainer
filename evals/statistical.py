"""
Statistical analysis utilities for evaluation results.

Provides statistical significance testing, effect size calculation,
and confidence interval estimation for evaluation metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    from scipy.stats import ttest_ind, mannwhitneyu, bootstrap
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("SciPy not available for advanced statistical analysis")


class StatisticalAnalyzer:
    """
    Statistical analysis for evaluation results.
    
    Features:
    - Significance testing (t-test, Mann-Whitney U)
    - Effect size calculation (Cohen's d, Glass's delta)
    - Bootstrap confidence intervals
    - Multiple comparison correction
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def compare_metrics(
        self,
        group_a: List[float],
        group_b: List[float],
        metric_name: str = "metric"
    ) -> Dict[str, float]:
        """
        Compare two groups of metric values.
        
        Args:
            group_a: First group of values
            group_b: Second group of values
            metric_name: Name of the metric being compared
            
        Returns:
            Dictionary with comparison statistics
        """
        if not HAS_SCIPY:
            return self._basic_comparison(group_a, group_b)
        
        # Basic statistics
        mean_a, mean_b = np.mean(group_a), np.mean(group_b)
        std_a, std_b = np.std(group_a, ddof=1), np.std(group_b, ddof=1)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group_a) - 1) * std_a**2 + (len(group_b) - 1) * std_b**2) / 
                            (len(group_a) + len(group_b) - 2))
        cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0
        
        # Statistical tests
        try:
            # T-test (assumes normality)
            t_stat, t_pvalue = ttest_ind(group_a, group_b)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pvalue = mannwhitneyu(group_a, group_b, alternative='two-sided')
            
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")
            t_pvalue = u_pvalue = 1.0
        
        # Confidence interval for difference
        diff_ci = self._bootstrap_difference_ci(group_a, group_b)
        
        return {
            'mean_a': mean_a,
            'mean_b': mean_b,
            'difference': mean_b - mean_a,
            'percent_change': ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0.0,
            'cohens_d': cohens_d,
            'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
            't_test_pvalue': t_pvalue,
            'mann_whitney_pvalue': u_pvalue,
            'significant_t': t_pvalue < self.alpha,
            'significant_mw': u_pvalue < self.alpha,
            'difference_ci_lower': diff_ci[0],
            'difference_ci_upper': diff_ci[1]
        }
    
    def multiple_comparison_correction(
        self,
        p_values: List[float],
        method: str = 'bonferroni'
    ) -> List[float]:
        """
        Apply multiple comparison correction to p-values.
        
        Args:
            p_values: List of p-values to correct
            method: Correction method ('bonferroni', 'holm')
            
        Returns:
            List of corrected p-values
        """
        if not HAS_SCIPY:
            return p_values
        
        try:
            from statsmodels.stats.multitest import multipletests
            _, corrected_p, _, _ = multipletests(p_values, method=method)
            return corrected_p.tolist()
        except ImportError:
            # Simple Bonferroni correction
            if method == 'bonferroni':
                return [min(p * len(p_values), 1.0) for p in p_values]
            return p_values
    
    def bootstrap_confidence_interval(
        self,
        data: List[float],
        statistic_func: callable = np.mean,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a statistic.
        
        Args:
            data: Input data
            statistic_func: Function to calculate statistic
            confidence_level: Confidence level (0-1)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not data:
            return (0.0, 0.0)
        
        # Bootstrap sampling
        bootstrap_stats = []
        data_array = np.array(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data_array, size=len(data), replace=True)
            stat = statistic_func(sample)
            bootstrap_stats.append(stat)
        
        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_stats, lower_percentile)
        upper_bound = np.percentile(bootstrap_stats, upper_percentile)
        
        return (float(lower_bound), float(upper_bound))
    
    def _basic_comparison(self, group_a: List[float], group_b: List[float]) -> Dict[str, float]:
        """Basic comparison without scipy."""
        mean_a, mean_b = np.mean(group_a), np.mean(group_b)
        
        return {
            'mean_a': mean_a,
            'mean_b': mean_b,
            'difference': mean_b - mean_a,
            'percent_change': ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0.0,
            'cohens_d': 0.0,  # Placeholder
            'effect_size_interpretation': 'unknown',
            't_test_pvalue': 1.0,
            'mann_whitney_pvalue': 1.0,
            'significant_t': False,
            'significant_mw': False,
            'difference_ci_lower': 0.0,
            'difference_ci_upper': 0.0
        }
    
    def _bootstrap_difference_ci(
        self,
        group_a: List[float],
        group_b: List[float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for difference in means."""
        differences = []
        
        for _ in range(n_bootstrap):
            sample_a = np.random.choice(group_a, size=len(group_a), replace=True)
            sample_b = np.random.choice(group_b, size=len(group_b), replace=True)
            diff = np.mean(sample_b) - np.mean(sample_a)
            differences.append(diff)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(differences, lower_percentile)
        upper_bound = np.percentile(differences, upper_percentile)
        
        return (float(lower_bound), float(upper_bound))
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
