"""
Main evaluation runner for the Code Explainer system.

Provides unified interface for running evaluations, ablation studies,
and benchmark comparisons with reproducible results.
"""

import os
import json
import time
import hashlib
import logging
import traceback
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import asdict

from .config import EvalConfig, load_config, save_config
from .metrics import MetricsCalculator, EvalResults
from .datasets import DatasetLoader
from .statistical import StatisticalAnalyzer

# Optional imports with fallbacks
try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class EvalRunner:
    """
    Main evaluation runner with reproducibility and statistical analysis.
    
    Features:
    - Deterministic runs with seed control
    - Run manifest generation for provenance
    - Parallel evaluation support
    - Statistical analysis with confidence intervals
    - Flexible output formats (JSON, CSV, HTML)
    """
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.metrics_calculator = MetricsCalculator(
            bootstrap_samples=config.metrics.bootstrap_samples,
            confidence_level=config.metrics.confidence_level
        )
        self.dataset_loader = DatasetLoader()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Set reproducibility
        self._set_seeds(config.seed)
    
    def run_evaluation(self) -> EvalResults:
        """
        Run complete evaluation pipeline.
        
        Returns:
            EvalResults object with comprehensive metrics
        """
        logger.info(f"Starting evaluation: {self.config.name}")
        logger.info(f"Output directory: {self.output_dir}")
        
        start_time = time.time()
        
        try:
            # Load dataset
            dataset = self.dataset_loader.load_dataset(self.config.dataset)
            logger.info(f"Loaded dataset with {len(dataset)} samples")
            
            # Generate predictions
            predictions, latencies, costs = self._generate_predictions(dataset)
            
            # Calculate metrics
            results = self.metrics_calculator.calculate_all_metrics(
                predictions=predictions,
                references=[item['reference'] for item in dataset],
                latencies=latencies,
                costs=costs
            )
            
            # Add metadata
            results.config_hash = self._calculate_config_hash()
            results.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Save results
            self._save_results(results, predictions, dataset)
            
            # Generate run manifest
            self._generate_run_manifest(results, time.time() - start_time)
            
            logger.info(f"Evaluation completed in {time.time() - start_time:.2f}s")
            logger.info(f"Results saved to: {self.output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def run_ablation_study(self, components: List[str]) -> Dict[str, EvalResults]:
        """
        Run ablation study by systematically removing components.
        
        Args:
            components: List of components to ablate ('retrieval', 'reranking', etc.)
            
        Returns:
            Dictionary mapping component names to evaluation results
        """
        logger.info(f"Starting ablation study: {components}")
        
        results = {}
        
        # Baseline (all components)
        logger.info("Running baseline evaluation (all components)")
        baseline_results = self.run_evaluation()
        results['baseline'] = baseline_results
        
        # Ablate each component
        for component in components:
            logger.info(f"Ablating component: {component}")
            
            # Create modified config
            ablated_config = self._create_ablated_config(component)
            
            # Run evaluation with ablated config
            ablated_runner = EvalRunner(ablated_config)
            ablated_results = ablated_runner.run_evaluation()
            results[f'no_{component}'] = ablated_results
        
        # Save comparison report
        self._save_ablation_report(results)
        
        return results
    
    def compare_strategies(self, strategy_configs: List[EvalConfig]) -> Dict[str, EvalResults]:
        """
        Compare multiple evaluation strategies.
        
        Args:
            strategy_configs: List of evaluation configurations to compare
            
        Returns:
            Dictionary mapping strategy names to evaluation results
        """
        logger.info(f"Comparing {len(strategy_configs)} strategies")
        
        results = {}
        
        for config in strategy_configs:
            logger.info(f"Evaluating strategy: {config.name}")
            runner = EvalRunner(config)
            strategy_results = runner.run_evaluation()
            results[config.name] = strategy_results
        
        # Statistical comparison
        self._save_strategy_comparison(results)
        
        return results
    
    def _generate_predictions(self, dataset: List[Dict]) -> tuple[List[str], List[float], List[float]]:
        """
        Generate predictions for dataset samples.
        
        Args:
            dataset: List of dataset samples
            
        Returns:
            Tuple of (predictions, latencies, costs)
        """
        predictions = []
        latencies = []
        costs = []
        
        # Import the explainer here to avoid circular imports
        try:
            from src.code_explainer.main import CodeExplainer
            explainer = CodeExplainer()
        except ImportError:
            logger.warning("CodeExplainer not available, using mock predictions")
            explainer = None
        
        for i, sample in enumerate(dataset):
            if self.config.verbose and i % 10 == 0:
                logger.info(f"Processing sample {i+1}/{len(dataset)}")
            
            start_time = time.time()
            
            try:
                if explainer:
                    # Real prediction
                    result = explainer.explain_code(
                        code=sample['code'],
                        strategy=getattr(self.config, 'strategy', 'enhanced_rag')
                    )
                    prediction = result.get('explanation', '')
                    cost = result.get('cost', 0.0)
                else:
                    # Mock prediction for testing
                    prediction = f"Mock explanation for: {sample['code'][:50]}..."
                    cost = 0.01
                
                latency = time.time() - start_time
                
                predictions.append(prediction)
                latencies.append(latency)
                costs.append(cost)
                
            except Exception as e:
                logger.warning(f"Failed to generate prediction for sample {i}: {e}")
                predictions.append("")
                latencies.append(0.0)
                costs.append(0.0)
        
        return predictions, latencies, costs
    
    def _create_ablated_config(self, component: str) -> EvalConfig:
        """Create configuration with specified component disabled."""
        # Create a copy of the current config
        config_dict = asdict(self.config)
        
        # Modify based on component
        if component == 'retrieval':
            config_dict['retrieval']['enabled'] = False
        elif component == 'reranking':
            config_dict['retrieval']['rerank'] = False
        elif component == 'temperature':
            config_dict['model']['temperature'] = 0.0
        # Add more components as needed
        
        # Update name and output directory
        config_dict['name'] = f"{self.config.name}_no_{component}"
        config_dict['output_dir'] = f"{self.config.output_dir}_no_{component}"
        
        # Convert back to EvalConfig
        from .config import _dict_to_config
        return _dict_to_config(config_dict)
    
    def _calculate_config_hash(self) -> str:
        """Calculate hash of configuration for reproducibility tracking."""
        config_str = json.dumps(asdict(self.config), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _save_results(self, results: EvalResults, predictions: List[str], dataset: List[Dict]) -> None:
        """Save evaluation results and predictions."""
        # Save metrics
        if self.config.save_metrics:
            results.save(self.output_dir / "metrics.json")
        
        # Save predictions
        if self.config.save_predictions:
            predictions_data = []
            for i, (pred, sample) in enumerate(zip(predictions, dataset)):
                predictions_data.append({
                    'index': i,
                    'code': sample.get('code', ''),
                    'reference': sample.get('reference', ''),
                    'prediction': pred,
                    'metadata': sample.get('metadata', {})
                })
            
            with open(self.output_dir / "predictions.json", 'w') as f:
                json.dump(predictions_data, f, indent=2)
        
        # Save configuration
        save_config(self.config, self.output_dir / "config.yaml")
    
    def _generate_run_manifest(self, results: EvalResults, runtime: float) -> None:
        """Generate run manifest for provenance tracking."""
        manifest = {
            'run_id': f"eval_{int(time.time())}",
            'config_hash': results.config_hash,
            'timestamp': results.timestamp,
            'runtime_seconds': runtime,
            'git_info': self._get_git_info(),
            'environment': self._get_environment_info(),
            'results_summary': {
                'num_samples': results.num_samples,
                'avg_latency': results.avg_latency,
                'total_cost': results.total_cost
            }
        }
        
        with open(self.output_dir / "run_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
    
    def _save_ablation_report(self, results: Dict[str, EvalResults]) -> None:
        """Save ablation study comparison report."""
        baseline = results.get('baseline')
        if not baseline:
            return
        
        report = {
            'study_type': 'ablation',
            'baseline': baseline.to_dict(),
            'ablations': {}
        }
        
        for name, result in results.items():
            if name != 'baseline':
                report['ablations'][name] = {
                    'results': result.to_dict(),
                    'delta_bleu': result.bleu_score - baseline.bleu_score,
                    'delta_latency': result.avg_latency - baseline.avg_latency,
                    'delta_cost': result.total_cost - baseline.total_cost
                }
        
        with open(self.output_dir / "ablation_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def _save_strategy_comparison(self, results: Dict[str, EvalResults]) -> None:
        """Save strategy comparison report."""
        comparison_data = []
        
        for name, result in results.items():
            comparison_data.append({
                'strategy': name,
                'bleu_score': result.bleu_score,
                'rouge_l': result.rouge_l,
                'avg_latency': result.avg_latency,
                'total_cost': result.total_cost,
                'num_samples': result.num_samples
            })
        
        # Sort by BLEU score
        comparison_data.sort(key=lambda x: x['bleu_score'], reverse=True)
        
        with open(self.output_dir / "strategy_comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2)
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get git repository information."""
        try:
            import subprocess
            
            def run_git_cmd(cmd):
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                return result.stdout.strip() if result.returncode == 0 else "unknown"
            
            return {
                'commit': run_git_cmd(['git', 'rev-parse', 'HEAD']),
                'branch': run_git_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
                'dirty': str(run_git_cmd(['git', 'diff', '--quiet']) != ""),
                'remote': run_git_cmd(['git', 'config', '--get', 'remote.origin.url'])
            }
        except Exception:
            return {'error': 'git info unavailable'}
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get environment information."""
        import sys
        import platform
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'hostname': platform.node()
        }
        
        # Add package versions
        try:
            import torch
            env_info['torch_version'] = torch.__version__
        except ImportError:
            pass
        
        try:
            import transformers
            env_info['transformers_version'] = transformers.__version__
        except ImportError:
            pass
        
        return env_info
    
    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        
        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass
        
        if HAS_TORCH:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.output_dir / "eval.log"
        
        logging.basicConfig(
            level=logging.INFO if self.config.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
