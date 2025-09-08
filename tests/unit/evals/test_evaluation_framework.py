"""
Comprehensive test suite for the evaluation framework.

Includes unit tests, integration tests, property-based tests,
and performance tests for all evaluation components.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from evals.config import EvalConfig, load_config, ModelConfig, RetrievalConfig
from evals.metrics import MetricsCalculator, EvalResults
from evals.datasets import DatasetLoader
from evals.runner import EvalRunner


class TestEvalConfig:
    """Test configuration management and validation."""
    
    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = EvalConfig()
        assert config.name == "default_eval"
        assert config.seed == 42
        assert config.model.name == "codet5-base"
        assert config.retrieval.enabled is True
    
    def test_config_serialization(self):
        """Test config to dict and back conversion."""
        config = EvalConfig(name="test", seed=123)
        config_dict = config.__dict__
        assert config_dict['name'] == "test"
        assert config_dict['seed'] == 123
    
    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = """
name: "test_config"
seed: 999
model:
  name: "custom-model"
  temperature: 0.5
retrieval:
  top_k: 20
  rerank: false
"""
            f.write(yaml_content)
            f.flush()
            
            config = load_config(f.name)
            assert config.name == "test_config"
            assert config.seed == 999
            assert config.model.name == "custom-model"
            assert config.model.temperature == 0.5
            assert config.retrieval.top_k == 20
            assert config.retrieval.rerank is False
    
    def test_config_overrides(self):
        """Test configuration overrides."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("name: 'base'\nseed: 42")
            f.flush()
            
            overrides = {'seed': 999, 'output_dir': '/custom/path'}
            config = load_config(f.name, overrides)
            assert config.seed == 999
            assert config.output_dir == '/custom/path'


class TestMetricsCalculator:
    """Test metrics calculation and statistical analysis."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = MetricsCalculator(bootstrap_samples=100)
        self.sample_predictions = [
            "This function calculates fibonacci numbers",
            "This is a recursive implementation",
            "Returns the nth fibonacci number"
        ]
        self.sample_references = [
            "Function to compute fibonacci sequence",
            "Recursive fibonacci calculation",
            "Computes fibonacci number at position n"
        ]
        self.sample_latencies = [0.1, 0.2, 0.15]
        self.sample_costs = [0.01, 0.02, 0.015]
    
    def test_basic_metrics_calculation(self):
        """Test basic metrics without ML libraries."""
        results = self.calculator.calculate_all_metrics(
            self.sample_predictions,
            self.sample_references,
            self.sample_latencies,
            self.sample_costs
        )
        
        assert isinstance(results, EvalResults)
        assert results.num_samples == 3
        assert results.avg_latency == pytest.approx(0.15, rel=1e-2)
        assert results.p95_latency >= results.avg_latency
        assert results.total_cost == pytest.approx(0.045, rel=1e-2)
    
    @pytest.mark.parametrize("bootstrap_samples", [0, 10, 100])
    def test_confidence_intervals(self, bootstrap_samples):
        """Test confidence interval calculation with different sample sizes."""
        calculator = MetricsCalculator(bootstrap_samples=bootstrap_samples)
        results = calculator.calculate_all_metrics(
            self.sample_predictions,
            self.sample_references,
            self.sample_latencies,
            self.sample_costs
        )
        
        if bootstrap_samples > 0:
            assert len(results.confidence_intervals) > 0
            for metric, (lower, upper) in results.confidence_intervals.items():
                assert lower <= upper
        else:
            assert len(results.confidence_intervals) == 0
    
    def test_retrieval_metrics(self):
        """Test retrieval-specific metrics calculation."""
        retrieval_results = [
            {
                'retrieved_docs': [
                    {'doc_id': '1', 'score': 0.9, 'relevant': True},
                    {'doc_id': '2', 'score': 0.8, 'relevant': False},
                    {'doc_id': '3', 'score': 0.7, 'relevant': True}
                ],
                'total_relevant': 2
            }
        ]
        
        results = self.calculator.calculate_all_metrics(
            ['test'],
            ['reference'],
            [0.1],
            [0.01],
            retrieval_results
        )
        
        assert results.retrieval_precision > 0
        assert results.retrieval_recall > 0
        assert results.retrieval_ndcg > 0
    
    def test_empty_inputs_handling(self):
        """Test handling of empty or invalid inputs."""
        results = self.calculator.calculate_all_metrics([], [], [], [])
        assert results.num_samples == 0
        assert results.avg_latency == 0.0
        assert results.total_cost == 0.0
    
    def test_results_serialization(self):
        """Test EvalResults serialization to dict."""
        results = EvalResults(
            accuracy=0.85,
            bleu_score=0.65,
            num_samples=100,
            confidence_intervals={'bleu_score': (0.6, 0.7)}
        )
        
        result_dict = results.to_dict()
        assert result_dict['accuracy'] == 0.85
        assert result_dict['bleu_score'] == 0.65
        assert result_dict['num_samples'] == 100
        assert 'bleu_score' in result_dict['confidence_intervals']


class TestDatasetLoader:
    """Test dataset loading and validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.loader = DatasetLoader()
    
    def test_mock_dataset_creation(self):
        """Test creation of mock dataset."""
        dataset = self.loader._create_mock_dataset(5)
        assert len(dataset) == 5
        
        for sample in dataset:
            assert 'code' in sample
            assert 'reference' in sample
            assert 'metadata' in sample
            assert isinstance(sample['metadata'], dict)
    
    def test_dataset_validation(self):
        """Test dataset format validation and correction."""
        invalid_dataset = [
            {'code': 'def func(): pass'},  # Missing reference
            {'reference': 'A function'},    # Missing code
            {'code': 'print("hello")', 'reference': 'Print statement', 'metadata': {'type': 'simple'}}
        ]
        
        validated = self.loader._validate_dataset(invalid_dataset)
        assert len(validated) == 2  # One sample should be filtered out
        
        for sample in validated:
            assert 'code' in sample
            assert 'reference' in sample
            assert 'metadata' in sample
    
    def test_json_dataset_loading(self):
        """Test loading dataset from JSON file."""
        dataset_data = [
            {
                'code': 'def add(a, b): return a + b',
                'reference': 'Function that adds two numbers',
                'metadata': {'difficulty': 'easy'}
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(dataset_data, f)
            f.flush()
            
            # Mock config
            config = Mock()
            config.eval_path = f.name
            config.max_samples = None
            config.shuffle = False
            
            dataset = self.loader.load_dataset(config)
            assert len(dataset) == 1
            assert dataset[0]['code'] == 'def add(a, b): return a + b'
    
    def test_dataset_sampling(self):
        """Test dataset sampling and shuffling."""
        # Create a larger mock dataset
        config = Mock()
        config.eval_path = "nonexistent.json"  # Will trigger mock data
        config.max_samples = 3
        config.shuffle = True
        
        dataset = self.loader.load_dataset(config)
        assert len(dataset) <= 3


class TestEvalRunner:
    """Test evaluation runner and integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = EvalConfig(
            name="test_eval",
            output_dir="test_output",
            seed=42
        )
    
    def test_runner_initialization(self):
        """Test runner initialization with config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config.output_dir = tmpdir
            runner = EvalRunner(self.config)
            
            assert runner.config.name == "test_eval"
            assert runner.config.seed == 42
            assert Path(tmpdir).exists()
    
    def test_config_hash_calculation(self):
        """Test configuration hash calculation for reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config.output_dir = tmpdir
            runner = EvalRunner(self.config)
            
            hash1 = runner._calculate_config_hash()
            hash2 = runner._calculate_config_hash()
            assert hash1 == hash2  # Should be deterministic
            
            # Change config and verify hash changes
            self.config.seed = 999
            runner2 = EvalRunner(self.config)
            hash3 = runner2._calculate_config_hash()
            assert hash1 != hash3
    
    def test_seed_setting(self):
        """Test deterministic seed setting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config.output_dir = tmpdir
            runner = EvalRunner(self.config)
            
            # Test that numpy random state is set
            np.random.seed(42)
            val1 = np.random.random()
            
            runner._set_seeds(42)
            val2 = np.random.random()
            
            # Should be deterministic
            runner._set_seeds(42)
            val3 = np.random.random()
            assert val2 == val3
    
    @patch('evals.runner.subprocess.run')
    def test_git_info_collection(self, mock_subprocess):
        """Test git information collection."""
        # Mock subprocess calls
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "mock_commit_hash"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config.output_dir = tmpdir
            runner = EvalRunner(self.config)
            
            git_info = runner._get_git_info()
            assert 'commit' in git_info
            assert 'branch' in git_info
    
    def test_environment_info_collection(self):
        """Test environment information collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config.output_dir = tmpdir
            runner = EvalRunner(self.config)
            
            env_info = runner._get_environment_info()
            assert 'python_version' in env_info
            assert 'platform' in env_info
            assert 'hostname' in env_info
    
    @patch('evals.runner.CodeExplainer')
    def test_mock_prediction_generation(self, mock_explainer_class):
        """Test prediction generation with mocked explainer."""
        # Setup mock
        mock_explainer = Mock()
        mock_explainer.explain_code.return_value = {
            'explanation': 'Mock explanation',
            'cost': 0.01
        }
        mock_explainer_class.return_value = mock_explainer
        
        dataset = [
            {'code': 'def test(): pass', 'reference': 'Test function'}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config.output_dir = tmpdir
            runner = EvalRunner(self.config)
            
            predictions, latencies, costs = runner._generate_predictions(dataset)
            
            assert len(predictions) == 1
            assert len(latencies) == 1
            assert len(costs) == 1
            assert predictions[0] == 'Mock explanation'
            assert costs[0] == 0.01


# Property-based tests
try:
    from hypothesis import given, strategies as st
    import hypothesis
    
    class TestPropertyBased:
        """Property-based tests using Hypothesis."""
        
        @given(
            predictions=st.lists(st.text(min_size=1), min_size=1, max_size=10),
            latencies=st.lists(st.floats(min_value=0.001, max_value=10.0), min_size=1, max_size=10)
        )
        def test_metrics_properties(self, predictions, latencies):
            """Test properties that should always hold for metrics calculation."""
            # Ensure equal length
            min_len = min(len(predictions), len(latencies))
            predictions = predictions[:min_len]
            latencies = latencies[:min_len]
            references = ['reference'] * min_len
            costs = [0.01] * min_len
            
            calculator = MetricsCalculator(bootstrap_samples=0)  # Skip bootstrap for speed
            results = calculator.calculate_all_metrics(predictions, references, latencies, costs)
            
            # Properties that should always hold
            assert results.num_samples == min_len
            assert results.avg_latency >= 0
            assert results.p95_latency >= results.avg_latency
            assert results.total_cost >= 0
        
        @given(seed=st.integers(min_value=0, max_value=2**31-1))
        def test_deterministic_seeds(self, seed):
            """Test that seed setting produces deterministic results."""
            config = EvalConfig(seed=seed, output_dir="test")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config.output_dir = tmpdir
                runner = EvalRunner(config)
                
                # Should not raise any exceptions
                runner._set_seeds(seed)
                
                # Check numpy determinism if available
                try:
                    import numpy as np
                    val1 = np.random.random()
                    runner._set_seeds(seed)
                    val2 = np.random.random()
                    assert val1 == val2
                except ImportError:
                    pass

except ImportError:
    # Hypothesis not available, skip property-based tests
    class TestPropertyBased:
        def test_hypothesis_not_available(self):
            pytest.skip("Hypothesis not available for property-based testing")


# Performance tests
class TestPerformance:
    """Performance and benchmark tests."""
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Create large mock dataset
        large_dataset = []
        for i in range(1000):
            large_dataset.append({
                'code': f'def func_{i}(): return {i}',
                'reference': f'Function that returns {i}'
            })
        
        calculator = MetricsCalculator(bootstrap_samples=0)  # Skip bootstrap for speed
        
        import time
        start_time = time.time()
        
        predictions = [f'Function {i}' for i in range(1000)]
        references = [item['reference'] for item in large_dataset]
        latencies = [0.1] * 1000
        costs = [0.01] * 1000
        
        results = calculator.calculate_all_metrics(predictions, references, latencies, costs)
        
        end_time = time.time()
        
        # Should complete in reasonable time (less than 5 seconds)
        assert end_time - start_time < 5.0
        assert results.num_samples == 1000
    
    def test_memory_usage(self):
        """Test memory usage with repeated calculations."""
        calculator = MetricsCalculator(bootstrap_samples=10)
        
        # Run multiple calculations to check for memory leaks
        for _ in range(100):
            predictions = ['test'] * 10
            references = ['reference'] * 10
            latencies = [0.1] * 10
            costs = [0.01] * 10
            
            results = calculator.calculate_all_metrics(predictions, references, latencies, costs)
            assert results.num_samples == 10
        
        # If we get here without memory errors, the test passes


# Integration tests
class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline end-to-end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvalConfig(
                name="integration_test",
                output_dir=tmpdir,
                seed=42
            )
            config.dataset.max_samples = 3  # Small dataset for testing
            
            runner = EvalRunner(config)
            
            # This should complete without errors
            results = runner.run_evaluation()
            
            # Verify outputs
            assert results.num_samples > 0
            assert Path(tmpdir, "metrics.json").exists()
            assert Path(tmpdir, "run_manifest.json").exists()
            assert Path(tmpdir, "config.yaml").exists()
    
    def test_ablation_study_integration(self):
        """Test ablation study integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EvalConfig(
                name="ablation_test",
                output_dir=tmpdir,
                seed=42
            )
            config.dataset.max_samples = 2  # Very small for testing
            
            runner = EvalRunner(config)
            
            # Run ablation study
            components = ['retrieval']
            results = runner.run_ablation_study(components)
            
            # Should have baseline and ablated results
            assert 'baseline' in results
            assert 'no_retrieval' in results
            assert Path(tmpdir, "ablation_report.json").exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
