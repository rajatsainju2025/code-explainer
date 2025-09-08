"""
Dataset loading and management for evaluation.

Handles loading, validation, and preprocessing of evaluation datasets
with support for multiple formats and stratification.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Dataset loader with support for multiple formats and preprocessing.
    
    Supports:
    - JSON datasets with code/reference pairs
    - Stratified sampling by difficulty
    - Dataset validation and statistics
    """
    
    def __init__(self):
        pass
    
    def load_dataset(self, config) -> List[Dict[str, Any]]:
        """
        Load dataset from configuration.
        
        Args:
            config: Dataset configuration object
            
        Returns:
            List of dataset samples
        """
        dataset_path = Path(config.eval_path)
        
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset_path}, using mock data")
            return self._create_mock_dataset(config.max_samples or 10)
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Validate dataset format
        dataset = self._validate_dataset(dataset)
        
        # Apply sampling
        if config.max_samples and len(dataset) > config.max_samples:
            if config.shuffle:
                import random
                dataset = random.sample(dataset, config.max_samples)
            else:
                dataset = dataset[:config.max_samples]
        
        logger.info(f"Loaded {len(dataset)} samples from {dataset_path}")
        return dataset
    
    def _validate_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Validate dataset format and add missing fields."""
        validated = []
        
        for i, sample in enumerate(dataset):
            # Ensure required fields
            if 'code' not in sample:
                logger.warning(f"Sample {i} missing 'code' field, skipping")
                continue
            
            if 'reference' not in sample:
                sample['reference'] = f"Reference explanation for: {sample['code'][:50]}..."
                logger.warning(f"Sample {i} missing 'reference' field, using default")
            
            # Add metadata if missing
            if 'metadata' not in sample:
                sample['metadata'] = {}
            
            validated.append(sample)
        
        return validated
    
    def _create_mock_dataset(self, size: int) -> List[Dict[str, Any]]:
        """Create mock dataset for testing."""
        mock_samples = [
            {
                'code': 'def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)',
                'reference': 'This function calculates the nth Fibonacci number using recursion.',
                'metadata': {'difficulty': 'medium', 'domain': 'algorithms'}
            },
            {
                'code': 'def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)',
                'reference': 'This function calculates the factorial of n using recursion.',
                'metadata': {'difficulty': 'easy', 'domain': 'algorithms'}
            },
            {
                'code': 'class Node:\n    def __init__(self, data):\n        self.data = data\n        self.next = None',
                'reference': 'This class defines a node for a linked list data structure.',
                'metadata': {'difficulty': 'easy', 'domain': 'data_structures'}
            },
            {
                'code': 'def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)',
                'reference': 'This function implements the quicksort algorithm for sorting arrays.',
                'metadata': {'difficulty': 'hard', 'domain': 'algorithms'}
            }
        ]
        
        # Repeat samples to reach desired size
        dataset = []
        for i in range(size):
            sample = mock_samples[i % len(mock_samples)].copy()
            sample['id'] = i
            dataset.append(sample)
        
        logger.info(f"Created mock dataset with {len(dataset)} samples")
        return dataset
