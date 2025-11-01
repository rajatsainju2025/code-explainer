#!/usr/bin/env python
"""Validate dataset schema for code-explainer with enhanced checks.

Usage:
  python scripts/validate_dataset.py data/train.json
  python scripts/validate_dataset.py data/*.json --verbose
  python scripts/validate_dataset.py data/train.json --max-code-length 5000
"""
from __future__ import annotations
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

REQUIRED_KEYS = {"code", "explanation"}


def validate_entry(
    rec: Dict[str, Any],
    index: int,
    max_code_length: int = 2000,
    max_explanation_length: int = 1000
) -> List[str]:
    """Validate a single dataset entry.
    
    Args:
        rec: Dataset entry
        index: Entry index
        max_code_length: Maximum code length
        max_explanation_length: Maximum explanation length
        
    Returns:
        List of error messages
    """
    errors = []
    
    # Check required keys
    missing = REQUIRED_KEYS - set(rec.keys())
    if missing:
        errors.append(f"[{index}] missing keys {sorted(missing)}")
        return errors
    
    # Validate code
    code = rec.get('code', '')
    if not isinstance(code, str):
        errors.append(f"[{index}] 'code' must be string")
    elif not code.strip():
        errors.append(f"[{index}] 'code' cannot be empty")
    elif len(code) > max_code_length:
        errors.append(
            f"[{index}] 'code' too long: {len(code)} chars "
            f"(max {max_code_length})"
        )
    
    # Validate explanation
    explanation = rec.get('explanation', '')
    if not isinstance(explanation, str):
        errors.append(f"[{index}] 'explanation' must be string")
    elif not explanation.strip():
        errors.append(f"[{index}] 'explanation' cannot be empty")
    elif len(explanation) > max_explanation_length:
        errors.append(
            f"[{index}] 'explanation' too long: {len(explanation)} chars "
            f"(max {max_explanation_length})"
        )
    
    return errors


def calculate_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate dataset statistics.
    
    Args:
        data: Dataset entries
        
    Returns:
        Statistics dictionary
    """
    # Use generators for memory-efficient processing
    valid_records = (rec for rec in data if isinstance(rec, dict))
    code_lengths_gen = (len(rec.get('code', '')) for rec in valid_records)
    exp_lengths_gen = (len(rec.get('explanation', '')) for rec in valid_records)
    
    # Convert to lists for statistics (needed for min/max)
    code_lengths = list(code_lengths_gen)
    exp_lengths = list(exp_lengths_gen)
    
    return {
        'total': len(data),
        'code_avg': sum(code_lengths) / len(code_lengths) if code_lengths else 0,
        'code_min': min(code_lengths) if code_lengths else 0,
        'code_max': max(code_lengths) if code_lengths else 0,
        'exp_avg': sum(exp_lengths) / len(exp_lengths) if exp_lengths else 0,
        'exp_min': min(exp_lengths) if exp_lengths else 0,
        'exp_max': max(exp_lengths) if exp_lengths else 0,
    }


def validate(
    path: Path,
    verbose: bool = False,
    max_code_length: int = 2000,
    max_explanation_length: int = 1000
) -> int:
    """Validate a dataset file.
    
    Args:
        path: Path to dataset
        verbose: Show statistics
        max_code_length: Maximum code length
        max_explanation_length: Maximum explanation length
        
    Returns:
        Exit code (0=success, 1=validation errors, 2=file errors)
    """
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"ERROR: failed to read {path}: {e}")
        return 2
    
    if not isinstance(data, list):
        print("ERROR: dataset must be a list of objects")
        return 2
    
    # Collect all errors
    all_errors = []
    
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            all_errors.append(f"[{i}] record is not an object")
            continue
        
        errors = validate_entry(rec, i, max_code_length, max_explanation_length)
        all_errors.extend(errors)
    
    # Print results
    if all_errors:
        print(f"✗ {path}: {len(all_errors)} errors found")
        for error in all_errors:
            print(f"  ERROR: {error}")
        return 1
    
    print(f"✓ {path}: valid with {len(data)} records")
    
    # Show statistics if verbose
    if verbose and data:
        stats = calculate_stats(data)
        print(f"  Statistics:")
        print(f"    Total entries: {stats['total']}")
        print(f"    Code length: avg={stats['code_avg']:.1f}, "
              f"min={stats['code_min']}, max={stats['code_max']}")
        print(f"    Explanation length: avg={stats['exp_avg']:.1f}, "
              f"min={stats['exp_min']}, max={stats['exp_max']}")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate code-explainer dataset files'
    )
    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help='Dataset JSON files to validate'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed statistics'
    )
    parser.add_argument(
        '--max-code-length',
        type=int,
        default=2000,
        help='Maximum code length (default: 2000)'
    )
    parser.add_argument(
        '--max-explanation-length',
        type=int,
        default=1000,
        help='Maximum explanation length (default: 1000)'
    )
    
    args = parser.parse_args()
    
    exit_code = 0
    for file_path in args.files:
        result = validate(
            file_path,
            args.verbose,
            args.max_code_length,
            args.max_explanation_length
        )
        if result != 0:
            exit_code = result
    
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_dataset.py <path.json> [options]")
        print("Try --help for more options")
        sys.exit(2)
    sys.exit(main())
