#!/usr/bin/env python3
"""
Type Hint Coverage Analysis

Analyzes the codebase for type hint coverage and missing annotations.
"""

import os
import ast
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class TypeHintAnalyzer(ast.NodeVisitor):
    """Analyze type hint coverage in Python files."""

    def __init__(self):
        self.functions_with_hints = 0
        self.functions_without_hints = 0
        self.functions_partial_hints = 0
        self.total_functions = 0
        self.class_methods = defaultdict(dict)
        self.current_class = None
        self.issues = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        if node.name.startswith('_') and node.name != '__init__':
            # Skip private methods for now
            self.generic_visit(node)
            return
        
        self.total_functions += 1
        
        # Check for return type hint
        has_return_hint = node.returns is not None
        
        # Check for parameter type hints
        param_hints = sum(1 for arg in node.args.args if arg.annotation is not None)
        total_params = len(node.args.args)
        
        if self.current_class:
            if total_params > 0:
                total_params -= 1  # Exclude 'self'
            if total_params > 0:
                param_hints = max(0, param_hints - 1)
        
        if has_return_hint and param_hints == total_params:
            self.functions_with_hints += 1
        elif has_return_hint or param_hints > 0:
            self.functions_partial_hints += 1
        else:
            self.functions_without_hints += 1
            if not node.name.startswith('_'):
                self.issues.append(f"Missing type hints: {node.name}")

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class


def analyze_file(file_path: str) -> Optional[Dict]:
    """Analyze a Python file for type hint coverage."""
    try:
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
    except:
        return None
    
    analyzer = TypeHintAnalyzer()
    analyzer.visit(tree)
    
    if analyzer.total_functions == 0:
        return None
    
    coverage = (analyzer.functions_with_hints + analyzer.functions_partial_hints * 0.5) / analyzer.total_functions
    
    return {
        "file": file_path,
        "total": analyzer.total_functions,
        "complete": analyzer.functions_with_hints,
        "partial": analyzer.functions_partial_hints,
        "missing": analyzer.functions_without_hints,
        "coverage": coverage,
        "issues": analyzer.issues[:3]  # Top 3 issues
    }


def scan_directory(directory: str) -> List[Dict]:
    """Scan all Python files and analyze type hint coverage."""
    results = []
    
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache']]
        
        for file in files:
            if not file.endswith('.py'):
                continue
            
            file_path = os.path.join(root, file)
            analysis = analyze_file(file_path)
            if analysis:
                results.append(analysis)
    
    return sorted(results, key=lambda x: x['coverage'])


# Run analysis
root_dir = "src/code_explainer"
results = scan_directory(root_dir)

print("=" * 80)
print("TYPE HINT COVERAGE ANALYSIS")
print("=" * 80)

# Calculate statistics
total_files = len(results)
total_functions = sum(r["total"] for r in results)
complete_hints = sum(r["complete"] for r in results)
partial_hints = sum(r["partial"] for r in results)
missing_hints = sum(r["missing"] for r in results)

overall_coverage = (complete_hints + partial_hints * 0.5) / total_functions if total_functions > 0 else 0

print(f"\nOverall Statistics:")
print(f"  Files analyzed: {total_files}")
print(f"  Total functions: {total_functions}")
print(f"  Complete type hints: {complete_hints} ({complete_hints/total_functions*100:.1f}%)")
print(f"  Partial type hints: {partial_hints} ({partial_hints/total_functions*100:.1f}%)")
print(f"  Missing type hints: {missing_hints} ({missing_hints/total_functions*100:.1f}%)")
print(f"  Overall coverage: {overall_coverage*100:.1f}%")

print("\n\nFiles Needing Type Hints (sorted by coverage):")
print("-" * 80)
low_coverage = [r for r in results if r['coverage'] < 0.5]
for r in low_coverage[:15]:
    rel_path = os.path.relpath(r['file'], root_dir)
    print(f"  {rel_path}")
    print(f"    Coverage: {r['coverage']*100:.1f}% ({r['complete']}/{r['total']} complete)")

print("\n\nFiles with Good Type Hints (>80%):")
print("-" * 80)
good_coverage = [r for r in results if r['coverage'] >= 0.8]
for r in good_coverage[:10]:
    rel_path = os.path.relpath(r['file'], root_dir)
    print(f"  ✓ {rel_path}: {r['coverage']*100:.1f}%")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

print("""
Priority 1: Core modules (high impact, widely used)
  - model/core.py: Central to all operations
  - model_loader.py: Used everywhere
  - utils/ modules: Used across codebase

Priority 2: API and business logic
  - api/endpoints.py: Public API surface
  - retrieval/ modules: Core functionality
  - cache/ modules: Performance critical

Priority 3: Utilities and helpers
  - utils/ modules: Various utilities
  - symbolic/ modules: Analysis code
  - multi_agent/ modules: Complex logic

Type Hint Quick Wins:
  1. Add return types to public functions (5-10 min per file)
  2. Add parameter types to functions with >2 parameters (3-5 min per file)
  3. Use Union types for flexible parameters (straightforward)
  4. Use Optional[T] instead of T | None (consistency)
  5. Use TypeVar for generic functions (when applicable)
""")

print("=" * 80)
