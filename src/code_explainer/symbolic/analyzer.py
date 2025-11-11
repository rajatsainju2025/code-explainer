"""Main symbolic analyzer combining all analysis components."""

import ast
from functools import lru_cache
from typing import Dict, List, Tuple

from .models import SymbolicExplanation
from .extractors import ConditionExtractors
from .generators import PropertyGenerators
from .analyzers import ComplexityAnalyzers


class SymbolicAnalyzer(ConditionExtractors, PropertyGenerators, ComplexityAnalyzers):
    """Analyzes code to extract symbolic conditions and generate property tests."""

    def __init__(self):
        ConditionExtractors.__init__(self)
        PropertyGenerators.__init__(self)
        ComplexityAnalyzers.__init__(self)

        # Initialize state
        self.control_flow: List[ast.AST] = []
        # Cache for parsed ASTs to avoid reparsing
        self._ast_cache: Dict[str, ast.AST] = {}

    def analyze_code(self, code: str) -> SymbolicExplanation:
        """Analyze code and return symbolic explanation."""
        # Parse code into AST (with caching for repeated analyses)
        tree = self._get_or_parse_ast(code)

        # Reset state
        self._reset_state()

        # Analyze AST
        self._analyze_ast(tree)

        # Extract all conditions
        input_conditions = self._extract_input_conditions(tree)
        preconditions = self._extract_preconditions(tree)
        postconditions = self._extract_postconditions(tree)
        invariants = self._extract_invariants(tree)

        # Generate property tests
        property_tests = self._generate_property_tests(tree, code)

        # Analyze complexity
        complexity_analysis = self._analyze_complexity(tree)

        # Analyze data flow
        data_flow = self._analyze_data_flow(tree)

        return SymbolicExplanation(
            input_conditions=input_conditions,
            preconditions=preconditions,
            postconditions=postconditions,
            invariants=invariants,
            property_tests=property_tests,
            complexity_analysis=complexity_analysis,
            data_flow=data_flow,
        )

    def _get_or_parse_ast(self, code: str) -> ast.AST:
        """Get cached AST or parse new one."""
        # Use first 50 chars as cache key for small codes
        cache_key = code[:50] if len(code) < 100 else code
        
        if cache_key in self._ast_cache:
            return self._ast_cache[cache_key]
        
        tree = ast.parse(code)
        
        # Cache only small programs to avoid unbounded growth
        if len(code) < 5000 and len(self._ast_cache) < 100:
            self._ast_cache[cache_key] = tree
        
        return tree

    def _reset_state(self):
        """Reset analyzer state."""
        self.variable_assignments = {}
        self.function_calls = []
        self.control_flow = []

    def _analyze_ast(self, tree: ast.AST):
        """Analyze AST to build internal state (optimized with early exit)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in self.variable_assignments:
                            self.variable_assignments[target.id] = []
                        self.variable_assignments[target.id].append(node)
            elif isinstance(node, ast.Call):
                self.function_calls.append(node)
            elif isinstance(node, (ast.If, ast.While, ast.For)):
                self.control_flow.append(node)