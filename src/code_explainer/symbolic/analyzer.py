"""Main symbolic analyzer combining all analysis components."""

import ast
from typing import Dict, List

from .models import SymbolicExplanation
from .extractors import ConditionExtractors
from .generators import PropertyGenerators
from .analyzers import ComplexityAnalyzers

# Pre-cache AST node types for faster isinstance checks
_ASSIGN_TYPE = ast.Assign
_CONTROL_FLOW_TYPES = (ast.If, ast.While, ast.For)
_NAME_TYPE = ast.Name


class SymbolicAnalyzer(ConditionExtractors, PropertyGenerators, ComplexityAnalyzers):
    """Analyzes code to extract symbolic conditions and generate property tests."""
    
    __slots__ = ('control_flow', 'variable_assignments', '_ast_cache')

    def __init__(self):
        ConditionExtractors.__init__(self)
        PropertyGenerators.__init__(self)
        ComplexityAnalyzers.__init__(self)

        # Initialize state with pre-allocated capacity hints
        self.control_flow: List[ast.AST] = []
        self.variable_assignments: Dict[str, List[ast.AST]] = {}
        # Larger cache for parsed ASTs (from 100 to 256)
        self._ast_cache: Dict[str, ast.AST] = {}
        self._cache_size_limit = 256

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
        # For very short code, don't cache (hash overhead not worth it)
        if len(code) < 50:
            return ast.parse(code)
        
        # Use hash for cache key to handle any code length efficiently
        cache_key = hash(code)
        
        cached = self._ast_cache.get(cache_key)
        if cached is not None:
            return cached
        
        tree = ast.parse(code)
        
        # Cache with size limit (use LRU-like eviction)
        if len(code) < 5000:
            if len(self._ast_cache) >= self._cache_size_limit:
                # Remove oldest (first) item for simple LRU behavior
                first_key = next(iter(self._ast_cache))
                del self._ast_cache[first_key]
            self._ast_cache[cache_key] = tree
        
        return tree

    def _reset_state(self):
        """Reset analyzer state."""
        self.variable_assignments.clear()
        self.control_flow.clear()

    def _analyze_ast(self, tree: ast.AST):
        """Analyze AST to build internal state (optimized with type caching)."""
        var_assigns = self.variable_assignments
        ctrl_flow = self.control_flow
        
        for node in ast.walk(tree):
            if isinstance(node, _ASSIGN_TYPE):
                for target in node.targets:
                    if isinstance(target, _NAME_TYPE):
                        var_id = target.id
                        if var_id not in var_assigns:
                            var_assigns[var_id] = []
                        var_assigns[var_id].append(node)
            elif isinstance(node, _CONTROL_FLOW_TYPES):
                ctrl_flow.append(node)