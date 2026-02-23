"""Data models for symbolic analysis.

Optimized with:
- __future__ annotations for deferred evaluation
- __slots__ for memory efficiency
- FrozenSet for immutable variable sets
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List


@dataclass(slots=True, frozen=True)
class SymbolicCondition:
    """Represents a symbolic condition in code (immutable)."""

    condition_type: str  # 'input', 'precondition', 'postcondition', 'invariant'
    expression: str
    line_number: int
    confidence: float
    variables: FrozenSet[str]


@dataclass(slots=True)
class PropertyTest:
    """Represents a property-based test."""

    test_name: str
    property_description: str
    test_code: str
    input_constraints: List[str]
    expected_behavior: str


@dataclass(slots=True)
class SymbolicExplanation:
    """Complete symbolic explanation of code."""

    input_conditions: List[SymbolicCondition]
    preconditions: List[SymbolicCondition]
    postconditions: List[SymbolicCondition]
    invariants: List[SymbolicCondition]
    property_tests: List[PropertyTest]
    complexity_analysis: Dict[str, Any]
    data_flow: Dict[str, List[str]]