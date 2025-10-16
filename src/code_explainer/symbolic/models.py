"""Data models for symbolic analysis."""

from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class SymbolicCondition:
    """Represents a symbolic condition in code."""

    condition_type: str  # 'input', 'precondition', 'postcondition', 'invariant'
    expression: str
    line_number: int
    confidence: float
    variables: Set[str]


@dataclass
class PropertyTest:
    """Represents a property-based test."""

    test_name: str
    property_description: str
    test_code: str
    input_constraints: List[str]
    expected_behavior: str


@dataclass
class SymbolicExplanation:
    """Complete symbolic explanation of code."""

    input_conditions: List[SymbolicCondition]
    preconditions: List[SymbolicCondition]
    postconditions: List[SymbolicCondition]
    invariants: List[SymbolicCondition]
    property_tests: List[PropertyTest]
    complexity_analysis: Dict[str, Any]
    data_flow: Dict[str, List[str]]