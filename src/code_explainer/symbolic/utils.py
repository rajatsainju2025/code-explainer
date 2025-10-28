"""Utility functions for symbolic analysis."""

from .models import SymbolicExplanation


def format_symbolic_explanation(explanation: SymbolicExplanation) -> str:
    """Format symbolic explanation as human-readable text (optimized)."""
    sections = []

    # Pre-allocate and use list comprehensions for better performance
    if explanation.input_conditions:
        sections.extend([
            "Input Conditions:",
            *[f"  - {cond.expression} (confidence: {cond.confidence:.1f})" 
              for cond in explanation.input_conditions]
        ])

    if explanation.preconditions:
        sections.extend([
            "\nPreconditions:",
            *[f"  - {cond.expression} (confidence: {cond.confidence:.1f})" 
              for cond in explanation.preconditions]
        ])

    if explanation.postconditions:
        sections.extend([
            "\nPostconditions:",
            *[f"  - {cond.expression} (confidence: {cond.confidence:.1f})" 
              for cond in explanation.postconditions]
        ])

    if explanation.invariants:
        sections.extend([
            "\nInvariants:",
            *[f"  - {cond.expression} (confidence: {cond.confidence:.1f})" 
              for cond in explanation.invariants]
        ])

    if explanation.complexity_analysis:
        comp = explanation.complexity_analysis
        sections.extend([
            "\nComplexity Analysis:",
            f"  - Time Complexity: {comp.get('estimated_time_complexity', 'Unknown')}",
            f"  - Cyclomatic Complexity: {comp.get('cyclomatic_complexity', 0)}",
            f"  - Max Nesting Depth: {comp.get('nesting_depth', 0)}"
        ])

    if explanation.property_tests:
        sections.extend([
            "\nProperty-Based Tests:",
            *[f"  - {test.property_description}" 
              for test in explanation.property_tests[:3]]  # Show first 3 tests
        ])

    return "\n".join(sections) if sections else "No symbolic conditions detected."