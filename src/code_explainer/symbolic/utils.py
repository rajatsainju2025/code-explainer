"""Utility functions for symbolic analysis."""

from .models import SymbolicExplanation


def format_symbolic_explanation(explanation: SymbolicExplanation) -> str:
    """Format symbolic explanation as human-readable text."""
    sections = []

    if explanation.input_conditions:
        sections.append("Input Conditions:")
        for cond in explanation.input_conditions:
            sections.append(f"  - {cond.expression} (confidence: {cond.confidence:.1f})")

    if explanation.preconditions:
        sections.append("\nPreconditions:")
        for cond in explanation.preconditions:
            sections.append(f"  - {cond.expression} (confidence: {cond.confidence:.1f})")

    if explanation.postconditions:
        sections.append("\nPostconditions:")
        for cond in explanation.postconditions:
            sections.append(f"  - {cond.expression} (confidence: {cond.confidence:.1f})")

    if explanation.invariants:
        sections.append("\nInvariants:")
        for cond in explanation.invariants:
            sections.append(f"  - {cond.expression} (confidence: {cond.confidence:.1f})")

    if explanation.complexity_analysis:
        sections.append("\nComplexity Analysis:")
        comp = explanation.complexity_analysis
        sections.append(f"  - Time Complexity: {comp.get('estimated_time_complexity', 'Unknown')}")
        sections.append(f"  - Cyclomatic Complexity: {comp.get('cyclomatic_complexity', 0)}")
        sections.append(f"  - Max Nesting Depth: {comp.get('nesting_depth', 0)}")

    if explanation.property_tests:
        sections.append("\nProperty-Based Tests:")
        for test in explanation.property_tests[:3]:  # Show first 3 tests
            sections.append(f"  - {test.property_description}")

    return "\n".join(sections) if sections else "No symbolic conditions detected."