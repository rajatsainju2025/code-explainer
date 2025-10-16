"""Verification agent for test generation and validation."""

import ast
import logging
from typing import Any, Dict, Optional

from ..base_agent import BaseAgent
from ..models import AgentMessage, AgentRole, ExplanationComponent

logger = logging.getLogger(__name__)


class VerificationAgent(BaseAgent):
    """Agent specialized in generating tests and verification conditions."""

    def __init__(self):
        super().__init__("verification_agent", AgentRole.VERIFICATION)

    def analyze_code(self, code: str, context: Dict[str, Any]) -> ExplanationComponent:
        """Generate verification tests and conditions."""
        try:
            test_suggestions = []
            verification_info = {}

            # Try to use symbolic analyzer for property tests
            try:
                from ...symbolic import SymbolicAnalyzer
                analyzer = SymbolicAnalyzer()
                symbolic_analysis = analyzer.analyze_code(code)

                for test in symbolic_analysis.property_tests[:3]:  # Top 3 tests
                    test_suggestions.append(f"- {test.property_description}")

                verification_info = {
                    "preconditions": [cond.expression for cond in symbolic_analysis.preconditions[:2]],
                    "postconditions": [cond.expression for cond in symbolic_analysis.postconditions[:2]]
                }
            except ImportError:
                test_suggestions.append("- Basic functionality tests")
                verification_info = {"preconditions": [], "postconditions": []}

            verification_description = f"""
**Verification and Testing:**
Property-based tests that could be generated:
{chr(10).join(test_suggestions)}

**Conditions to Verify:**
{self._format_conditions(verification_info)}

**Test Strategy:**
{self._suggest_test_strategy(code)}
"""

            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="verification",
                content=verification_description.strip(),
                confidence=0.8,
                metadata={"verification_info": verification_info},
            )

        except Exception as e:
            logger.error(f"Verification analysis failed: {e}")
            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="verification",
                content="Basic testing recommendations apply.",
                confidence=0.4,
                metadata={"error": str(e)},
            )

    def _format_conditions(self, verification_info: Dict[str, Any]) -> str:
        """Format symbolic conditions for verification."""
        conditions = []

        if verification_info.get("preconditions"):
            conditions.append(
                "- Preconditions: " + ", ".join(verification_info["preconditions"])
            )

        if verification_info.get("postconditions"):
            conditions.append(
                "- Postconditions: " + ", ".join(verification_info["postconditions"])
            )

        return "\n".join(conditions) if conditions else "- No explicit conditions detected"

    def _suggest_test_strategy(self, code: str) -> str:
        """Suggest testing strategy based on code characteristics."""
        try:
            tree = ast.parse(code)

            has_loops = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
            has_conditions = any(isinstance(node, ast.If) for node in ast.walk(tree))
            has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))

            if has_functions and has_loops and has_conditions:
                return "Comprehensive testing with edge cases, boundary conditions, and performance tests"
            elif has_functions and has_conditions:
                return "Unit testing with focus on conditional branches and edge cases"
            elif has_functions:
                return "Basic unit testing with input/output validation"
            else:
                return "Simple validation testing"

        except Exception:
            return "Standard testing practices recommended"

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process messages from other agents."""
        if message.message_type == "request_verification":
            code = message.content.get("code", "")
            analysis = self.analyze_code(code, {})

            return self.send_message(
                recipient=message.sender,
                content={"analysis": analysis.content},
                message_type="verification_response",
            )
        return None