"""Verification agent for test generation and validation."""

import ast
import logging
from typing import Any, Dict, Optional

from ..base_agent import BaseAgent
from ..models import AgentMessage, AgentRole, ExplanationComponent

logger = logging.getLogger(__name__)

# Pre-cache AST types for faster checks
_AST_FOR = ast.For
_AST_WHILE = ast.While
_AST_IF = ast.If
_AST_FUNCTIONDEF = ast.FunctionDef
_AST_LOOP_TYPES = (_AST_FOR, _AST_WHILE)


class VerificationAgent(BaseAgent):
    """Agent specialized in generating tests and verification conditions."""

    __slots__ = ()

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

                # Use islice-like early termination with enumerate
                for test in symbolic_analysis.property_tests[:3]:
                    test_suggestions.append(f"- {test.property_description}")

                # Build dicts inline using list comprehension
                pre = symbolic_analysis.preconditions[:2]
                post = symbolic_analysis.postconditions[:2]
                verification_info = {
                    "preconditions": [c.expression for c in pre],
                    "postconditions": [c.expression for c in post]
                }
            except ImportError:
                test_suggestions.append("- Basic functionality tests")
                verification_info = {"preconditions": [], "postconditions": []}

            # Build description efficiently
            tests_str = chr(10).join(test_suggestions)
            conditions_str = self._format_conditions(verification_info)
            strategy_str = self._suggest_test_strategy(code)

            verification_description = f"""**Verification and Testing:**
Property-based tests that could be generated:
{tests_str}

**Conditions to Verify:**
{conditions_str}

**Test Strategy:**
{strategy_str}"""

            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="verification",
                content=verification_description,
                confidence=0.8,
                metadata={"verification_info": verification_info},
            )

        except Exception as e:
            logger.error("Verification analysis failed: %s", e)
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

        pre = verification_info.get("preconditions")
        if pre:
            conditions.append("- Preconditions: " + ", ".join(pre))

        post = verification_info.get("postconditions")
        if post:
            conditions.append("- Postconditions: " + ", ".join(post))

        return "\n".join(conditions) if conditions else "- No explicit conditions detected"

    def _suggest_test_strategy(self, code: str) -> str:
        """Suggest testing strategy based on code characteristics."""
        try:
            tree = ast.parse(code)

            # Single pass through AST to detect all characteristics
            has_loops = has_conditions = has_functions = False
            for node in ast.walk(tree):
                node_type = type(node)
                if node_type is _AST_FUNCTIONDEF:
                    has_functions = True
                elif node_type is _AST_IF:
                    has_conditions = True
                elif node_type in (_AST_FOR, _AST_WHILE):
                    has_loops = True

                # Early exit if all detected
                if has_functions and has_conditions and has_loops:
                    break

            if has_functions and has_loops and has_conditions:
                return "Comprehensive testing with edge cases, boundary conditions, and performance tests"
            if has_functions and has_conditions:
                return "Unit testing with focus on conditional branches and edge cases"
            if has_functions:
                return "Basic unit testing with input/output validation"
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