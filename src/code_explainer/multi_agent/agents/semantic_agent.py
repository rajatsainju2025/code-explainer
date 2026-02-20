"""Semantic agent for logic and algorithm analysis."""

import ast
import logging
from typing import Any, Dict

from ..base_agent import BaseAgent
from ..models import AgentRole, ExplanationComponent

logger = logging.getLogger(__name__)

# Pre-cache AST types for faster isinstance checks
_AST_FUNCTIONDEF = ast.FunctionDef
_AST_IF = ast.If
_AST_FOR = ast.For
_AST_WHILE = ast.While
_AST_RETURN = ast.Return
_AST_LOOP_TYPES = (_AST_FOR, _AST_WHILE)


class SemanticAgent(BaseAgent):
    """Agent specialized in semantic analysis and logic understanding."""

    __slots__ = ("explainer",)

    def __init__(self, explainer):
        super().__init__("semantic_agent", AgentRole.SEMANTIC)
        self.explainer = explainer

    def analyze_code(self, code: str, context: Dict[str, Any]) -> ExplanationComponent:
        """Analyze code semantics using the trained model."""
        try:
            # Use the main explainer for semantic analysis
            explanation = self.explainer.explain_code(code)
            logic_flow = self._analyze_logic_flow(code)

            semantic_description = f"""**Semantic Analysis:**
{explanation}

**Logic Flow:**
The code implements the following logical flow:
{logic_flow}"""

            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="logic",
                content=semantic_description,
                confidence=0.8,
                metadata={"base_explanation": explanation},
            )

        except Exception as e:
            logger.error("Semantic analysis failed: %s", e)
            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="logic",
                content="Unable to perform semantic analysis.",
                confidence=0.1,
                metadata={"error": str(e)},
            )

    def _analyze_logic_flow(self, code: str) -> str:
        """Analyze the logical flow of the code."""
        try:
            tree = ast.parse(code)
            flow_steps = []
            # Track what we've seen to avoid duplicates
            seen_func = seen_if = seen_loop = seen_return = False

            for node in ast.walk(tree):
                node_type = type(node)
                if not seen_func and node_type is _AST_FUNCTIONDEF:
                    flow_steps.append(f"1. Function '{node.name}' defines the main logic")
                    seen_func = True
                elif not seen_if and node_type is _AST_IF:
                    flow_steps.append("2. Conditional branching occurs")
                    seen_if = True
                elif not seen_loop and isinstance(node, _AST_LOOP_TYPES):
                    flow_steps.append("3. Iterative processing takes place")
                    seen_loop = True
                elif not seen_return and node_type is _AST_RETURN:
                    flow_steps.append("4. Result is returned")
                    seen_return = True

                # Early exit if we have all 4 flow steps
                if seen_func and seen_if and seen_loop and seen_return:
                    break

            return "\n".join(flow_steps) if flow_steps else "Simple sequential execution"

        except Exception:
            return "Complex control flow"
