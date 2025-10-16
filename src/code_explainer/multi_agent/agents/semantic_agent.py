"""Semantic agent for logic and algorithm analysis."""

import ast
import logging
from typing import Any, Dict, Optional

from ..base_agent import BaseAgent
from ..models import AgentMessage, AgentRole, ExplanationComponent

logger = logging.getLogger(__name__)


class SemanticAgent(BaseAgent):
    """Agent specialized in semantic analysis and logic understanding."""

    def __init__(self, explainer):
        super().__init__("semantic_agent", AgentRole.SEMANTIC)
        self.explainer = explainer

    def analyze_code(self, code: str, context: Dict[str, Any]) -> ExplanationComponent:
        """Analyze code semantics using the trained model."""
        try:
            # Use the main explainer for semantic analysis
            explanation = self.explainer.explain_code(code)

            semantic_description = f"""
**Semantic Analysis:**
{explanation}

**Logic Flow:**
The code implements the following logical flow:
{self._analyze_logic_flow(code)}
"""

            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="logic",
                content=semantic_description.strip(),
                confidence=0.8,
                metadata={"base_explanation": explanation},
            )

        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
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

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    flow_steps.append(f"1. Function '{node.name}' defines the main logic")
                elif isinstance(node, ast.If):
                    flow_steps.append("2. Conditional branching occurs")
                elif isinstance(node, (ast.For, ast.While)):
                    flow_steps.append("3. Iterative processing takes place")
                elif isinstance(node, ast.Return):
                    flow_steps.append("4. Result is returned")

            return "\n".join(flow_steps[:4]) if flow_steps else "Simple sequential execution"

        except Exception:
            return "Complex control flow"

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process messages from other agents."""
        if message.message_type == "request_semantics":
            code = message.content.get("code", "")
            analysis = self.analyze_code(code, {})

            return self.send_message(
                recipient=message.sender,
                content={"analysis": analysis.content},
                message_type="semantics_response",
            )
        return None