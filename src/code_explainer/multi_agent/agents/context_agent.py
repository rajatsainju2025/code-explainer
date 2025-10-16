"""Context agent for external documentation and best practices."""

import ast
import logging
from typing import Any, Dict, Optional

from ..base_agent import BaseAgent
from ..models import AgentMessage, AgentRole, ExplanationComponent

logger = logging.getLogger(__name__)


class ContextAgent(BaseAgent):
    """Agent specialized in retrieving external context and documentation."""

    def __init__(self):
        super().__init__("context_agent", AgentRole.CONTEXT)

    def analyze_code(self, code: str, context: Dict[str, Any]) -> ExplanationComponent:
        """Provide contextual information about the code."""
        try:
            context_info = self._gather_context(code)

            context_description = f"""
**Contextual Information:**
{context_info}

**Best Practices:**
{self._suggest_best_practices(code)}
"""

            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="context",
                content=context_description.strip(),
                confidence=0.7,
                metadata={"context_sources": ["stdlib", "common_patterns"]},
            )

        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="context",
                content="Limited contextual information available.",
                confidence=0.3,
                metadata={"error": str(e)},
            )

    def _gather_context(self, code: str) -> str:
        """Gather contextual information about the code."""
        try:
            tree = ast.parse(code)
            context_notes = []

            # Check for common patterns
            has_error_handling = any(isinstance(node, ast.Try) for node in ast.walk(tree))
            has_type_hints = any(
                hasattr(node, "annotation") and node.annotation
                for node in ast.walk(tree)
                if isinstance(node, ast.arg)
            )
            has_docstrings = any(isinstance(node, ast.Str) for node in ast.walk(tree))

            if has_error_handling:
                context_notes.append("- Uses proper error handling with try/except blocks")
            if has_type_hints:
                context_notes.append("- Includes type hints for better code documentation")
            if has_docstrings:
                context_notes.append("- Contains docstrings for documentation")

            # Detect common algorithms/patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if "sort" in node.name.lower():
                        context_notes.append("- Implements a sorting algorithm")
                    elif "search" in node.name.lower():
                        context_notes.append("- Implements a search algorithm")
                    elif any(
                        keyword in node.name.lower() for keyword in ["factorial", "fibonacci"]
                    ):
                        context_notes.append("- Implements a mathematical/recursive function")

            return "\n".join(context_notes) if context_notes else "Standard Python code structure"

        except Exception:
            return "Code pattern analysis unavailable"

    def _suggest_best_practices(self, code: str) -> str:
        """Suggest best practices based on code analysis."""
        suggestions = []

        if "def " in code and '"""' not in code and "'''" not in code:
            suggestions.append("- Consider adding docstrings to functions")
        if "print(" in code:
            suggestions.append("- Consider using logging instead of print for production code")
        if len(code.split("\n")) > 20:
            suggestions.append("- Consider breaking long functions into smaller, focused functions")

        return "\n".join(suggestions) if suggestions else "Code follows good practices"

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process messages from other agents."""
        if message.message_type == "request_context":
            code = message.content.get("code", "")
            analysis = self.analyze_code(code, {})

            return self.send_message(
                recipient=message.sender,
                content={"analysis": analysis.content},
                message_type="context_response",
            )
        return None