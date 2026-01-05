"""Context agent for external documentation and best practices."""

import ast
import logging
from typing import Any, Dict, Optional

from ..base_agent import BaseAgent
from ..models import AgentMessage, AgentRole, ExplanationComponent

logger = logging.getLogger(__name__)

# Pre-cache AST node types for faster isinstance checks
_TRY_TYPE = ast.Try
_FUNC_DEF_TYPE = ast.FunctionDef
_ARG_TYPE = ast.arg
_STR_TYPE = ast.Str

# Pre-computed pattern keywords
_SORT_KEYWORDS = frozenset(["sort"])
_SEARCH_KEYWORDS = frozenset(["search"])
_MATH_KEYWORDS = frozenset(["factorial", "fibonacci"])


class ContextAgent(BaseAgent):
    """Agent specialized in retrieving external context and documentation."""
    
    __slots__ = ()  # Inherits slots from BaseAgent

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

            # Check for common patterns using cached types
            has_error_handling = False
            has_type_hints = False
            has_docstrings = False
            
            for node in ast.walk(tree):
                if isinstance(node, _TRY_TYPE):
                    has_error_handling = True
                elif isinstance(node, _ARG_TYPE) and getattr(node, "annotation", None):
                    has_type_hints = True
                elif isinstance(node, _STR_TYPE):
                    has_docstrings = True
                elif isinstance(node, _FUNC_DEF_TYPE):
                    name_lower = node.name.lower()
                    if any(kw in name_lower for kw in _SORT_KEYWORDS):
                        context_notes.append("- Implements a sorting algorithm")
                    elif any(kw in name_lower for kw in _SEARCH_KEYWORDS):
                        context_notes.append("- Implements a search algorithm")
                    elif any(kw in name_lower for kw in _MATH_KEYWORDS):
                        context_notes.append("- Implements a mathematical/recursive function")

            if has_error_handling:
                context_notes.append("- Uses proper error handling with try/except blocks")
            if has_type_hints:
                context_notes.append("- Includes type hints for better code documentation")
            if has_docstrings:
                context_notes.append("- Contains docstrings for documentation")

            return "\n".join(context_notes) if context_notes else "Standard Python code structure"

        except (SyntaxError, ValueError):
            return "Code pattern analysis unavailable"

    def _suggest_best_practices(self, code: str) -> str:
        """Suggest best practices based on code analysis."""
        suggestions = []

        if "def " in code and '"""' not in code and "'''" not in code:
            suggestions.append("- Consider adding docstrings to functions")
        if "print(" in code:
            suggestions.append("- Consider using logging instead of print for production code")
        
        # Cache line count to avoid multiple splits
        line_count = code.count("\n") + 1
        if line_count > 20:
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