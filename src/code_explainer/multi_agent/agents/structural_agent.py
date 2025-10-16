"""Structural agent for code structure analysis."""

import ast
import logging
from typing import Any, Dict, Optional

from ..base_agent import BaseAgent
from ..models import AgentMessage, AgentRole, ExplanationComponent

logger = logging.getLogger(__name__)


class StructuralAgent(BaseAgent):
    """Agent specialized in code structure analysis."""

    def __init__(self):
        super().__init__("structural_agent", AgentRole.STRUCTURAL)

    def analyze_code(self, code: str, context: Dict[str, Any]) -> ExplanationComponent:
        """Analyze code structure using AST."""
        try:
            tree = ast.parse(code)

            # Extract structural information
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.extend([f"{module}.{alias.name}" for alias in node.names])

            # Get complexity metrics using symbolic analyzer
            try:
                from ...symbolic import SymbolicAnalyzer
                analyzer = SymbolicAnalyzer()
                symbolic_analysis = analyzer.analyze_code(code)
                complexity = symbolic_analysis.complexity_analysis
            except ImportError:
                complexity = {"cyclomatic_complexity": "Unknown", "estimated_time_complexity": "Unknown"}

            # Safely get complexity value
            complexity_val = complexity.get('cyclomatic_complexity', 0)
            if isinstance(complexity_val, str):
                complexity_level = "unknown"
            elif complexity_val > 5:
                complexity_level = "high"
            elif complexity_val > 2:
                complexity_level = "moderate"
            else:
                complexity_level = "low"

            structure_description = f"""
**Code Structure Analysis:**
- Functions: {', '.join(functions) if functions else 'None'}
- Classes: {', '.join(classes) if classes else 'None'}
- Imports: {', '.join(imports[:5]) if imports else 'None'}
- Cyclomatic Complexity: {complexity.get('cyclomatic_complexity', 'Unknown')}
- Estimated Time Complexity: {complexity.get('estimated_time_complexity', 'Unknown')}
- Number of Loops: {complexity.get('number_of_loops', 0)}
- Number of Conditions: {complexity.get('number_of_conditions', 0)}

This code has {complexity_level} complexity.
"""

            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="structure",
                content=structure_description.strip(),
                confidence=0.9,
                metadata={
                    "functions": functions,
                    "classes": classes,
                    "imports": imports,
                    "complexity": complexity,
                },
            )

        except Exception as e:
            logger.error(f"Structural analysis failed: {e}")
            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="structure",
                content="Unable to perform structural analysis.",
                confidence=0.1,
                metadata={"error": str(e)},
            )

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process messages from other agents."""
        if message.message_type == "request_structure":
            # Another agent is requesting structural information
            code = message.content.get("code", "")
            analysis = self.analyze_code(code, {})

            return self.send_message(
                recipient=message.sender,
                content={"analysis": analysis.metadata},
                message_type="structure_response",
            )
        return None