"""Structural agent for code structure analysis."""

import ast
import logging
from typing import Any, Dict, List, Optional

from ..base_agent import BaseAgent
from ..models import AgentRole, ExplanationComponent

logger = logging.getLogger(__name__)

# Pre-cache AST node types for faster isinstance checks
_FUNC_DEF_TYPE = ast.FunctionDef
_CLASS_DEF_TYPE = ast.ClassDef
_IMPORT_TYPE = ast.Import
_IMPORT_FROM_TYPE = ast.ImportFrom

# Module-level lazy singleton — avoids constructing a new SymbolicAnalyzer
# (and cold-starting its internal AST cache) on every analyze_code() call.
_symbolic_analyzer: Optional[Any] = None


def _get_symbolic_analyzer() -> Any:
    """Return the module-level SymbolicAnalyzer singleton, creating it on first use."""
    global _symbolic_analyzer
    if _symbolic_analyzer is None:
        from ...symbolic import SymbolicAnalyzer  # deferred import keeps module load fast
        _symbolic_analyzer = SymbolicAnalyzer()
    return _symbolic_analyzer


class StructuralAgent(BaseAgent):
    """Agent specialized in code structure analysis."""
    
    __slots__ = ()  # Inherits slots from BaseAgent

    def __init__(self):
        super().__init__("structural_agent", AgentRole.STRUCTURAL)

    def analyze_code(self, code: str, context: Dict[str, Any]) -> ExplanationComponent:
        """Analyze code structure using AST."""
        try:
            tree = ast.parse(code)

            # Extract structural information in single pass
            functions: List[str] = []
            classes: List[str] = []
            imports: List[str] = []

            for node in ast.walk(tree):
                if isinstance(node, _FUNC_DEF_TYPE):
                    functions.append(node.name)
                elif isinstance(node, _CLASS_DEF_TYPE):
                    classes.append(node.name)
                elif isinstance(node, _IMPORT_TYPE):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, _IMPORT_FROM_TYPE):
                    module = node.module or ""
                    imports.extend(f"{module}.{alias.name}" for alias in node.names)

            # Get complexity metrics using the module-level SymbolicAnalyzer singleton.
            # Re-using the same instance preserves its internal AST cache across calls.
            try:
                analyzer = _get_symbolic_analyzer()
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
            logger.error("Structural analysis failed: %s", e)
            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="structure",
                content="Unable to perform structural analysis.",
                confidence=0.1,
                metadata={"error": str(e)},
            )
