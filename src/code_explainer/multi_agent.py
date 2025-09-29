"""Multi-agent framework for collaborative code explanation."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Defines different agent roles in the multi-agent system."""

    STRUCTURAL = "structural"  # AST analysis, code structure
    SEMANTIC = "semantic"  # Logic understanding, algorithms
    CONTEXT = "context"  # External documentation, similar code
    VERIFICATION = "verification"  # Test generation, validation


@dataclass
class AgentMessage:
    """Message passed between agents."""

    sender: str
    recipient: str
    content: Dict[str, Any]
    message_type: str
    timestamp: float


@dataclass
class ExplanationComponent:
    """A component of the final explanation."""

    agent_id: str
    component_type: str  # 'structure', 'logic', 'context', 'verification'
    content: str
    confidence: float
    metadata: Dict[str, Any]


class BaseAgent(ABC):
    """Base class for all code explanation agents."""

    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.inbox: List[AgentMessage] = []
        self.knowledge_base: Dict[str, Any] = {}

    @abstractmethod
    def analyze_code(self, code: str, context: Dict[str, Any]) -> ExplanationComponent:
        """Analyze code and return explanation component."""
        pass

    @abstractmethod
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and optionally return response."""
        pass

    def send_message(
        self, recipient: str, content: Dict[str, Any], message_type: str
    ) -> AgentMessage:
        """Create a message to send to another agent."""
        import time

        return AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            message_type=message_type,
            timestamp=time.time(),
        )

    def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent."""
        self.inbox.append(message)


class StructuralAgent(BaseAgent):
    """Agent specialized in code structure analysis."""

    def __init__(self):
        super().__init__("structural_agent", AgentRole.STRUCTURAL)

    def analyze_code(self, code: str, context: Dict[str, Any]) -> ExplanationComponent:
        """Analyze code structure using AST."""
        import ast

        from ..symbolic import SymbolicAnalyzer

        try:
            tree = ast.parse(code)
            analyzer = SymbolicAnalyzer()

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

            # Get complexity metrics
            symbolic_analysis = analyzer.analyze_code(code)
            complexity = symbolic_analysis.complexity_analysis

            structure_description = f"""
**Code Structure Analysis:**
- Functions: {', '.join(functions) if functions else 'None'}
- Classes: {', '.join(classes) if classes else 'None'}
- Imports: {', '.join(imports[:5]) if imports else 'None'}
- Cyclomatic Complexity: {complexity.get('cyclomatic_complexity', 'Unknown')}
- Estimated Time Complexity: {complexity.get('estimated_time_complexity', 'Unknown')}
- Number of Loops: {complexity.get('number_of_loops', 0)}
- Number of Conditions: {complexity.get('number_of_conditions', 0)}

This code has {'high' if complexity.get('cyclomatic_complexity', 0) > 5 else 'moderate' if complexity.get('cyclomatic_complexity', 0) > 2 else 'low'} complexity.
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
        import ast

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
        import ast

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


class VerificationAgent(BaseAgent):
    """Agent specialized in generating tests and verification conditions."""

    def __init__(self):
        super().__init__("verification_agent", AgentRole.VERIFICATION)

    def analyze_code(self, code: str, context: Dict[str, Any]) -> ExplanationComponent:
        """Generate verification tests and conditions."""
        try:
            from ..symbolic import SymbolicAnalyzer

            analyzer = SymbolicAnalyzer()
            symbolic_analysis = analyzer.analyze_code(code)

            test_suggestions = []
            for test in symbolic_analysis.property_tests[:3]:  # Top 3 tests
                test_suggestions.append(f"- {test.property_description}")

            verification_description = f"""
**Verification and Testing:**
Property-based tests that could be generated:
{chr(10).join(test_suggestions) if test_suggestions else "- Basic functionality tests"}

**Conditions to Verify:**
{self._format_conditions(symbolic_analysis)}

**Test Strategy:**
{self._suggest_test_strategy(code)}
"""

            return ExplanationComponent(
                agent_id=self.agent_id,
                component_type="verification",
                content=verification_description.strip(),
                confidence=0.8,
                metadata={"property_tests": symbolic_analysis.property_tests},
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

    def _format_conditions(self, symbolic_analysis) -> str:
        """Format symbolic conditions for verification."""
        conditions = []

        if symbolic_analysis.preconditions:
            conditions.append(
                "- Preconditions: "
                + ", ".join(cond.expression for cond in symbolic_analysis.preconditions[:2])
            )

        if symbolic_analysis.postconditions:
            conditions.append(
                "- Postconditions: "
                + ", ".join(cond.expression for cond in symbolic_analysis.postconditions[:2])
            )

        return "\n".join(conditions) if conditions else "- No explicit conditions detected"

    def _suggest_test_strategy(self, code: str) -> str:
        """Suggest testing strategy based on code characteristics."""
        import ast

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


class MultiAgentOrchestrator:
    """Orchestrates collaboration between multiple agents."""

    def __init__(self, explainer):
        self.agents: Dict[str, BaseAgent] = {
            "structural": StructuralAgent(),
            "semantic": SemanticAgent(explainer),
            "context": ContextAgent(),
            "verification": VerificationAgent(),
        }
        self.message_queue: List[AgentMessage] = []

    def explain_code_collaborative(self, code: str) -> str:
        """Generate collaborative explanation using multiple agents."""
        logger.info("Starting multi-agent collaborative explanation")

        # Get analysis from each agent
        components: List[ExplanationComponent] = []

        for agent_name, agent in self.agents.items():
            try:
                logger.info(f"Getting analysis from {agent_name} agent")
                component = agent.analyze_code(code, {})
                components.append(component)
            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                continue

        # Synthesize final explanation
        return self._synthesize_explanation(components)

    def _synthesize_explanation(self, components: List[ExplanationComponent]) -> str:
        """Synthesize individual agent analyses into cohesive explanation."""
        if not components:
            return "Unable to generate collaborative explanation."

        # Sort components by confidence and type priority
        type_priority = {"logic": 1, "structure": 2, "context": 3, "verification": 4}
        components.sort(key=lambda x: (type_priority.get(x.component_type, 5), -x.confidence))

        explanation_parts = [
            "# Multi-Agent Code Explanation",
            "",
            "This explanation was generated collaboratively by multiple specialized AI agents:",
            "",
        ]

        for component in components:
            if component.confidence > 0.5:  # Only include confident analyses
                explanation_parts.append(component.content)
                explanation_parts.append("")

        explanation_parts.extend(
            [
                "---",
                "",
                "**Collaboration Summary:**",
                f"This analysis combined insights from {len(components)} specialized agents, "
                f"providing a comprehensive view of the code from multiple perspectives: "
                f"structural analysis, semantic understanding, contextual information, and verification strategies.",
            ]
        )

        return "\n".join(explanation_parts)

    def send_message(self, message: AgentMessage) -> None:
        """Route message to appropriate agent."""
        if message.recipient in self.agents:
            self.agents[message.recipient].receive_message(message)
        self.message_queue.append(message)

    def process_messages(self) -> None:
        """Process all pending messages between agents."""
        for agent in self.agents.values():
            while agent.inbox:
                message = agent.inbox.pop(0)
                response = agent.process_message(message)
                if response:
                    self.send_message(response)
