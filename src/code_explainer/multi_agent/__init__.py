"""Multi-agent framework for collaborative code explanation."""

from .agents.context_agent import ContextAgent
from .agents.semantic_agent import SemanticAgent
from .agents.structural_agent import StructuralAgent
from .agents.verification_agent import VerificationAgent
from .agents import get_shared_symbolic_analyzer
from .base_agent import BaseAgent
from .models import AgentRole, ExplanationComponent
from .orchestrator import MultiAgentOrchestrator

__all__ = [
    # Orchestration
    "MultiAgentOrchestrator",
    # Base
    "BaseAgent",
    # Concrete agents
    "ContextAgent",
    "SemanticAgent",
    "StructuralAgent",
    "VerificationAgent",
    # Data models
    "AgentRole",
    "ExplanationComponent",
    # Shared helpers
    "get_shared_symbolic_analyzer",
]