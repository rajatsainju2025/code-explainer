"""Multi-agent framework for collaborative code explanation."""

from .models import AgentRole, ExplanationComponent
from .orchestrator import MultiAgentOrchestrator

__all__ = [
    "AgentRole",
    "ExplanationComponent",
    "MultiAgentOrchestrator",
]