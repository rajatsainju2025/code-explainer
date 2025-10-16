"""Multi-agent framework for collaborative code explanation."""

from .models import AgentRole, AgentMessage, ExplanationComponent
from .orchestrator import MultiAgentOrchestrator

__all__ = [
    "AgentRole",
    "AgentMessage",
    "ExplanationComponent",
    "MultiAgentOrchestrator",
]