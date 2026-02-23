"""Base agent classes for multi-agent system."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from .models import AgentRole, ExplanationComponent


class BaseAgent(ABC):
    """Base class for all code explanation agents."""
    
    __slots__ = ('agent_id',)

    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id

    @abstractmethod
    def analyze_code(self, code: str, context: Dict[str, Any]) -> ExplanationComponent:
        """Analyze code and return explanation component."""
        pass
