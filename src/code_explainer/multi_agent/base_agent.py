"""Base agent classes for multi-agent system."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from .models import AgentRole, ExplanationComponent

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all code explanation agents."""
    
    __slots__ = ('agent_id', 'role', 'knowledge_base')

    def __init__(self, agent_id: str, role: AgentRole):
        self.agent_id = agent_id
        self.role = role
        self.knowledge_base: Dict[str, Any] = {}

    @abstractmethod
    def analyze_code(self, code: str, context: Dict[str, Any]) -> ExplanationComponent:
        """Analyze code and return explanation component."""
        pass
