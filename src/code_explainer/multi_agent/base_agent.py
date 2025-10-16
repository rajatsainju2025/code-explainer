"""Base agent classes for multi-agent system."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import AgentMessage, AgentRole, ExplanationComponent

logger = logging.getLogger(__name__)


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