"""Data models for multi-agent system."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class AgentRole(Enum):
    """Defines different agent roles in the multi-agent system."""

    STRUCTURAL = "structural"  # AST analysis, code structure
    SEMANTIC = "semantic"  # Logic understanding, algorithms
    CONTEXT = "context"  # External documentation, similar code
    VERIFICATION = "verification"  # Test generation, validation


@dataclass(slots=True)
class AgentMessage:
    """Message passed between agents."""

    sender: str
    recipient: str
    content: Dict[str, Any]
    message_type: str
    timestamp: float


@dataclass(slots=True)
class ExplanationComponent:
    """A component of the final explanation."""

    agent_id: str
    component_type: str  # 'structure', 'logic', 'context', 'verification'
    content: str
    confidence: float
    metadata: Dict[str, Any]