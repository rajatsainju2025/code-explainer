"""Multi-agent orchestration and collaborative explanation patterns."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class AgentRole(Enum):
    """Roles in multi-agent explanation system."""
    ANALYZER = "analyzer"        # Analyzes code structure
    EXPLAINER = "explainer"      # Generates explanations
    VALIDATOR = "validator"      # Validates explanations
    JUDGE = "judge"              # Evaluates explanation quality


@dataclass
class AgentMessage:
    """Message in multi-agent communication."""
    source: AgentRole
    target: AgentRole
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExplanationResult:
    """Result from explanation process."""
    code: str
    explanations: List[str]
    confidence: float  # 0.0-1.0
    consensus_explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MultiAgentOrchestrator:
    """Orchestrates multiple explanation agents for consensus-based explanations."""
    
    def __init__(self, num_agents: int = 3):
        """Initialize orchestrator with multiple agents.
        
        Args:
            num_agents: Number of explanation agents (default: 3 for consensus)
        """
        self.num_agents = num_agents
        self.agents: List[Dict[str, Any]] = [
            {"role": AgentRole.ANALYZER, "active": True},
            {"role": AgentRole.EXPLAINER, "active": True},
            {"role": AgentRole.VALIDATOR, "active": True},
            {"role": AgentRole.JUDGE, "active": True},
        ]
        self.message_queue: List[AgentMessage] = []
    
    def add_agent_message(self, message: AgentMessage) -> None:
        """Queue message in multi-agent communication.
        
        Args:
            message: AgentMessage to queue
        """
        self.message_queue.append(message)
    
    def get_messages(self, role: AgentRole) -> List[AgentMessage]:
        """Get messages directed at specific agent role.
        
        Args:
            role: Target agent role
        
        Returns:
            List of messages for that role
        """
        return [msg for msg in self.message_queue if msg.target == role]
    
    def explain_with_consensus(
        self,
        code: str,
        explainer_fn,
        num_explanations: int = 3
    ) -> ExplanationResult:
        """Generate multiple explanations and compute consensus.
        
        Args:
            code: Source code to explain
            explainer_fn: Function to generate explanations
            num_explanations: Number of independent explanations (default: 3)
        
        Returns:
            ExplanationResult with multiple explanations and consensus
        """
        explanations = []
        
        for i in range(num_explanations):
            try:
                explanation = explainer_fn(code)
                explanations.append(explanation)
            except Exception as e:
                # Continue even if one agent fails
                continue
        
        # Compute consensus (simple: longest explanation or average)
        consensus = self._compute_consensus(explanations)
        
        # Compute confidence based on agreement
        confidence = self._compute_confidence(explanations)
        
        return ExplanationResult(
            code=code,
            explanations=explanations,
            confidence=confidence,
            consensus_explanation=consensus,
            metadata={"num_agents": len(explanations), "consensus_method": "voting"}
        )
    
    def _compute_consensus(self, explanations: List[str]) -> Optional[str]:
        """Compute consensus explanation from multiple candidates.
        
        Args:
            explanations: List of explanation strings
        
        Returns:
            Consensus explanation or None if empty
        """
        if not explanations:
            return None
        
        # Simple heuristic: return longest (most detailed) explanation
        return max(explanations, key=len)
    
    def _compute_confidence(self, explanations: List[str]) -> float:
        """Compute confidence score based on explanation agreement.
        
        Args:
            explanations: List of explanation strings
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not explanations:
            return 0.0
        
        # Simple metric: if all explanations are identical, confidence = 1.0
        # Otherwise, compute based on similarity
        if len(set(explanations)) == 1:
            return 1.0
        
        # Approximate confidence based on number of unique explanations
        unique_count = len(set(explanations))
        total_count = len(explanations)
        
        # Confidence decreases with diversity
        return max(0.0, 1.0 - (unique_count - 1) / total_count)
    
    def clear_message_queue(self) -> None:
        """Clear inter-agent message queue."""
        self.message_queue.clear()
