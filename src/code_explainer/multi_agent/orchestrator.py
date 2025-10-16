"""Multi-agent orchestrator for collaborative code explanation."""

import logging
from typing import Dict, List

from .agents.context_agent import ContextAgent
from .agents.semantic_agent import SemanticAgent
from .agents.structural_agent import StructuralAgent
from .agents.verification_agent import VerificationAgent
from .base_agent import BaseAgent
from .models import AgentMessage, ExplanationComponent

logger = logging.getLogger(__name__)


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