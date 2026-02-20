"""Multi-agent orchestrator for collaborative code explanation."""

import logging
import concurrent.futures
from typing import Dict, List, Tuple

from .agents.context_agent import ContextAgent
from .agents.semantic_agent import SemanticAgent
from .agents.structural_agent import StructuralAgent
from .agents.verification_agent import VerificationAgent
from .base_agent import BaseAgent
from .models import AgentMessage, ExplanationComponent

logger = logging.getLogger(__name__)

# Pre-computed priority mapping for efficient sorting
_TYPE_PRIORITY: Dict[str, int] = {"logic": 1, "structure": 2, "context": 3, "verification": 4}


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
        """Generate collaborative explanation using multiple agents (with parallelization)."""
        logger.info("Starting multi-agent collaborative explanation")

        # Analyze with all agents in parallel for faster processing
        components: List[ExplanationComponent] = []
        
        def analyze_with_agent(agent_tuple: Tuple[str, BaseAgent]) -> ExplanationComponent:
            agent_name, agent = agent_tuple
            try:
                logger.info("Getting analysis from %s agent", agent_name)
                return agent.analyze_code(code, {})
            except Exception as e:
                logger.error("Agent %s failed: %s", agent_name, e)
                return None
        
        # Use thread pool to run agents in parallel (I/O bound operations)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_with_agent, item) for item in self.agents.items()]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    components.append(result)

        # Synthesize final explanation
        return self._synthesize_explanation(components)

    def _synthesize_explanation(self, components: List[ExplanationComponent]) -> str:
        """Synthesize individual agent analyses into cohesive explanation (optimized)."""
        if not components:
            return "Unable to generate collaborative explanation."

        # Sort using pre-computed priority map (avoids lambda function overhead)
        components.sort(key=lambda x: (_TYPE_PRIORITY.get(x.component_type, 5), -x.confidence))

        # Use list comprehension for high-confidence components (faster than generator for small lists)
        confident_parts = [
            component.content
            for component in components
            if component.confidence > 0.5
        ]

        # Build explanation parts efficiently with f-strings
        parts = [
            "# Multi-Agent Code Explanation",
            "",
            "This explanation was generated collaboratively by multiple specialized AI agents:",
            "",
            *confident_parts,  # Unpack confident components directly
            "",
            "---",
            "",
            "**Collaboration Summary:**",
            f"This analysis combined insights from {len(components)} specialized agents, "
            f"providing a comprehensive view of the code from multiple perspectives: "
            f"structural analysis, semantic understanding, contextual information, and verification strategies.",
        ]

        return "\n".join(parts)

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