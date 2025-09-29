"""Multi-agent evaluation system for collaborative code understanding."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Protocol, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles for different evaluation agents."""
    EXPLAINER = "explainer"
    REVIEWER = "reviewer"
    CRITIC = "critic"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"
    SPECIALIST = "specialist"


class InteractionType(Enum):
    """Types of agent interactions."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DEBATE = "debate"
    CONSENSUS = "consensus"
    HIERARCHICAL = "hierarchical"


@dataclass
class AgentMessage:
    """Message between agents."""
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    content: str
    message_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationTask:
    """Task for multi-agent evaluation."""
    task_id: str
    prompt: str
    context: Dict[str, Any]
    required_roles: List[AgentRole]
    interaction_type: InteractionType
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300


@dataclass
class AgentResponse:
    """Response from an evaluation agent."""
    agent_id: str
    role: AgentRole
    response: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_time: float = 0.0


@dataclass
class MultiAgentResult:
    """Result from multi-agent evaluation."""
    task_id: str
    individual_responses: List[AgentResponse]
    consensus_response: Optional[str]
    collaboration_metrics: Dict[str, float]
    final_score: float
    interaction_log: List[AgentMessage] = field(default_factory=list)


class EvaluationAgent(ABC):
    """Abstract base class for evaluation agents."""

    def __init__(self, agent_id: str, role: AgentRole, config: Optional[Dict[str, Any]] = None):
        """Initialize evaluation agent.

        Args:
            agent_id: Unique identifier for the agent
            role: Role of the agent
            config: Configuration dictionary
        """
        self.agent_id = agent_id
        self.role = role
        self.config = config or {}
        self.message_history: List[AgentMessage] = []
        self.knowledge_base: Dict[str, Any] = {}

    @abstractmethod
    async def process_task(self, task: EvaluationTask) -> AgentResponse:
        """Process an evaluation task.

        Args:
            task: Task to process

        Returns:
            Agent response
        """
        pass

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and process a message from another agent.

        Args:
            message: Incoming message

        Returns:
            Optional response message
        """
        self.message_history.append(message)
        return await self._process_message(message)

    @abstractmethod
    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming message and optionally respond.

        Args:
            message: Incoming message

        Returns:
            Optional response message
        """
        pass

    def update_knowledge(self, key: str, value: Any) -> None:
        """Update agent's knowledge base.

        Args:
            key: Knowledge key
            value: Knowledge value
        """
        self.knowledge_base[key] = value


class CodeExplainerAgent(EvaluationAgent):
    """Agent specialized in explaining code."""

    def __init__(self, agent_id: str, model_fn: Callable[[str], str], config: Optional[Dict[str, Any]] = None):
        """Initialize code explainer agent.

        Args:
            agent_id: Unique identifier
            model_fn: Function to call underlying model
            config: Configuration
        """
        super().__init__(agent_id, AgentRole.EXPLAINER, config)
        self.model_fn = model_fn

    async def process_task(self, task: EvaluationTask) -> AgentResponse:
        """Process code explanation task.

        Args:
            task: Evaluation task

        Returns:
            Agent response
        """
        start_time = time.time()

        # Enhance prompt with explainer-specific instructions
        enhanced_prompt = f"""As a code explanation specialist, analyze the following:

{task.prompt}

Provide a comprehensive explanation that includes:
1. What the code does (high-level purpose)
2. How it works (step-by-step breakdown)
3. Key algorithms or patterns used
4. Potential issues or improvements
5. Code quality assessment

Be clear, accurate, and educational in your explanation."""

        try:
            response = await asyncio.to_thread(self.model_fn, enhanced_prompt)
            confidence = self._calculate_confidence(response, task)
            reasoning = "Generated explanation using code analysis expertise"

        except Exception as e:
            response = f"Error processing explanation: {e}"
            confidence = 0.0
            reasoning = f"Failed due to: {e}"

        response_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            response=response,
            confidence=confidence,
            reasoning=reasoning,
            response_time=response_time,
            metadata={"enhanced_prompt_used": True}
        )

    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process message from another agent.

        Args:
            message: Incoming message

        Returns:
            Optional response
        """
        if message.message_type == "clarification_request":
            # Provide clarification on explanation
            clarification = f"Regarding my explanation: {message.content}"
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=clarification,
                message_type="clarification_response"
            )

        return None

    def _calculate_confidence(self, response: str, task: EvaluationTask) -> float:
        """Calculate confidence in the response.

        Args:
            response: Generated response
            task: Original task

        Returns:
            Confidence score between 0 and 1
        """
        # Simplified confidence calculation
        factors = []

        # Length factor (reasonable explanations should be substantial)
        length_score = min(1.0, len(response) / 500)
        factors.append(length_score)

        # Structure factor (look for numbered points, sections)
        structure_indicators = ["1.", "2.", "3.", "however", "therefore", "because"]
        structure_score = sum(1 for indicator in structure_indicators if indicator in response.lower()) / len(structure_indicators)
        factors.append(structure_score)

        # Technical terms factor
        technical_terms = ["function", "variable", "loop", "condition", "algorithm", "complexity"]
        tech_score = sum(1 for term in technical_terms if term in response.lower()) / len(technical_terms)
        factors.append(tech_score)

        return sum(factors) / len(factors)


class CodeReviewerAgent(EvaluationAgent):
    """Agent specialized in code review and critique."""

    def __init__(self, agent_id: str, model_fn: Callable[[str], str], config: Optional[Dict[str, Any]] = None):
        """Initialize code reviewer agent.

        Args:
            agent_id: Unique identifier
            model_fn: Function to call underlying model
            config: Configuration
        """
        super().__init__(agent_id, AgentRole.REVIEWER, config)
        self.model_fn = model_fn

    async def process_task(self, task: EvaluationTask) -> AgentResponse:
        """Process code review task.

        Args:
            task: Evaluation task

        Returns:
            Agent response
        """
        start_time = time.time()

        enhanced_prompt = f"""As a senior code reviewer, critically analyze the following:

{task.prompt}

Conduct a thorough review focusing on:
1. Code correctness and potential bugs
2. Security vulnerabilities
3. Performance issues
4. Code style and best practices
5. Maintainability and readability
6. Test coverage considerations

Provide specific, actionable feedback with severity levels."""

        try:
            response = await asyncio.to_thread(self.model_fn, enhanced_prompt)
            confidence = self._calculate_review_confidence(response)
            reasoning = "Generated review using code review expertise and security analysis"

        except Exception as e:
            response = f"Error during review: {e}"
            confidence = 0.0
            reasoning = f"Review failed: {e}"

        response_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            response=response,
            confidence=confidence,
            reasoning=reasoning,
            response_time=response_time,
            metadata={"review_categories_covered": True}
        )

    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process message from another agent.

        Args:
            message: Incoming message

        Returns:
            Optional response
        """
        if message.message_type == "review_challenge":
            # Respond to challenges of the review
            response_content = f"Regarding the challenge to my review: {message.content}. I maintain my assessment based on standard code review practices."
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=response_content,
                message_type="review_defense"
            )

        return None

    def _calculate_review_confidence(self, response: str) -> float:
        """Calculate confidence in the review.

        Args:
            response: Generated review

        Returns:
            Confidence score
        """
        # Look for review-specific indicators
        review_indicators = ["bug", "issue", "vulnerability", "improvement", "consider", "suggest"]
        severity_indicators = ["critical", "major", "minor", "low", "high", "medium"]

        review_score = sum(1 for indicator in review_indicators if indicator in response.lower()) / len(review_indicators)
        severity_score = sum(1 for indicator in severity_indicators if indicator in response.lower()) / len(severity_indicators)

        return (review_score + severity_score) / 2


class ValidatorAgent(EvaluationAgent):
    """Agent that validates responses from other agents."""

    def __init__(self, agent_id: str, model_fn: Callable[[str], str], config: Optional[Dict[str, Any]] = None):
        """Initialize validator agent.

        Args:
            agent_id: Unique identifier
            model_fn: Function to call underlying model
            config: Configuration
        """
        super().__init__(agent_id, AgentRole.VALIDATOR, config)
        self.model_fn = model_fn

    async def process_task(self, task: EvaluationTask) -> AgentResponse:
        """Process validation task.

        Args:
            task: Evaluation task

        Returns:
            Agent response
        """
        start_time = time.time()

        # Get responses from other agents from task context
        other_responses = task.context.get("agent_responses", [])

        validation_prompt = f"""As a validation specialist, assess the quality and accuracy of these responses to the task:

Original Task: {task.prompt}

Responses to validate:
{json.dumps([r.response for r in other_responses], indent=2)}

Evaluate each response for:
1. Factual accuracy
2. Completeness
3. Consistency between responses
4. Adherence to best practices
5. Overall quality

Provide a validation summary with scores and recommendations."""

        try:
            response = await asyncio.to_thread(self.model_fn, validation_prompt)
            confidence = 0.8  # Validators typically have high confidence in their assessment
            reasoning = "Validated responses using cross-checking and accuracy assessment"

        except Exception as e:
            response = f"Validation error: {e}"
            confidence = 0.0
            reasoning = f"Validation failed: {e}"

        response_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            role=self.role,
            response=response,
            confidence=confidence,
            reasoning=reasoning,
            response_time=response_time,
            metadata={"responses_validated": len(other_responses)}
        )

    async def _process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process message from another agent.

        Args:
            message: Incoming message

        Returns:
            Optional response
        """
        if message.message_type == "validation_request":
            # Provide quick validation
            validation = f"Quick validation of: {message.content} - This appears reasonable based on standard practices."
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content=validation,
                message_type="validation_response"
            )

        return None


class MultiAgentEvaluator:
    """Orchestrates multi-agent evaluation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize multi-agent evaluator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.agents: Dict[str, EvaluationAgent] = {}
        self.message_queue: List[AgentMessage] = []
        self.evaluation_history: List[MultiAgentResult] = []

    def register_agent(self, agent: EvaluationAgent) -> None:
        """Register an agent with the evaluator.

        Args:
            agent: Agent to register
        """
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.agent_id} with role {agent.role.value}")

    async def evaluate(self, task: EvaluationTask) -> MultiAgentResult:
        """Conduct multi-agent evaluation.

        Args:
            task: Task to evaluate

        Returns:
            Multi-agent evaluation result
        """
        start_time = time.time()

        # Get agents for required roles
        task_agents = []
        for role in task.required_roles:
            agent = self._get_agent_by_role(role)
            if agent:
                task_agents.append(agent)
            else:
                logger.warning(f"No agent found for role {role.value}")

        if not task_agents:
            raise ValueError("No suitable agents found for task")

        # Execute evaluation based on interaction type
        if task.interaction_type == InteractionType.SEQUENTIAL:
            responses = await self._sequential_evaluation(task, task_agents)
        elif task.interaction_type == InteractionType.PARALLEL:
            responses = await self._parallel_evaluation(task, task_agents)
        elif task.interaction_type == InteractionType.DEBATE:
            responses = await self._debate_evaluation(task, task_agents)
        elif task.interaction_type == InteractionType.CONSENSUS:
            responses = await self._consensus_evaluation(task, task_agents)
        else:
            # Default to parallel
            responses = await self._parallel_evaluation(task, task_agents)

        # Calculate collaboration metrics
        collaboration_metrics = self._calculate_collaboration_metrics(responses)

        # Generate consensus response
        consensus_response = await self._generate_consensus(responses, task)

        # Calculate final score
        final_score = self._calculate_final_score(responses, collaboration_metrics)

        result = MultiAgentResult(
            task_id=task.task_id,
            individual_responses=responses,
            consensus_response=consensus_response,
            collaboration_metrics=collaboration_metrics,
            final_score=final_score,
            interaction_log=self.message_queue.copy()
        )

        self.evaluation_history.append(result)
        self.message_queue.clear()

        logger.info(f"Multi-agent evaluation completed in {time.time() - start_time:.2f}s with score {final_score:.3f}")

        return result

    def _get_agent_by_role(self, role: AgentRole) -> Optional[EvaluationAgent]:
        """Get an agent by role.

        Args:
            role: Desired agent role

        Returns:
            Agent with the role, or None if not found
        """
        for agent in self.agents.values():
            if agent.role == role:
                return agent
        return None

    async def _sequential_evaluation(self, task: EvaluationTask, agents: List[EvaluationAgent]) -> List[AgentResponse]:
        """Conduct sequential evaluation.

        Args:
            task: Task to evaluate
            agents: List of agents

        Returns:
            List of agent responses
        """
        responses = []
        current_task = task

        for i, agent in enumerate(agents):
            # Add context from previous responses
            if responses:
                current_task.context["previous_responses"] = responses

            response = await agent.process_task(current_task)
            responses.append(response)

            # Allow agents to communicate
            if i < len(agents) - 1:
                message = AgentMessage(
                    sender_id=agent.agent_id,
                    receiver_id=agents[i + 1].agent_id,
                    content=response.response,
                    message_type="sequential_handoff"
                )
                self.message_queue.append(message)

        return responses

    async def _parallel_evaluation(self, task: EvaluationTask, agents: List[EvaluationAgent]) -> List[AgentResponse]:
        """Conduct parallel evaluation.

        Args:
            task: Task to evaluate
            agents: List of agents

        Returns:
            List of agent responses
        """
        # Execute all agents in parallel
        tasks = [agent.process_task(task) for agent in agents]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_responses = []
        for response in responses:
            if isinstance(response, AgentResponse):
                valid_responses.append(response)
            else:
                logger.error(f"Agent evaluation failed: {response}")

        return valid_responses

    async def _debate_evaluation(self, task: EvaluationTask, agents: List[EvaluationAgent]) -> List[AgentResponse]:
        """Conduct debate-style evaluation.

        Args:
            task: Task to evaluate
            agents: List of agents

        Returns:
            List of agent responses
        """
        # Initial responses
        initial_responses = await self._parallel_evaluation(task, agents)

        # Debate rounds
        debate_rounds = self.config.get("debate_rounds", 2)
        current_responses = initial_responses

        for round_num in range(debate_rounds):
            new_responses = []

            for i, agent in enumerate(agents):
                # Provide other agents' responses for debate
                other_responses = [r for j, r in enumerate(current_responses) if j != i]
                debate_task = EvaluationTask(
                    task_id=f"{task.task_id}_debate_{round_num}",
                    prompt=f"Consider these alternative responses and refine your answer:\n\nOriginal: {task.prompt}\n\nOther responses:\n{json.dumps([r.response for r in other_responses], indent=2)}\n\nProvide your refined response:",
                    context={"debate_round": round_num, "other_responses": other_responses},
                    required_roles=[agent.role],
                    interaction_type=InteractionType.PARALLEL
                )

                response = await agent.process_task(debate_task)
                new_responses.append(response)

            current_responses = new_responses

        return current_responses

    async def _consensus_evaluation(self, task: EvaluationTask, agents: List[EvaluationAgent]) -> List[AgentResponse]:
        """Conduct consensus-building evaluation.

        Args:
            task: Task to evaluate
            agents: List of agents

        Returns:
            List of agent responses
        """
        # Start with parallel evaluation
        responses = await self._parallel_evaluation(task, agents)

        # Iteratively build consensus
        consensus_rounds = self.config.get("consensus_rounds", 3)

        for round_num in range(consensus_rounds):
            # Check for consensus
            if self._check_consensus(responses):
                break

            # Ask agents to consider others' responses and move toward consensus
            new_responses = []

            for i, agent in enumerate(agents):
                other_responses = [r for j, r in enumerate(responses) if j != i]
                consensus_task = EvaluationTask(
                    task_id=f"{task.task_id}_consensus_{round_num}",
                    prompt=f"Consider these responses and provide a response that builds consensus:\n\nOriginal: {task.prompt}\n\nOther responses:\n{json.dumps([r.response for r in other_responses], indent=2)}\n\nWhat would be a consensus response?",
                    context={"consensus_round": round_num, "other_responses": other_responses},
                    required_roles=[agent.role],
                    interaction_type=InteractionType.PARALLEL
                )

                response = await agent.process_task(consensus_task)
                new_responses.append(response)

            responses = new_responses

        return responses

    def _check_consensus(self, responses: List[AgentResponse]) -> bool:
        """Check if responses have reached consensus.

        Args:
            responses: List of agent responses

        Returns:
            True if consensus reached
        """
        if len(responses) < 2:
            return True

        # Simple consensus check based on confidence and similarity
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        return avg_confidence > 0.8  # Simplified consensus threshold

    def _calculate_collaboration_metrics(self, responses: List[AgentResponse]) -> Dict[str, float]:
        """Calculate collaboration metrics.

        Args:
            responses: List of agent responses

        Returns:
            Collaboration metrics
        """
        if not responses:
            return {}

        metrics = {}

        # Agreement level (simplified)
        confidences = [r.confidence for r in responses]
        metrics["avg_confidence"] = sum(confidences) / len(confidences)
        metrics["confidence_variance"] = sum((c - metrics["avg_confidence"]) ** 2 for c in confidences) / len(confidences)

        # Response time consistency
        response_times = [r.response_time for r in responses]
        metrics["avg_response_time"] = sum(response_times) / len(response_times)
        metrics["response_time_variance"] = sum((t - metrics["avg_response_time"]) ** 2 for t in response_times) / len(response_times)

        # Coverage (different roles)
        unique_roles = len(set(r.role for r in responses))
        metrics["role_diversity"] = unique_roles / len(AgentRole)

        # Communication effectiveness
        metrics["message_count"] = len(self.message_queue)
        metrics["communication_efficiency"] = metrics["message_count"] / len(responses) if responses else 0

        return metrics

    async def _generate_consensus(self, responses: List[AgentResponse], task: EvaluationTask) -> Optional[str]:
        """Generate consensus response from individual responses.

        Args:
            responses: Individual agent responses
            task: Original task

        Returns:
            Consensus response
        """
        if not responses:
            return None

        if len(responses) == 1:
            return responses[0].response

        # Simple consensus by selecting highest confidence response
        # In practice, this would use more sophisticated consensus building
        best_response = max(responses, key=lambda r: r.confidence)

        return f"Consensus based on {len(responses)} agents: {best_response.response}"

    def _calculate_final_score(self, responses: List[AgentResponse], collaboration_metrics: Dict[str, float]) -> float:
        """Calculate final evaluation score.

        Args:
            responses: Agent responses
            collaboration_metrics: Collaboration metrics

        Returns:
            Final score between 0 and 1
        """
        if not responses:
            return 0.0

        # Weighted combination of individual scores and collaboration quality
        individual_score = sum(r.confidence for r in responses) / len(responses)
        collaboration_score = collaboration_metrics.get("avg_confidence", 0.0) * (1 - collaboration_metrics.get("confidence_variance", 1.0))

        # Weight individual performance more heavily, but consider collaboration
        final_score = 0.7 * individual_score + 0.3 * collaboration_score

        return min(1.0, max(0.0, final_score))


# Example usage
async def demo_multi_agent_evaluation():
    """Demonstrate multi-agent evaluation system."""

    # Mock model function
    def mock_model(prompt: str) -> str:
        return f"Mock response to: {prompt[:100]}..."

    # Create evaluator
    evaluator = MultiAgentEvaluator({
        "debate_rounds": 2,
        "consensus_rounds": 3
    })

    # Register agents
    explainer = CodeExplainerAgent("explainer_1", mock_model)
    reviewer = CodeReviewerAgent("reviewer_1", mock_model)
    validator = ValidatorAgent("validator_1", mock_model)

    evaluator.register_agent(explainer)
    evaluator.register_agent(reviewer)
    evaluator.register_agent(validator)

    # Create evaluation task
    task = EvaluationTask(
        task_id=str(uuid.uuid4()),
        prompt="Explain this Python function:\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        context={},
        required_roles=[AgentRole.EXPLAINER, AgentRole.REVIEWER, AgentRole.VALIDATOR],
        interaction_type=InteractionType.SEQUENTIAL
    )

    # Run evaluation
    result = await evaluator.evaluate(task)

    print("Multi-Agent Evaluation Result:")
    print(f"Task ID: {result.task_id}")
    print(f"Final Score: {result.final_score:.3f}")
    print(f"Individual Responses: {len(result.individual_responses)}")
    print(f"Collaboration Metrics: {result.collaboration_metrics}")

    return evaluator, result


if __name__ == "__main__":
    asyncio.run(demo_multi_agent_evaluation())
