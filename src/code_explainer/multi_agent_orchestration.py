"""
Multi-Agent Orchestration Module

This module implements sophisticated orchestration patterns for coordinating
multiple LLM agents in code intelligence tasks. Based on recent research in
multi-agent systems, it enables collaborative problem-solving, task decomposition,
and consensus-based decision making.

Features:
- Agent role specialization (code analysis, testing, documentation, review)
- Task decomposition and workflow orchestration
- Inter-agent communication protocols
- Consensus mechanisms and conflict resolution
- Hierarchical agent coordination
- Performance monitoring and optimization
- Research-backed orchestration patterns
- Scalable agent deployment and management
- Quality assurance through agent collaboration
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class AgentRole(Enum):
    """Roles that agents can specialize in."""
    CODE_ANALYZER = "code_analyzer"
    TEST_GENERATOR = "test_generator"
    DOCUMENTATION_WRITER = "documentation_writer"
    SECURITY_REVIEWER = "security_reviewer"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    CODE_REVIEWER = "code_reviewer"
    ARCHITECTURE_DESIGNER = "architecture_designer"
    DEBUGGING_SPECIALIST = "debugging_specialist"


class CommunicationProtocol(Enum):
    """Communication protocols between agents."""
    DIRECT_MESSAGE = "direct_message"
    BROADCAST = "broadcast"
    HIERARCHICAL = "hierarchical"
    CONSENSUS_BASED = "consensus_based"
    AUCTION_BASED = "auction_based"


class TaskStatus(Enum):
    """Status of agent tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentMessage:
    """Message between agents."""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Task assigned to an agent."""
    task_id: str
    task_type: str
    description: str
    assigned_agent: str
    status: TaskStatus
    priority: int
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['AgentTask'] = field(default_factory=list)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentProfile:
    """Profile of an agent."""
    agent_id: str
    name: str
    role: AgentRole
    capabilities: List[str]
    performance_score: float = 0.0
    specialization_score: Dict[str, float] = field(default_factory=dict)
    active_tasks: int = 0
    total_tasks_completed: int = 0
    average_response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result of orchestration process."""
    orchestration_id: str
    main_task: str
    participating_agents: List[str]
    subtasks_completed: int
    total_subtasks: int
    consensus_reached: bool
    final_result: Any
    execution_time: float
    agent_contributions: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class Agent:
    """Base agent class."""

    def __init__(self, agent_id: str, name: str, role: AgentRole,
                 model_function: Callable, capabilities: List[str] = None):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.model_function = model_function
        self.capabilities = capabilities or []
        self.message_queue = deque()
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        self.performance_metrics = defaultdict(float)

    def assign_task(self, task: AgentTask) -> None:
        """Assign a task to this agent."""
        self.active_tasks[task.task_id] = task
        task.status = TaskStatus.IN_PROGRESS

    def complete_task(self, task_id: str, result: Any = None,
                     error: str = None) -> None:
        """Mark a task as completed."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.result = result
            task.error = error
            task.status = TaskStatus.COMPLETED if not error else TaskStatus.FAILED
            task.completed_at = datetime.utcnow()

            self.completed_tasks.append(task)
            del self.active_tasks[task_id]

    def send_message(self, receiver_id: str, message_type: str,
                    content: Dict[str, Any]) -> AgentMessage:
        """Send a message to another agent."""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content
        )
        return message

    def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent."""
        self.message_queue.append(message)

    def process_messages(self) -> List[AgentMessage]:
        """Process pending messages."""
        processed = []
        while self.message_queue:
            message = self.message_queue.popleft()
            processed.append(message)
            # Process message based on type
            self._handle_message(message)
        return processed

    def _handle_message(self, message: AgentMessage) -> None:
        """Handle incoming message."""
        # Base implementation - subclasses can override
        pass

    def execute_task(self, task: AgentTask) -> Any:
        """Execute a task using the agent's model."""
        try:
            start_time = time.time()

            # Prepare prompt based on task type and agent role
            prompt = self._prepare_prompt(task)

            # Execute model
            result = self.model_function(prompt)

            execution_time = time.time() - start_time
            self.performance_metrics["average_response_time"] = (
                (self.performance_metrics["average_response_time"] +
                 execution_time) / 2
            )

            return result

        except Exception as e:
            return {"error": str(e)}

    def _prepare_prompt(self, task: AgentTask) -> str:
        """Prepare prompt for task execution."""
        role_context = f"You are a {self.role.value.replace('_', ' ')} agent."

        prompt = f"""{role_context}

Task: {task.description}
Task Type: {task.task_type}

Please provide a comprehensive response based on your expertise."""

        if task.metadata:
            prompt += f"\n\nAdditional Context: {json.dumps(task.metadata, indent=2)}"

        return prompt


class SpecializedAgent(Agent):
    """Agent with specialized capabilities."""

    def __init__(self, agent_id: str, name: str, role: AgentRole,
                 model_function: Callable, specializations: Dict[str, float]):
        super().__init__(agent_id, name, role, model_function)
        self.specializations = specializations

    def get_specialization_score(self, task_type: str) -> float:
        """Get specialization score for a task type."""
        return self.specializations.get(task_type, 0.5)


class TaskDecomposer:
    """Decomposes complex tasks into subtasks."""

    def __init__(self):
        self.decomposition_patterns = {
            "code_review": [
                "analyze_code_structure",
                "check_best_practices",
                "identify_potential_bugs",
                "review_security",
                "assess_performance"
            ],
            "code_generation": [
                "understand_requirements",
                "design_solution",
                "implement_code",
                "add_error_handling",
                "write_tests"
            ],
            "debugging": [
                "reproduce_issue",
                "analyze_error_messages",
                "identify_root_cause",
                "propose_fix",
                "test_fix"
            ]
        }

    def decompose_task(self, main_task: str, task_type: str,
                      complexity: str = "medium") -> List[AgentTask]:
        """Decompose a main task into subtasks."""
        if task_type not in self.decomposition_patterns:
            # Generic decomposition
            return self._generic_decomposition(main_task, task_type)

        pattern = self.decomposition_patterns[task_type]
        subtasks = []

        for i, subtask_name in enumerate(pattern):
            subtask = AgentTask(
                task_id=f"{main_task}_sub_{i}",
                task_type=subtask_name,
                description=f"{subtask_name.replace('_', ' ').title()}: {main_task}",
                assigned_agent="",  # To be assigned by orchestrator
                status=TaskStatus.PENDING,
                priority=len(pattern) - i,  # Higher priority for earlier subtasks
                metadata={"parent_task": main_task, "step": i + 1}
            )
            subtasks.append(subtask)

        return subtasks

    def _generic_decomposition(self, main_task: str, task_type: str) -> List[AgentTask]:
        """Generic task decomposition."""
        subtasks = [
            AgentTask(
                task_id=f"{main_task}_analysis",
                task_type="analysis",
                description=f"Analyze the {task_type} task: {main_task}",
                assigned_agent="",
                status=TaskStatus.PENDING,
                priority=3
            ),
            AgentTask(
                task_id=f"{main_task}_execution",
                task_type="execution",
                description=f"Execute the main {task_type} task: {main_task}",
                assigned_agent="",
                status=TaskStatus.PENDING,
                priority=2
            ),
            AgentTask(
                task_id=f"{main_task}_validation",
                task_type="validation",
                description=f"Validate the {task_type} results: {main_task}",
                assigned_agent="",
                status=TaskStatus.PENDING,
                priority=1
            )
        ]
        return subtasks


class AgentCoordinator:
    """Coordinates multiple agents for task execution."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.task_queue: deque = deque()
        self.completed_tasks: List[AgentTask] = []
        self.message_bus: List[AgentMessage] = []
        self.task_decomposer = TaskDecomposer()
        self.executor = ThreadPoolExecutor(max_workers=10)

    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the coordinator."""
        self.agents[agent.agent_id] = agent

    def submit_task(self, task_description: str, task_type: str,
                   priority: int = 1) -> str:
        """Submit a task for orchestration."""
        task_id = str(uuid.uuid4())

        main_task = AgentTask(
            task_id=task_id,
            task_type=task_type,
            description=task_description,
            assigned_agent="",
            status=TaskStatus.PENDING,
            priority=priority
        )

        self.task_queue.append(main_task)
        return task_id

    def orchestrate_task(self, task_id: str) -> OrchestrationResult:
        """Orchestrate execution of a task using multiple agents."""
        start_time = time.time()

        # Find the main task
        main_task = None
        for task in self.task_queue:
            if task.task_id == task_id:
                main_task = task
                break

        if not main_task:
            raise ValueError(f"Task {task_id} not found")

        # Decompose task into subtasks
        subtasks = self.task_decomposer.decompose_task(
            main_task.description, main_task.task_type
        )

        # Assign subtasks to appropriate agents
        participating_agents = set()
        for subtask in subtasks:
            agent_id = self._assign_subtask_to_agent(subtask)
            if agent_id:
                participating_agents.add(agent_id)
                subtask.assigned_agent = agent_id
                self.agents[agent_id].assign_task(subtask)

        # Execute subtasks
        futures = []
        for subtask in subtasks:
            if subtask.assigned_agent:
                future = self.executor.submit(
                    self._execute_agent_task,
                    subtask.assigned_agent,
                    subtask
                )
                futures.append((subtask, future))

        # Collect results
        subtask_results = []
        for subtask, future in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                subtask_results.append((subtask, result))
            except Exception as e:
                subtask_results.append((subtask, {"error": str(e)}))

        # Reach consensus on final result
        final_result = self._reach_consensus(subtask_results)
        consensus_reached = final_result is not None

        execution_time = time.time() - start_time

        # Calculate agent contributions
        agent_contributions = self._calculate_agent_contributions(subtask_results)

        result = OrchestrationResult(
            orchestration_id=str(uuid.uuid4()),
            main_task=main_task.description,
            participating_agents=list(participating_agents),
            subtasks_completed=len([r for r in subtask_results if "error" not in r[1]]),
            total_subtasks=len(subtasks),
            consensus_reached=consensus_reached,
            final_result=final_result,
            execution_time=execution_time,
            agent_contributions=agent_contributions
        )

        return result

    def _assign_subtask_to_agent(self, subtask: AgentTask) -> Optional[str]:
        """Assign a subtask to the most suitable agent."""
        best_agent = None
        best_score = -1

        for agent_id, agent in self.agents.items():
            if isinstance(agent, SpecializedAgent):
                score = agent.get_specialization_score(subtask.task_type)
            else:
                # Basic scoring based on role match
                score = 1.0 if agent.role.value in subtask.task_type else 0.5

            # Consider agent workload
            workload_penalty = agent.active_tasks / 10.0  # Max 10 concurrent tasks
            score = score * (1 - workload_penalty)

            if score > best_score:
                best_score = score
                best_agent = agent_id

        return best_agent

    def _execute_agent_task(self, agent_id: str, task: AgentTask) -> Any:
        """Execute a task using a specific agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return {"error": f"Agent {agent_id} not found"}

        result = agent.execute_task(task)
        agent.complete_task(task.task_id, result=result)
        return result

    def _reach_consensus(self, subtask_results: List[Tuple[AgentTask, Any]]) -> Any:
        """Reach consensus from multiple subtask results."""
        if not subtask_results:
            return None

        # Simple consensus: majority vote or first successful result
        successful_results = [result for _, result in subtask_results
                            if isinstance(result, dict) and "error" not in result]

        if not successful_results:
            return {"error": "No successful results from subtasks"}

        # For now, return the first successful result
        # In a more sophisticated implementation, this could use voting,
        # confidence scores, or other consensus mechanisms
        return successful_results[0]

    def _calculate_agent_contributions(self, subtask_results: List[Tuple[AgentTask, Any]]) -> Dict[str, Any]:
        """Calculate contributions of each agent."""
        contributions = defaultdict(lambda: {"tasks_completed": 0, "quality_score": 0})

        for subtask, result in subtask_results:
            agent_id = subtask.assigned_agent
            contributions[agent_id]["tasks_completed"] += 1

            # Simple quality score based on result
            quality = 1.0 if "error" not in result else 0.0
            contributions[agent_id]["quality_score"] += quality

        # Normalize quality scores
        for agent_data in contributions.values():
            if agent_data["tasks_completed"] > 0:
                agent_data["quality_score"] /= agent_data["tasks_completed"]

        return dict(contributions)

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        status = {}
        for agent_id, agent in self.agents.items():
            status[agent_id] = {
                "name": agent.name,
                "role": agent.role.value,
                "active_tasks": len(agent.active_tasks),
                "completed_tasks": len(agent.completed_tasks),
                "performance_metrics": dict(agent.performance_metrics)
            }
        return status


class HierarchicalOrchestrator:
    """Hierarchical orchestration for complex multi-agent systems."""

    def __init__(self):
        self.coordinators: Dict[str, AgentCoordinator] = {}
        self.hierarchy_levels = ["strategic", "tactical", "operational"]

    def create_hierarchy_level(self, level: str) -> AgentCoordinator:
        """Create a coordinator for a specific hierarchy level."""
        coordinator = AgentCoordinator()
        self.coordinators[level] = coordinator
        return coordinator

    def orchestrate_hierarchically(self, main_task: str, task_type: str) -> OrchestrationResult:
        """Orchestrate a task using hierarchical approach."""
        # Strategic level: High-level planning
        strategic_coordinator = self.coordinators.get("strategic")
        if strategic_coordinator:
            strategic_result = strategic_coordinator.orchestrate_task(
                strategic_coordinator.submit_task(
                    f"Strategic planning for: {main_task}", "planning"
                )
            )

        # Tactical level: Detailed planning and coordination
        tactical_coordinator = self.coordinators.get("tactical")
        if tactical_coordinator:
            tactical_result = tactical_coordinator.orchestrate_task(
                tactical_coordinator.submit_task(
                    f"Tactical coordination for: {main_task}", task_type
                )
            )
            return tactical_result

        # Fallback to operational level
        operational_coordinator = self.coordinators.get("operational")
        if operational_coordinator:
            return operational_coordinator.orchestrate_task(
                operational_coordinator.submit_task(main_task, task_type)
            )

        raise ValueError("No coordinators available for orchestration")


class ConsensusMechanism:
    """Advanced consensus mechanisms for multi-agent decisions."""

    def __init__(self):
        self.consensus_algorithms = {
            "majority_vote": self._majority_vote,
            "weighted_vote": self._weighted_vote,
            "confidence_weighted": self._confidence_weighted,
            "delphi_method": self._delphi_method
        }

    def reach_consensus(self, agent_responses: List[Dict[str, Any]],
                       algorithm: str = "majority_vote") -> Dict[str, Any]:
        """Reach consensus using specified algorithm."""
        if algorithm not in self.consensus_algorithms:
            algorithm = "majority_vote"

        return self.consensus_algorithms[algorithm](agent_responses)

    def _majority_vote(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simple majority vote consensus."""
        if not responses:
            return {"error": "No responses to evaluate"}

        # Count votes for each response
        response_counts = defaultdict(int)
        for response in responses:
            response_str = json.dumps(response, sort_keys=True)
            response_counts[response_str] += 1

        # Find majority response
        majority_response_str = max(response_counts, key=response_counts.get)
        majority_count = response_counts[majority_response_str]

        return {
            "consensus_result": json.loads(majority_response_str),
            "agreement_level": majority_count / len(responses),
            "total_responses": len(responses),
            "algorithm": "majority_vote"
        }

    def _weighted_vote(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Weighted vote based on agent expertise."""
        # Placeholder - would need agent expertise scores
        return self._majority_vote(responses)

    def _confidence_weighted(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consensus weighted by confidence scores."""
        # Placeholder - would need confidence scores in responses
        return self._majority_vote(responses)

    def _delphi_method(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Delphi method for iterative consensus building."""
        # Simplified implementation
        return self._majority_vote(responses)


class MultiAgentOrchestrator:
    """Main orchestrator for multi-agent systems."""

    def __init__(self):
        self.agent_coordinator = AgentCoordinator()
        self.hierarchical_orchestrator = HierarchicalOrchestrator()
        self.consensus_mechanism = ConsensusMechanism()
        self.orchestration_history: List[OrchestrationResult] = []

    def setup_standard_agents(self, model_function: Callable) -> None:
        """Set up standard specialized agents."""
        agents_config = [
            ("code_analyzer", "Code Analyzer", AgentRole.CODE_ANALYZER,
             {"code_analysis": 0.9, "bug_detection": 0.8}),
            ("test_generator", "Test Generator", AgentRole.TEST_GENERATOR,
             {"test_generation": 0.9, "edge_case_identification": 0.8}),
            ("doc_writer", "Documentation Writer", AgentRole.DOCUMENTATION_WRITER,
             {"documentation": 0.9, "api_description": 0.8}),
            ("security_reviewer", "Security Reviewer", AgentRole.SECURITY_REVIEWER,
             {"security_analysis": 0.9, "vulnerability_detection": 0.8}),
            ("performance_optimizer", "Performance Optimizer", AgentRole.PERFORMANCE_OPTIMIZER,
             {"optimization": 0.9, "profiling": 0.8})
        ]

        for agent_id, name, role, specializations in agents_config:
            agent = SpecializedAgent(
                agent_id=agent_id,
                name=name,
                role=role,
                model_function=model_function,
                specializations=specializations
            )
            self.agent_coordinator.register_agent(agent)

    def orchestrate_complex_task(self, task_description: str, task_type: str,
                               use_hierarchy: bool = False) -> OrchestrationResult:
        """Orchestrate a complex task using multiple agents."""
        if use_hierarchy:
            result = self.hierarchical_orchestrator.orchestrate_hierarchically(
                task_description, task_type
            )
        else:
            task_id = self.agent_coordinator.submit_task(task_description, task_type)
            result = self.agent_coordinator.orchestrate_task(task_id)

        self.orchestration_history.append(result)
        return result

    def get_orchestration_analytics(self) -> Dict[str, Any]:
        """Get analytics about orchestration performance."""
        if not self.orchestration_history:
            return {}

        analytics = {
            "total_orchestrations": len(self.orchestration_history),
            "average_execution_time": np.mean([r.execution_time for r in self.orchestration_history]),
            "consensus_rate": sum(1 for r in self.orchestration_history if r.consensus_reached) / len(self.orchestration_history),
            "average_agents_per_task": np.mean([len(r.participating_agents) for r in self.orchestration_history]),
            "task_completion_rate": np.mean([r.subtasks_completed / r.total_subtasks for r in self.orchestration_history])
        }

        return analytics

    def optimize_agent_assignment(self) -> None:
        """Optimize agent assignment based on performance history."""
        # Analyze past performance and adjust agent specializations
        # This would implement machine learning-based optimization
        pass


# Export main classes
__all__ = [
    "AgentRole",
    "CommunicationProtocol",
    "TaskStatus",
    "AgentMessage",
    "AgentTask",
    "AgentProfile",
    "OrchestrationResult",
    "Agent",
    "SpecializedAgent",
    "TaskDecomposer",
    "AgentCoordinator",
    "HierarchicalOrchestrator",
    "ConsensusMechanism",
    "MultiAgentOrchestrator"
]
