"""Tests for multi-agent orchestration and consensus-based explanations."""

import pytest

from code_explainer.multi_agent import (
    AgentRole,
    AgentMessage,
    ExplanationResult,
    MultiAgentOrchestrator,
)


def test_agent_message_creation():
    msg = AgentMessage(
        source=AgentRole.ANALYZER,
        target=AgentRole.EXPLAINER,
        content="Analyze this code"
    )
    
    assert msg.source == AgentRole.ANALYZER
    assert msg.target == AgentRole.EXPLAINER
    assert msg.content == "Analyze this code"


def test_explanation_result_creation():
    result = ExplanationResult(
        code="x = 1",
        explanations=["Assigns 1 to x"],
        confidence=0.95
    )
    
    assert result.code == "x = 1"
    assert len(result.explanations) == 1
    assert result.confidence == 0.95


def test_multi_agent_orchestrator_init():
    orchestrator = MultiAgentOrchestrator(num_agents=3)
    
    assert orchestrator.num_agents == 3
    assert len(orchestrator.agents) == 4  # ANALYZER, EXPLAINER, VALIDATOR, JUDGE


def test_add_agent_message():
    orchestrator = MultiAgentOrchestrator()
    
    msg = AgentMessage(
        source=AgentRole.ANALYZER,
        target=AgentRole.EXPLAINER,
        content="Analyze code"
    )
    
    orchestrator.add_agent_message(msg)
    assert len(orchestrator.message_queue) == 1


def test_get_messages_by_role():
    orchestrator = MultiAgentOrchestrator()
    
    msg1 = AgentMessage(
        source=AgentRole.ANALYZER,
        target=AgentRole.EXPLAINER,
        content="Msg 1"
    )
    msg2 = AgentMessage(
        source=AgentRole.VALIDATOR,
        target=AgentRole.EXPLAINER,
        content="Msg 2"
    )
    msg3 = AgentMessage(
        source=AgentRole.ANALYZER,
        target=AgentRole.VALIDATOR,
        content="Msg 3"
    )
    
    orchestrator.add_agent_message(msg1)
    orchestrator.add_agent_message(msg2)
    orchestrator.add_agent_message(msg3)
    
    explainer_msgs = orchestrator.get_messages(AgentRole.EXPLAINER)
    assert len(explainer_msgs) == 2
    assert all(msg.target == AgentRole.EXPLAINER for msg in explainer_msgs)


def test_explain_with_consensus_single_explanation():
    orchestrator = MultiAgentOrchestrator()
    
    def mock_explainer(code):
        return f"Explanation of: {code}"
    
    result = orchestrator.explain_with_consensus(
        code="x = 1",
        explainer_fn=mock_explainer,
        num_explanations=1
    )
    
    assert result.code == "x = 1"
    assert len(result.explanations) == 1
    assert result.confidence == 1.0  # Perfect agreement with 1 agent


def test_explain_with_consensus_multiple_identical():
    orchestrator = MultiAgentOrchestrator()
    
    def mock_explainer(code):
        return "This is an explanation"
    
    result = orchestrator.explain_with_consensus(
        code="x = 1",
        explainer_fn=mock_explainer,
        num_explanations=3
    )
    
    assert len(result.explanations) == 3
    assert result.confidence == 1.0  # All identical


def test_explain_with_consensus_multiple_different():
    orchestrator = MultiAgentOrchestrator()
    
    counter = [0]
    
    def mock_explainer(code):
        counter[0] += 1
        return f"Explanation variant {counter[0]}"
    
    result = orchestrator.explain_with_consensus(
        code="x = 1",
        explainer_fn=mock_explainer,
        num_explanations=3
    )
    
    assert len(result.explanations) == 3
    assert result.confidence < 1.0  # Disagreement reduces confidence


def test_explain_with_consensus_handles_failures():
    orchestrator = MultiAgentOrchestrator()
    
    counter = [0]
    
    def mock_explainer(code):
        counter[0] += 1
        if counter[0] == 2:
            raise ValueError("Agent 2 failed")
        return f"Explanation {counter[0]}"
    
    result = orchestrator.explain_with_consensus(
        code="x = 1",
        explainer_fn=mock_explainer,
        num_explanations=3
    )
    
    # Should have 2 explanations (one agent failed)
    assert len(result.explanations) == 2


def test_compute_consensus_empty():
    orchestrator = MultiAgentOrchestrator()
    
    consensus = orchestrator._compute_consensus([])
    assert consensus is None


def test_compute_consensus_longest():
    orchestrator = MultiAgentOrchestrator()
    
    explanations = [
        "Short",
        "This is a longer explanation",
        "Medium explanation"
    ]
    
    consensus = orchestrator._compute_consensus(explanations)
    assert consensus == "This is a longer explanation"


def test_compute_confidence_perfect():
    orchestrator = MultiAgentOrchestrator()
    
    explanations = ["Same"] * 5
    confidence = orchestrator._compute_confidence(explanations)
    assert confidence == 1.0


def test_compute_confidence_diverse():
    orchestrator = MultiAgentOrchestrator()
    
    explanations = [f"Explanation {i}" for i in range(5)]
    confidence = orchestrator._compute_confidence(explanations)
    assert 0.0 <= confidence < 1.0


def test_clear_message_queue():
    orchestrator = MultiAgentOrchestrator()
    
    msg = AgentMessage(
        source=AgentRole.ANALYZER,
        target=AgentRole.EXPLAINER,
        content="Test"
    )
    
    orchestrator.add_agent_message(msg)
    assert len(orchestrator.message_queue) == 1
    
    orchestrator.clear_message_queue()
    assert len(orchestrator.message_queue) == 0
