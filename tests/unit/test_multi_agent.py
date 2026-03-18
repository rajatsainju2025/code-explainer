"""Tests for multi-agent orchestration system."""

import pytest
from code_explainer.multi_agent import (
    MultiAgentOrchestrator,
    AgentRole,
    AgentMessage,
    ExplanationResult,
)


class TestAgentRole:
    """Tests for AgentRole enum."""
    
    def test_agent_roles_exist(self):
        """Test that all expected agent roles exist."""
        assert AgentRole.ANALYZER.value == "analyzer"
        assert AgentRole.EXPLAINER.value == "explainer"
        assert AgentRole.VALIDATOR.value == "validator"
        assert AgentRole.JUDGE.value == "judge"


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        msg = AgentMessage(
            source=AgentRole.ANALYZER,
            target=AgentRole.EXPLAINER,
            content="Analysis complete"
        )
        assert msg.source == AgentRole.ANALYZER
        assert msg.target == AgentRole.EXPLAINER
        assert msg.content == "Analysis complete"
        assert msg.metadata is None
    
    def test_message_with_metadata(self):
        """Test message with metadata."""
        msg = AgentMessage(
            source=AgentRole.ANALYZER,
            target=AgentRole.EXPLAINER,
            content="Analysis",
            metadata={"confidence": 0.95}
        )
        assert msg.metadata == {"confidence": 0.95}


class TestExplanationResult:
    """Tests for ExplanationResult dataclass."""
    
    def test_result_creation(self):
        """Test basic result creation."""
        result = ExplanationResult(
            code="x = 1",
            explanations=["Sets x to 1"],
            confidence=0.9
        )
        assert result.code == "x = 1"
        assert len(result.explanations) == 1
        assert result.confidence == 0.9
        assert result.consensus_explanation is None


class TestMultiAgentOrchestrator:
    """Tests for MultiAgentOrchestrator class."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator for each test."""
        return MultiAgentOrchestrator()
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.num_agents == 3
        assert len(orchestrator.agents) == 4
        assert len(orchestrator.message_queue) == 0
    
    def test_add_agent_message(self, orchestrator):
        """Test adding messages to queue."""
        msg = AgentMessage(
            source=AgentRole.ANALYZER,
            target=AgentRole.EXPLAINER,
            content="Test"
        )
        orchestrator.add_agent_message(msg)
        assert len(orchestrator.message_queue) == 1
    
    def test_get_messages_for_role(self, orchestrator):
        """Test filtering messages by role."""
        msg1 = AgentMessage(
            source=AgentRole.ANALYZER,
            target=AgentRole.EXPLAINER,
            content="For explainer"
        )
        msg2 = AgentMessage(
            source=AgentRole.ANALYZER,
            target=AgentRole.VALIDATOR,
            content="For validator"
        )
        orchestrator.add_agent_message(msg1)
        orchestrator.add_agent_message(msg2)
        
        explainer_msgs = orchestrator.get_messages(AgentRole.EXPLAINER)
        assert len(explainer_msgs) == 1
        assert explainer_msgs[0].content == "For explainer"
    
    def test_compute_consensus_empty(self, orchestrator):
        """Test consensus with empty explanations."""
        result = orchestrator._compute_consensus([])
        assert result is None
    
    def test_compute_consensus_single(self, orchestrator):
        """Test consensus with single explanation."""
        result = orchestrator._compute_consensus(["Single explanation"])
        assert result == "Single explanation"
    
    def test_compute_consensus_returns_longest(self, orchestrator):
        """Test that consensus returns longest explanation."""
        explanations = [
            "Short",
            "Medium length explanation",
            "This is the longest explanation in the list"
        ]
        result = orchestrator._compute_consensus(explanations)
        assert result == "This is the longest explanation in the list"
    
    def test_compute_confidence_empty(self, orchestrator):
        """Test confidence with empty explanations."""
        confidence = orchestrator._compute_confidence([])
        assert confidence == 0.0
    
    def test_compute_confidence_identical(self, orchestrator):
        """Test confidence with identical explanations."""
        explanations = ["Same", "Same", "Same"]
        confidence = orchestrator._compute_confidence(explanations)
        assert confidence == 1.0
    
    def test_compute_confidence_diverse(self, orchestrator):
        """Test confidence with diverse explanations."""
        explanations = ["One", "Two", "Three"]
        confidence = orchestrator._compute_confidence(explanations)
        # 3 unique out of 3 = low confidence
        assert 0.0 <= confidence < 1.0
    
    def test_explain_with_consensus(self, orchestrator):
        """Test full explanation with consensus."""
        def mock_explainer(code):
            return f"Explanation for: {code}"
        
        result = orchestrator.explain_with_consensus(
            code="x = 1",
            explainer_fn=mock_explainer,
            num_explanations=3
        )
        
        assert isinstance(result, ExplanationResult)
        assert result.code == "x = 1"
        assert len(result.explanations) == 3
        assert result.consensus_explanation is not None
        assert result.metadata["num_agents"] == 3
    
    def test_explain_with_consensus_handles_failures(self, orchestrator):
        """Test that consensus handles agent failures gracefully."""
        call_count = 0
        
        def failing_explainer(code):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Simulated failure")
            return f"Explanation {call_count}"
        
        result = orchestrator.explain_with_consensus(
            code="x = 1",
            explainer_fn=failing_explainer,
            num_explanations=3
        )
        
        # Should have 2 explanations (one failed)
        assert len(result.explanations) == 2
    
    def test_clear_message_queue(self, orchestrator):
        """Test clearing message queue."""
        msg = AgentMessage(
            source=AgentRole.ANALYZER,
            target=AgentRole.EXPLAINER,
            content="Test"
        )
        orchestrator.add_agent_message(msg)
        assert len(orchestrator.message_queue) == 1
        
        orchestrator.clear_message_queue()
        assert len(orchestrator.message_queue) == 0
    
    def test_confidence_caching(self, orchestrator):
        """Test that confidence computation uses lru_cache."""
        explanations = ["Same", "Same", "Same"]
        
        # Call multiple times
        c1 = orchestrator._compute_confidence(explanations)
        c2 = orchestrator._compute_confidence(explanations)
        
        assert c1 == c2
        # The cached version should be used
        cache_info = orchestrator._compute_confidence_cached.cache_info()
        assert cache_info.hits >= 1
