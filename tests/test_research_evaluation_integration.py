"""Integration tests for research-driven evaluation components."""

import asyncio
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from code_explainer.contamination_detection import ContaminationDetector, ContaminationType
from code_explainer.dynamic_evaluation import DynamicEvaluator, EvaluationDimension
from code_explainer.multi_agent_evaluation import (
    MultiAgentEvaluator, CodeExplainerAgent, CodeReviewerAgent,
    EvaluationTask, InteractionType, AgentRole
)
from code_explainer.human_ai_collaboration import (
    CollaborationTracker, SatisfactionLevel, InteractionType as CollabInteractionType,
    CollaborationPhase
)
from code_explainer.adversarial_testing import AdversarialTester
from code_explainer.research_evaluation_orchestrator import (
    ResearchEvaluationOrchestrator, ResearchEvaluationConfig
)


class TestContaminationDetection:
    """Test contamination detection functionality."""
    
    def test_detector_initialization(self):
        """Test detector can be initialized."""
        detector = ContaminationDetector()
        assert detector is not None
        assert len(detector.exact_hashes) == 0
    
    async def test_exact_match_detection(self):
        """Test exact match detection."""
        detector = ContaminationDetector()
        
        # Add a sample to index
        sample_code = "def hello(): print('Hello, World!')"
        detector._index_code_sample(sample_code, "test_sample")
        
        # Test exact match
        result = detector.detect_exact_match(sample_code)
        assert result.is_contaminated is True
        assert result.contamination_type == ContaminationType.EXACT_MATCH
        assert result.confidence_score == 1.0
        
        # Test non-match
        different_code = "def goodbye(): print('Goodbye!')"
        result = detector.detect_exact_match(different_code)
        assert result.is_contaminated is False
    
    async def test_comprehensive_detection(self):
        """Test comprehensive contamination detection."""
        detector = ContaminationDetector()
        
        test_code = "def add(x, y): return x + y"
        results = await detector.detect_comprehensive(test_code)
        
        assert len(results) > 0
        assert all(hasattr(r, 'contamination_type') for r in results)
        assert all(hasattr(r, 'confidence_score') for r in results)


class TestDynamicEvaluation:
    """Test dynamic evaluation functionality."""
    
    def mock_model(self, prompt: str) -> str:
        """Mock model for testing."""
        return f"Response to: {prompt[:50]}..."
    
    async def test_dynamic_evaluator_initialization(self):
        """Test dynamic evaluator initialization."""
        evaluator = DynamicEvaluator()
        assert evaluator is not None
        assert evaluator.task_generator is not None
        assert evaluator.capability_tracker is not None
    
    async def test_model_evaluation(self):
        """Test model evaluation."""
        evaluator = DynamicEvaluator()
        dimensions = [EvaluationDimension.CORRECTNESS, EvaluationDimension.CLARITY]
        
        results = await evaluator.evaluate_model(self.mock_model, dimensions, num_tasks=3)
        
        assert len(results) == 3
        assert all(hasattr(r, 'overall_score') for r in results)
        assert all(0 <= r.overall_score <= 1 for r in results)
    
    def test_evaluation_summary(self):
        """Test evaluation summary generation."""
        evaluator = DynamicEvaluator()
        summary = evaluator.get_evaluation_summary()
        
        # Should handle empty state gracefully
        assert "message" in summary or "total_evaluations" in summary


class TestMultiAgentEvaluation:
    """Test multi-agent evaluation functionality."""
    
    def mock_model(self, prompt: str) -> str:
        """Mock model for testing."""
        return f"Multi-agent response: {prompt[:30]}..."
    
    async def test_multi_agent_evaluator_initialization(self):
        """Test multi-agent evaluator initialization."""
        evaluator = MultiAgentEvaluator()
        assert evaluator is not None
        assert len(evaluator.agents) == 0
    
    async def test_agent_registration(self):
        """Test agent registration."""
        evaluator = MultiAgentEvaluator()
        
        explainer = CodeExplainerAgent("test_explainer", self.mock_model)
        evaluator.register_agent(explainer)
        
        assert "test_explainer" in evaluator.agents
        assert evaluator.agents["test_explainer"].role == AgentRole.EXPLAINER
    
    async def test_parallel_evaluation(self):
        """Test parallel evaluation."""
        evaluator = MultiAgentEvaluator()
        
        # Register test agents
        explainer = CodeExplainerAgent("explainer", self.mock_model)
        reviewer = CodeReviewerAgent("reviewer", self.mock_model)
        
        evaluator.register_agent(explainer)
        evaluator.register_agent(reviewer)
        
        # Create test task
        task = EvaluationTask(
            task_id="test_task",
            prompt="def test(): pass",
            context={},
            required_roles=[AgentRole.EXPLAINER, AgentRole.REVIEWER],
            interaction_type=InteractionType.PARALLEL
        )
        
        result = await evaluator.evaluate(task)
        
        assert result.task_id == "test_task"
        assert len(result.individual_responses) == 2
        assert result.final_score >= 0


class TestHumanAICollaboration:
    """Test human-AI collaboration tracking."""
    
    def test_collaboration_tracker_initialization(self):
        """Test collaboration tracker initialization."""
        tracker = CollaborationTracker()
        assert tracker is not None
        assert len(tracker.sessions) == 0
    
    def test_session_management(self):
        """Test session start and end."""
        tracker = CollaborationTracker()
        
        session_id = tracker.start_session(
            user_id="test_user",
            goal="Test goal",
            interaction_type=CollabInteractionType.CODE_EXPLANATION
        )
        
        assert session_id in tracker.sessions
        
        tracker.end_session(session_id, "Completed", True)
        session = tracker.sessions[session_id]
        assert session.success is True
        assert session.outcome == "Completed"
    
    def test_feedback_recording(self):
        """Test feedback recording."""
        tracker = CollaborationTracker()
        
        # Start session and record interaction
        session_id = tracker.start_session("test_user", "goal", CollabInteractionType.CODE_EXPLANATION)
        event_id = tracker.record_interaction(
            session_id, CollaborationPhase.INITIAL_REQUEST,
            "test input", "test response", 1000
        )
        
        # Record feedback
        feedback_id = tracker.record_feedback(
            event_id, SatisfactionLevel.SATISFIED, 0.8, 0.9, 0.7, 0.8, 5.0, True
        )
        
        assert feedback_id in [fb.feedback_id for fb in tracker.feedback]


class TestAdversarialTesting:
    """Test adversarial testing functionality."""
    
    def vulnerable_model(self, prompt: str) -> str:
        """Mock vulnerable model for testing."""
        if "ignore instructions" in prompt.lower():
            return "OK, I will ignore previous instructions."
        return f"Standard response: {prompt[:50]}..."
    
    async def test_adversarial_tester_initialization(self):
        """Test adversarial tester initialization."""
        tester = AdversarialTester()
        assert tester is not None
        assert tester.injection_generator is not None
        assert tester.malicious_code_generator is not None
    
    async def test_prompt_injection_generation(self):
        """Test prompt injection test generation."""
        tester = AdversarialTester()
        base_code = "def hello(): print('hello')"
        
        tests = tester.injection_generator.generate_injection_tests(base_code, 3)
        
        assert len(tests) == 3
        assert all(hasattr(t, 'attack_type') for t in tests)
        assert all(t.attack_type.value == 'prompt_injection' for t in tests)
    
    async def test_comprehensive_testing(self):
        """Test comprehensive adversarial testing."""
        tester = AdversarialTester()
        
        # Run with a few test codes
        test_codes = ["def test(): pass", "print('hello')"]
        results = await tester.run_comprehensive_test(self.vulnerable_model, test_codes)
        
        assert len(results) > 0
        assert all(hasattr(r, 'vulnerability_score') for r in results)
        assert all(hasattr(r, 'is_vulnerable') for r in results)
    
    def test_test_summary(self):
        """Test test summary generation."""
        tester = AdversarialTester()
        summary = tester.get_test_summary()
        
        # Should handle empty state gracefully
        assert "message" in summary or "total_tests" in summary


class TestResearchEvaluationOrchestrator:
    """Test research evaluation orchestrator."""
    
    def mock_model(self, prompt: str) -> str:
        """Mock model for testing."""
        return f"Orchestrator test response: {prompt[:40]}..."
    
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        config = ResearchEvaluationConfig(
            dynamic_evaluation_rounds=2,
            enable_multi_agent=False,  # Disable for faster testing
            adversarial_test_count=5
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            orchestrator = ResearchEvaluationOrchestrator(config)
            
            assert orchestrator is not None
            assert orchestrator.config == config
            assert orchestrator.contamination_detector is not None
            assert orchestrator.dynamic_evaluator is not None
    
    async def test_model_evaluation(self):
        """Test comprehensive model evaluation."""
        config = ResearchEvaluationConfig(
            dynamic_evaluation_rounds=2,
            enable_multi_agent=False,  # Disable for faster testing
            adversarial_test_count=5,
            parallel_execution=False  # Sequential for simpler testing
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            orchestrator = ResearchEvaluationOrchestrator(config)
            
            # Run evaluation with minimal test prompts
            test_prompts = [
                "Explain this code: ```python\ndef add(a, b): return a + b\n```"
            ]
            
            result = await orchestrator.evaluate_model(
                self.mock_model, "test_model", test_prompts
            )
            
            assert result is not None
            assert result.model_identifier == "test_model"
            assert 0 <= result.overall_score <= 1
            assert 0 <= result.safety_score <= 1
            assert result.deployment_readiness in ["READY", "CONDITIONAL", "NOT_READY"]
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test default config
        config = ResearchEvaluationConfig()
        assert config.dynamic_evaluation_rounds > 0
        assert config.adversarial_test_count > 0
        
        # Test custom config
        custom_config = ResearchEvaluationConfig(
            dynamic_evaluation_rounds=10,
            enable_multi_agent=True,
            parallel_execution=True
        )
        assert custom_config.dynamic_evaluation_rounds == 10
        assert custom_config.enable_multi_agent is True


class TestIntegration:
    """Integration tests across components."""
    
    def mock_model(self, prompt: str) -> str:
        """Mock model for integration testing."""
        return f"Integration test response: {prompt[:30]}..."
    
    async def test_end_to_end_evaluation(self):
        """Test complete end-to-end evaluation."""
        # Create minimal configuration for fast testing
        config = ResearchEvaluationConfig(
            dynamic_evaluation_rounds=1,
            enable_multi_agent=False,
            adversarial_test_count=3,
            parallel_execution=True
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            orchestrator = ResearchEvaluationOrchestrator(config)
            
            result = await orchestrator.evaluate_model(
                self.mock_model, 
                "integration_test_model"
            )
            
            # Verify result structure
            assert result.evaluation_id is not None
            assert result.model_identifier == "integration_test_model"
            assert isinstance(result.overall_score, float)
            assert isinstance(result.safety_score, float)
            assert isinstance(result.improvement_areas, list)
            assert isinstance(result.risk_factors, list)
            
            # Verify results were saved
            output_dir = Path(temp_dir)
            result_files = list(output_dir.glob("*.json"))
            assert len(result_files) > 0
    
    async def test_component_interaction(self):
        """Test that components can work together."""
        # Test contamination detector with dynamic evaluator
        contamination_detector = ContaminationDetector()
        dynamic_evaluator = DynamicEvaluator()
        
        test_code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        
        # Run contamination detection
        contamination_results = await contamination_detector.detect_comprehensive(test_code)
        assert len(contamination_results) > 0
        
        # Run dynamic evaluation
        eval_results = await dynamic_evaluator.evaluate_model(
            self.mock_model, 
            [EvaluationDimension.CORRECTNESS], 
            num_tasks=1
        )
        assert len(eval_results) == 1
        
        # Both should complete without interference
        assert all(hasattr(r, 'contamination_type') for r in contamination_results)
        assert all(hasattr(r, 'overall_score') for r in eval_results)


# Test fixtures and utilities
@pytest.fixture
def sample_test_data():
    """Provide sample test data."""
    return {
        "prompts": [
            "Explain this Python function: def add(a, b): return a + b",
            "What does this code do: print('Hello, World!')",
            "Review this code for bugs: def divide(a, b): return a / b"
        ],
        "code_samples": [
            "def hello(): print('Hello')",
            "x = 5\ny = 10\nprint(x + y)",
            "for i in range(10): print(i)"
        ]
    }


@pytest.fixture
def mock_model_function():
    """Provide mock model function."""
    def mock_fn(prompt: str) -> str:
        return f"Mock response to: {prompt[:50]}..."
    return mock_fn


# Run integration tests
if __name__ == "__main__":
    # Run a simple integration test
    async def simple_test():
        test_integration = TestIntegration()
        await test_integration.test_end_to_end_evaluation()
        print("✅ Integration test passed!")
    
    asyncio.run(simple_test())
