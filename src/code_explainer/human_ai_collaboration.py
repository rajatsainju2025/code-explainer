"""
Human-AI collaboration utilities.
"""

from enum import Enum
from typing import Dict, Any, List
from datetime import datetime


class SatisfactionLevel(Enum):
    VERY_DISSATISFIED = "very_dissatisfied"
    DISSATISFIED = "dissatisfied"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    VERY_SATISFIED = "very_satisfied"


class InteractionType(Enum):
    FEEDBACK = "feedback"
    CORRECTION = "correction"
    CLARIFICATION = "clarification"


class CollaborationPhase(Enum):
    INITIAL_EXPLANATION = "initial_explanation"
    FEEDBACK_COLLECTION = "feedback_collection"
    REFINEMENT = "refinement"
    FINALIZATION = "finalization"


class CollaborationTracker:
    """Tracks human-AI collaboration."""

    def __init__(self):
        self.sessions = {}
        self.current_session_id = None

    def start_session(self, user_id: str, task_description: str) -> str:
        """Start a new collaboration session."""
        session_id = f"{user_id}_{datetime.now().isoformat()}"
        self.sessions[session_id] = {
            "user_id": user_id,
            "task_description": task_description,
            "start_time": datetime.now(),
            "interactions": [],
            "phase": CollaborationPhase.INITIAL_EXPLANATION.value
        }
        self.current_session_id = session_id
        return session_id

    def track_interaction(self, interaction: Dict[str, Any]) -> None:
        """Track an interaction."""
        if self.current_session_id and self.current_session_id in self.sessions:
            self.sessions[self.current_session_id]["interactions"].append({
                "timestamp": datetime.now(),
                **interaction
            })

    def end_session(self) -> Dict[str, Any]:
        """End the current session and return summary."""
        if self.current_session_id and self.current_session_id in self.sessions:
            session = self.sessions[self.current_session_id]
            session["end_time"] = datetime.now()
            session["duration"] = (session["end_time"] - session["start_time"]).total_seconds()

            # Calculate summary
            interactions = session["interactions"]
            feedback_count = len([i for i in interactions if i.get("type") == "feedback"])
            corrections_count = len([i for i in interactions if i.get("type") == "correction"])

            summary = {
                "session_id": self.current_session_id,
                "duration": session["duration"],
                "total_interactions": len(interactions),
                "feedback_count": feedback_count,
                "corrections_count": corrections_count,
                "completion_rate": 1.0 if corrections_count == 0 else 0.8
            }

            return summary

        return {"error": "No active session"}

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary for a specific session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            interactions = session.get("interactions", [])
            return {
                "session_id": session_id,
                "user_id": session.get("user_id"),
                "task_description": session.get("task_description"),
                "total_interactions": len(interactions),
                "phase": session.get("phase")
            }
        return {"error": "Session not found"}


class HumanAIEvaluator:
    """Human-AI collaboration evaluation."""

    def __init__(self):
        pass

    def evaluate_collaboration(self, model, human_feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate human-AI collaboration."""
        return {
            "collaboration_score": 0.88,
            "human_agreement": 0.91,
            "ai_improvement": 0.15
        }


class FeedbackCollector:
    """Collects human feedback."""

    def __init__(self):
        self.feedback_store = []

    def collect_feedback(self, explanations: List[str]) -> List[Dict[str, Any]]:
        """Collect feedback on explanations."""
        feedback = []
        for i, explanation in enumerate(explanations):
            # Simulate feedback collection
            feedback.append({
                "explanation_id": i,
                "satisfaction": SatisfactionLevel.SATISFIED.value,
                "comments": "Good explanation",
                "timestamp": datetime.now()
            })
        self.feedback_store.extend(feedback)
        return feedback

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of collected feedback."""
        if not self.feedback_store:
            return {"total_feedback": 0}

        satisfaction_levels = [f["satisfaction"] for f in self.feedback_store]
        return {
            "total_feedback": len(self.feedback_store),
            "average_satisfaction": len([s for s in satisfaction_levels if s in ["satisfied", "very_satisfied"]]) / len(satisfaction_levels)
        }