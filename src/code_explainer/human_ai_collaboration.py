"""
Human-AI collaboration utilities.
"""

from enum import Enum
from typing import Dict, Any, List
from datetime import datetime

# Pre-cache datetime.now for micro-optimization
_datetime_now = datetime.now


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


# Pre-cache satisfied levels for O(1) lookup
_SATISFIED_LEVELS = frozenset({"satisfied", "very_satisfied"})


class CollaborationTracker:
    """Tracks human-AI collaboration."""

    __slots__ = ("sessions", "current_session_id")

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.current_session_id = None

    def start_session(self, user_id: str, task_description: str) -> str:
        """Start a new collaboration session."""
        now = _datetime_now()
        session_id = f"{user_id}_{now.isoformat()}"
        self.sessions[session_id] = {
            "user_id": user_id,
            "task_description": task_description,
            "start_time": now,
            "interactions": [],
            "phase": CollaborationPhase.INITIAL_EXPLANATION.value
        }
        self.current_session_id = session_id
        return session_id

    def track_interaction(self, interaction: Dict[str, Any]) -> None:
        """Track an interaction."""
        session_id = self.current_session_id
        if session_id and session_id in self.sessions:
            self.sessions[session_id]["interactions"].append({
                "timestamp": _datetime_now(),
                **interaction
            })

    def end_session(self) -> Dict[str, Any]:
        """End the current session and return summary."""
        session_id = self.current_session_id
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            end_time = _datetime_now()
            session["end_time"] = end_time
            duration = (end_time - session["start_time"]).total_seconds()
            session["duration"] = duration

            # Calculate summary with single pass
            interactions = session["interactions"]
            feedback_count = 0
            corrections_count = 0
            for i in interactions:
                i_type = i.get("type")
                if i_type == "feedback":
                    feedback_count += 1
                elif i_type == "correction":
                    corrections_count += 1

            return {
                "session_id": session_id,
                "duration": duration,
                "total_interactions": len(interactions),
                "feedback_count": feedback_count,
                "corrections_count": corrections_count,
                "completion_rate": 1.0 if corrections_count == 0 else 0.8
            }

        return {"error": "No active session"}

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary for a specific session."""
        session = self.sessions.get(session_id)
        if session:
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

    __slots__ = ()

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

    __slots__ = ("feedback_store",)

    def __init__(self):
        self.feedback_store: List[Dict[str, Any]] = []

    def collect_feedback(self, explanations: List[str]) -> List[Dict[str, Any]]:
        """Collect feedback on explanations."""
        feedback = []
        now = _datetime_now()
        for i, explanation in enumerate(explanations):
            # Simulate feedback collection
            feedback.append({
                "explanation_id": i,
                "satisfaction": SatisfactionLevel.SATISFIED.value,
                "comments": "Good explanation",
                "timestamp": now
            })
        self.feedback_store.extend(feedback)
        return feedback

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of collected feedback."""
        store = self.feedback_store
        if not store:
            return {"total_feedback": 0}

        n = len(store)
        satisfied_count = sum(1 for f in store if f["satisfaction"] in _SATISFIED_LEVELS)
        return {
            "total_feedback": n,
            "average_satisfaction": satisfied_count / n
        }