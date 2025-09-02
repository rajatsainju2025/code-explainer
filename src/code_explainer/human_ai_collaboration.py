"""Human-AI collaboration metrics for evaluating developer productivity and satisfaction."""

import asyncio
import json
import logging
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from collections import defaultdict

logger = logging.getLogger(__name__)


class CollaborationPhase(Enum):
    """Phases of human-AI collaboration."""
    INITIAL_REQUEST = "initial_request"
    AI_RESPONSE = "ai_response"
    HUMAN_FEEDBACK = "human_feedback"
    ITERATION = "iteration"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"
    MODIFICATION = "modification"


class InteractionType(Enum):
    """Types of human-AI interactions."""
    CODE_EXPLANATION = "code_explanation"
    CODE_REVIEW = "code_review"
    BUG_DETECTION = "bug_detection"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    LEARNING = "learning"
    TROUBLESHOOTING = "troubleshooting"


class SatisfactionLevel(Enum):
    """Human satisfaction levels."""
    VERY_DISSATISFIED = 1
    DISSATISFIED = 2
    NEUTRAL = 3
    SATISFIED = 4
    VERY_SATISFIED = 5


@dataclass
class InteractionEvent:
    """Single interaction event in human-AI collaboration."""
    event_id: str
    session_id: str
    phase: CollaborationPhase
    interaction_type: InteractionType
    timestamp: datetime
    duration_ms: int
    user_input: str
    ai_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserFeedback:
    """User feedback on AI responses."""
    feedback_id: str
    event_id: str
    satisfaction_level: SatisfactionLevel
    usefulness_score: float  # 0-1
    accuracy_score: float    # 0-1
    clarity_score: float     # 0-1
    completeness_score: float # 0-1
    time_saved_minutes: float
    would_recommend: bool
    comments: str = ""
    feedback_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CollaborationSession:
    """Complete collaboration session between human and AI."""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime]
    goal: str
    interaction_type: InteractionType
    events: List[InteractionEvent] = field(default_factory=list)
    feedback: List[UserFeedback] = field(default_factory=list)
    outcome: Optional[str] = None
    success: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductivityMetrics:
    """Productivity metrics for human-AI collaboration."""
    total_sessions: int
    successful_sessions: int
    average_session_duration: float
    average_iterations_per_session: float
    time_to_first_useful_response: float
    time_saved_per_session: float
    task_completion_rate: float
    user_satisfaction_avg: float
    recommendation_rate: float
    learning_curve_slope: float


@dataclass
class CollaborationAnalytics:
    """Analytics for human-AI collaboration patterns."""
    user_id: str
    analysis_period: Tuple[datetime, datetime]
    productivity_metrics: ProductivityMetrics
    interaction_patterns: Dict[str, Any]
    improvement_suggestions: List[str]
    collaboration_quality_score: float


class CollaborationTracker:
    """Tracks human-AI collaboration patterns and metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize collaboration tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sessions: Dict[str, CollaborationSession] = {}
        self.events: List[InteractionEvent] = []
        self.feedback: List[UserFeedback] = []
        self.user_profiles: Dict[str, Dict[str, Any]] = defaultdict(dict)
    
    def start_session(self, 
                     user_id: str, 
                     goal: str, 
                     interaction_type: InteractionType) -> str:
        """Start a new collaboration session.
        
        Args:
            user_id: User identifier
            goal: Session goal description
            interaction_type: Type of interaction
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        session = CollaborationSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            end_time=None,
            goal=goal,
            interaction_type=interaction_type
        )
        
        self.sessions[session_id] = session
        logger.info(f"Started collaboration session {session_id} for user {user_id}")
        
        return session_id
    
    def record_interaction(self,
                          session_id: str,
                          phase: CollaborationPhase,
                          user_input: str,
                          ai_response: str,
                          duration_ms: int,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Record an interaction event.
        
        Args:
            session_id: Session identifier
            phase: Collaboration phase
            user_input: User's input
            ai_response: AI's response
            duration_ms: Duration in milliseconds
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        event_id = str(uuid.uuid4())
        session = self.sessions[session_id]
        
        event = InteractionEvent(
            event_id=event_id,
            session_id=session_id,
            phase=phase,
            interaction_type=session.interaction_type,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            user_input=user_input,
            ai_response=ai_response,
            metadata=metadata or {}
        )
        
        session.events.append(event)
        self.events.append(event)
        
        logger.debug(f"Recorded interaction {event_id} in session {session_id}")
        
        return event_id
    
    def record_feedback(self,
                       event_id: str,
                       satisfaction_level: SatisfactionLevel,
                       usefulness_score: float,
                       accuracy_score: float,
                       clarity_score: float,
                       completeness_score: float,
                       time_saved_minutes: float,
                       would_recommend: bool,
                       comments: str = "") -> str:
        """Record user feedback.
        
        Args:
            event_id: Event identifier
            satisfaction_level: Overall satisfaction
            usefulness_score: Usefulness score (0-1)
            accuracy_score: Accuracy score (0-1)
            clarity_score: Clarity score (0-1)
            completeness_score: Completeness score (0-1)
            time_saved_minutes: Time saved in minutes
            would_recommend: Whether user would recommend
            comments: Additional comments
            
        Returns:
            Feedback ID
        """
        feedback_id = str(uuid.uuid4())
        
        feedback = UserFeedback(
            feedback_id=feedback_id,
            event_id=event_id,
            satisfaction_level=satisfaction_level,
            usefulness_score=usefulness_score,
            accuracy_score=accuracy_score,
            clarity_score=clarity_score,
            completeness_score=completeness_score,
            time_saved_minutes=time_saved_minutes,
            would_recommend=would_recommend,
            comments=comments
        )
        
        self.feedback.append(feedback)
        
        # Add feedback to corresponding session
        for session in self.sessions.values():
            for event in session.events:
                if event.event_id == event_id:
                    session.feedback.append(feedback)
                    break
        
        logger.info(f"Recorded feedback {feedback_id} for event {event_id}")
        
        return feedback_id
    
    def end_session(self, 
                   session_id: str, 
                   outcome: str, 
                   success: bool) -> None:
        """End a collaboration session.
        
        Args:
            session_id: Session identifier
            outcome: Session outcome description
            success: Whether session was successful
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.end_time = datetime.now()
        session.outcome = outcome
        session.success = success
        
        logger.info(f"Ended session {session_id} with success: {success}")
    
    def analyze_user_collaboration(self, 
                                  user_id: str, 
                                  days_back: int = 30) -> CollaborationAnalytics:
        """Analyze collaboration patterns for a user.
        
        Args:
            user_id: User identifier
            days_back: Number of days to analyze
            
        Returns:
            Collaboration analytics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Filter sessions for user and time period
        user_sessions = [
            session for session in self.sessions.values()
            if session.user_id == user_id and 
               session.start_time >= start_date and
               session.start_time <= end_date
        ]
        
        if not user_sessions:
            return CollaborationAnalytics(
                user_id=user_id,
                analysis_period=(start_date, end_date),
                productivity_metrics=ProductivityMetrics(
                    total_sessions=0,
                    successful_sessions=0,
                    average_session_duration=0,
                    average_iterations_per_session=0,
                    time_to_first_useful_response=0,
                    time_saved_per_session=0,
                    task_completion_rate=0,
                    user_satisfaction_avg=0,
                    recommendation_rate=0,
                    learning_curve_slope=0
                ),
                interaction_patterns={},
                improvement_suggestions=[],
                collaboration_quality_score=0
            )
        
        # Calculate productivity metrics
        productivity_metrics = self._calculate_productivity_metrics(user_sessions)
        
        # Analyze interaction patterns
        interaction_patterns = self._analyze_interaction_patterns(user_sessions)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            productivity_metrics, interaction_patterns
        )
        
        # Calculate overall collaboration quality score
        collaboration_quality_score = self._calculate_collaboration_quality(
            productivity_metrics, interaction_patterns
        )
        
        return CollaborationAnalytics(
            user_id=user_id,
            analysis_period=(start_date, end_date),
            productivity_metrics=productivity_metrics,
            interaction_patterns=interaction_patterns,
            improvement_suggestions=improvement_suggestions,
            collaboration_quality_score=collaboration_quality_score
        )
    
    def _calculate_productivity_metrics(self, sessions: List[CollaborationSession]) -> ProductivityMetrics:
        """Calculate productivity metrics from sessions.
        
        Args:
            sessions: List of collaboration sessions
            
        Returns:
            Productivity metrics
        """
        total_sessions = len(sessions)
        successful_sessions = sum(1 for s in sessions if s.success)
        
        # Session duration
        durations = []
        for session in sessions:
            if session.end_time:
                duration = (session.end_time - session.start_time).total_seconds() / 60  # minutes
                durations.append(duration)
        
        average_session_duration = statistics.mean(durations) if durations else 0
        
        # Iterations per session
        iterations = [len(session.events) for session in sessions]
        average_iterations_per_session = statistics.mean(iterations) if iterations else 0
        
        # Time to first useful response
        first_response_times = []
        for session in sessions:
            if session.events:
                first_event = session.events[0]
                first_response_times.append(first_event.duration_ms / 1000)  # seconds
        
        time_to_first_useful_response = statistics.mean(first_response_times) if first_response_times else 0
        
        # Time saved per session
        time_saved_values = []
        for session in sessions:
            session_time_saved = sum(fb.time_saved_minutes for fb in session.feedback)
            if session_time_saved > 0:
                time_saved_values.append(session_time_saved)
        
        time_saved_per_session = statistics.mean(time_saved_values) if time_saved_values else 0
        
        # Task completion rate
        task_completion_rate = successful_sessions / total_sessions if total_sessions > 0 else 0
        
        # User satisfaction metrics
        all_feedback = []
        for session in sessions:
            all_feedback.extend(session.feedback)
        
        if all_feedback:
            satisfaction_scores = [fb.satisfaction_level.value for fb in all_feedback]
            user_satisfaction_avg = statistics.mean(satisfaction_scores) / 5.0  # Normalize to 0-1
            
            recommendation_rate = sum(1 for fb in all_feedback if fb.would_recommend) / len(all_feedback)
        else:
            user_satisfaction_avg = 0
            recommendation_rate = 0
        
        # Learning curve (simplified)
        learning_curve_slope = self._calculate_learning_curve(sessions)
        
        return ProductivityMetrics(
            total_sessions=total_sessions,
            successful_sessions=successful_sessions,
            average_session_duration=average_session_duration,
            average_iterations_per_session=average_iterations_per_session,
            time_to_first_useful_response=time_to_first_useful_response,
            time_saved_per_session=time_saved_per_session,
            task_completion_rate=task_completion_rate,
            user_satisfaction_avg=user_satisfaction_avg,
            recommendation_rate=recommendation_rate,
            learning_curve_slope=learning_curve_slope
        )
    
    def _calculate_learning_curve(self, sessions: List[CollaborationSession]) -> float:
        """Calculate learning curve slope.
        
        Args:
            sessions: List of sessions ordered by time
            
        Returns:
            Learning curve slope
        """
        if len(sessions) < 2:
            return 0
        
        # Sort sessions by start time
        sorted_sessions = sorted(sessions, key=lambda s: s.start_time)
        
        # Calculate success rate over time
        success_rates = []
        window_size = min(5, len(sorted_sessions))
        
        for i in range(len(sorted_sessions) - window_size + 1):
            window_sessions = sorted_sessions[i:i + window_size]
            success_rate = sum(1 for s in window_sessions if s.success) / len(window_sessions)
            success_rates.append(success_rate)
        
        if len(success_rates) < 2:
            return 0
        
        # Simple linear regression slope
        n = len(success_rates)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(success_rates)
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(success_rates))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        return slope
    
    def _analyze_interaction_patterns(self, sessions: List[CollaborationSession]) -> Dict[str, Any]:
        """Analyze interaction patterns.
        
        Args:
            sessions: List of collaboration sessions
            
        Returns:
            Interaction patterns analysis
        """
        patterns = {}
        
        # Phase distribution
        phase_counts = defaultdict(int)
        for session in sessions:
            for event in session.events:
                phase_counts[event.phase.value] += 1
        
        total_events = sum(phase_counts.values())
        patterns["phase_distribution"] = {
            phase: count / total_events for phase, count in phase_counts.items()
        } if total_events > 0 else {}
        
        # Common interaction types
        type_counts = defaultdict(int)
        for session in sessions:
            type_counts[session.interaction_type.value] += 1
        
        patterns["interaction_type_distribution"] = dict(type_counts)
        
        # Average response times by phase
        phase_durations = defaultdict(list)
        for session in sessions:
            for event in session.events:
                phase_durations[event.phase.value].append(event.duration_ms)
        
        patterns["average_response_times"] = {
            phase: statistics.mean(durations) / 1000  # Convert to seconds
            for phase, durations in phase_durations.items()
            if durations
        }
        
        # Iteration patterns
        iteration_counts = [len(session.events) for session in sessions]
        patterns["iteration_statistics"] = {
            "mean": statistics.mean(iteration_counts) if iteration_counts else 0,
            "median": statistics.median(iteration_counts) if iteration_counts else 0,
            "max": max(iteration_counts) if iteration_counts else 0,
            "min": min(iteration_counts) if iteration_counts else 0
        }
        
        # Success patterns
        successful_sessions = [s for s in sessions if s.success]
        failed_sessions = [s for s in sessions if s.success is False]
        
        patterns["success_analysis"] = {
            "successful_avg_iterations": statistics.mean([len(s.events) for s in successful_sessions]) if successful_sessions else 0,
            "failed_avg_iterations": statistics.mean([len(s.events) for s in failed_sessions]) if failed_sessions else 0,
            "success_rate_by_type": self._calculate_success_rate_by_type(sessions)
        }
        
        return patterns
    
    def _calculate_success_rate_by_type(self, sessions: List[CollaborationSession]) -> Dict[str, float]:
        """Calculate success rate by interaction type.
        
        Args:
            sessions: List of sessions
            
        Returns:
            Success rates by interaction type
        """
        type_stats = defaultdict(lambda: {"total": 0, "successful": 0})
        
        for session in sessions:
            if session.success is not None:
                type_stats[session.interaction_type.value]["total"] += 1
                if session.success:
                    type_stats[session.interaction_type.value]["successful"] += 1
        
        success_rates = {}
        for interaction_type, stats in type_stats.items():
            if stats["total"] > 0:
                success_rates[interaction_type] = stats["successful"] / stats["total"]
        
        return success_rates
    
    def _generate_improvement_suggestions(self, 
                                        metrics: ProductivityMetrics, 
                                        patterns: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions.
        
        Args:
            metrics: Productivity metrics
            patterns: Interaction patterns
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Low satisfaction suggestions
        if metrics.user_satisfaction_avg < 0.6:
            suggestions.append("Consider improving AI response quality and relevance")
        
        # High iteration count suggestions
        avg_iterations = patterns.get("iteration_statistics", {}).get("mean", 0)
        if avg_iterations > 5:
            suggestions.append("AI could provide more comprehensive initial responses to reduce back-and-forth")
        
        # Low completion rate suggestions
        if metrics.task_completion_rate < 0.7:
            suggestions.append("Focus on better understanding user intent and providing actionable guidance")
        
        # Slow response time suggestions
        avg_response_time = patterns.get("average_response_times", {}).get("ai_response", 0)
        if avg_response_time > 10:  # seconds
            suggestions.append("Optimize AI response time for better user experience")
        
        # Learning curve suggestions
        if metrics.learning_curve_slope < 0.1:
            suggestions.append("Provide more educational content to help users learn and improve over time")
        
        # Low recommendation rate
        if metrics.recommendation_rate < 0.7:
            suggestions.append("Focus on increasing user trust through more accurate and helpful responses")
        
        return suggestions
    
    def _calculate_collaboration_quality(self, 
                                       metrics: ProductivityMetrics, 
                                       patterns: Dict[str, Any]) -> float:
        """Calculate overall collaboration quality score.
        
        Args:
            metrics: Productivity metrics
            patterns: Interaction patterns
            
        Returns:
            Collaboration quality score (0-1)
        """
        # Weighted combination of key metrics
        factors = [
            (metrics.user_satisfaction_avg, 0.3),
            (metrics.task_completion_rate, 0.25),
            (metrics.recommendation_rate, 0.2),
            (min(1.0, metrics.time_saved_per_session / 30), 0.15),  # Normalize time saved
            (max(0.0, metrics.learning_curve_slope * 10), 0.1)  # Normalize learning curve
        ]
        
        quality_score = sum(value * weight for value, weight in factors)
        
        return min(1.0, max(0.0, quality_score))
    
    def get_global_analytics(self) -> Dict[str, Any]:
        """Get global analytics across all users.
        
        Returns:
            Global analytics summary
        """
        if not self.sessions:
            return {"message": "No collaboration data available"}
        
        all_sessions = list(self.sessions.values())
        completed_sessions = [s for s in all_sessions if s.end_time is not None]
        
        # Global metrics
        total_users = len(set(s.user_id for s in all_sessions))
        total_sessions = len(all_sessions)
        successful_sessions = sum(1 for s in completed_sessions if s.success)
        
        # Overall satisfaction
        all_feedback = self.feedback
        avg_satisfaction = statistics.mean([fb.satisfaction_level.value for fb in all_feedback]) / 5.0 if all_feedback else 0
        
        # Popular interaction types
        type_counts = defaultdict(int)
        for session in all_sessions:
            type_counts[session.interaction_type.value] += 1
        
        most_popular_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "global_statistics": {
                "total_users": total_users,
                "total_sessions": total_sessions,
                "completed_sessions": len(completed_sessions),
                "success_rate": successful_sessions / len(completed_sessions) if completed_sessions else 0,
                "average_satisfaction": avg_satisfaction,
                "total_feedback_entries": len(all_feedback)
            },
            "popular_interaction_types": most_popular_types,
            "user_engagement": {
                "active_users_last_7_days": self._count_active_users(7),
                "active_users_last_30_days": self._count_active_users(30),
                "average_sessions_per_user": total_sessions / total_users if total_users > 0 else 0
            }
        }
    
    def _count_active_users(self, days_back: int) -> int:
        """Count active users in the last N days.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Number of active users
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        active_users = set()
        
        for session in self.sessions.values():
            if session.start_time >= cutoff_date:
                active_users.add(session.user_id)
        
        return len(active_users)
    
    def export_analytics(self, filepath: str) -> None:
        """Export analytics to JSON file.
        
        Args:
            filepath: Path to export file
        """
        try:
            analytics_data = {
                "export_timestamp": datetime.now().isoformat(),
                "global_analytics": self.get_global_analytics(),
                "session_count": len(self.sessions),
                "feedback_count": len(self.feedback),
                "user_count": len(self.user_profiles)
            }
            
            with open(filepath, 'w') as f:
                json.dump(analytics_data, f, indent=2)
            
            logger.info(f"Analytics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export analytics: {e}")


# Example usage
def demo_collaboration_tracking():
    """Demonstrate collaboration tracking system."""
    
    tracker = CollaborationTracker()
    
    # Simulate user sessions
    user_id = "developer_123"
    
    # Session 1: Code explanation
    session_id = tracker.start_session(
        user_id=user_id,
        goal="Understand complex algorithm",
        interaction_type=InteractionType.CODE_EXPLANATION
    )
    
    # Record interactions
    event_id = tracker.record_interaction(
        session_id=session_id,
        phase=CollaborationPhase.INITIAL_REQUEST,
        user_input="Can you explain how this sorting algorithm works?",
        ai_response="This is a merge sort algorithm that uses divide-and-conquer...",
        duration_ms=2500
    )
    
    # Record feedback
    tracker.record_feedback(
        event_id=event_id,
        satisfaction_level=SatisfactionLevel.SATISFIED,
        usefulness_score=0.8,
        accuracy_score=0.9,
        clarity_score=0.85,
        completeness_score=0.7,
        time_saved_minutes=15,
        would_recommend=True,
        comments="Very helpful explanation with good examples"
    )
    
    # End session
    tracker.end_session(
        session_id=session_id,
        outcome="Successfully understood the algorithm",
        success=True
    )
    
    # Analyze collaboration
    analytics = tracker.analyze_user_collaboration(user_id)
    
    print("Collaboration Analytics:")
    print(f"User: {analytics.user_id}")
    print(f"Total Sessions: {analytics.productivity_metrics.total_sessions}")
    print(f"Success Rate: {analytics.productivity_metrics.task_completion_rate:.2%}")
    print(f"Average Satisfaction: {analytics.productivity_metrics.user_satisfaction_avg:.2f}")
    print(f"Collaboration Quality: {analytics.collaboration_quality_score:.2f}")
    print(f"Improvement Suggestions: {analytics.improvement_suggestions}")
    
    return tracker, analytics


if __name__ == "__main__":
    demo_collaboration_tracking()
