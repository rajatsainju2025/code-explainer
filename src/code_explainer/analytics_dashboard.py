"""Advanced analytics dashboard for usage metrics and insights."""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

@dataclass
class UsageEvent:
    """Usage event for analytics."""
    event_type: str
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[int] = None

@dataclass
class AnalyticsMetrics:
    """Analytics metrics."""
    total_users: int
    total_sessions: int
    total_events: int
    avg_session_duration: float
    top_features: List[Tuple[str, int]]
    user_engagement: Dict[str, float]
    performance_metrics: Dict[str, float]

class AnalyticsDashboard:
    """Advanced analytics dashboard."""

    def __init__(self):
        self.events: List[UsageEvent] = []
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)
        self.session_events: Dict[str, List[UsageEvent]] = defaultdict(list)

    def track_event(self, event: UsageEvent):
        """Track usage event."""
        self.events.append(event)

        if event.user_id and event.session_id:
            if event.session_id not in self.user_sessions[event.user_id]:
                self.user_sessions[event.user_id].append(event.session_id)

        if event.session_id:
            self.session_events[event.session_id].append(event)

        logger.debug(f"Tracked event: {event.event_type}")

    def get_metrics(self, days: int = 30) -> AnalyticsMetrics:
        """Get analytics metrics."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_events = [e for e in self.events if e.timestamp >= cutoff]

        # Basic counts
        total_users = len(set(e.user_id for e in recent_events if e.user_id))
        total_sessions = len(set(e.session_id for e in recent_events if e.session_id))
        total_events = len(recent_events)

        # Session duration
        session_durations = []
        for session_id, events in self.session_events.items():
            if events:
                start = min(e.timestamp for e in events)
                end = max(e.timestamp for e in events)
                duration = (end - start).total_seconds()
                session_durations.append(duration)

        avg_session_duration = statistics.mean(session_durations) if session_durations else 0

        # Top features
        feature_counts = defaultdict(int)
        for event in recent_events:
            feature_counts[event.event_type] += 1

        top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # User engagement
        user_engagement = {}
        for user_id, sessions in self.user_sessions.items():
            user_events = [e for e in recent_events if e.user_id == user_id]
            if user_events:
                engagement_score = len(user_events) / len(sessions) if sessions else 0
                user_engagement[user_id] = engagement_score

        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(recent_events)

        return AnalyticsMetrics(
            total_users=total_users,
            total_sessions=total_sessions,
            total_events=total_events,
            avg_session_duration=avg_session_duration,
            top_features=top_features,
            user_engagement=user_engagement,
            performance_metrics=performance_metrics
        )

    def _calculate_performance_metrics(self, events: List[UsageEvent]) -> Dict[str, float]:
        """Calculate performance metrics."""
        durations = [e.duration_ms for e in events if e.duration_ms]

        if not durations:
            return {"avg_response_time": 0, "p95_response_time": 0, "error_rate": 0}

        avg_response_time = statistics.mean(durations)
        p95_response_time = statistics.quantiles(durations, n=20)[18]  # 95th percentile

        error_events = [e for e in events if "error" in e.event_type.lower()]
        error_rate = len(error_events) / len(events) if events else 0

        return {
            "avg_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "error_rate": error_rate
        }

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights for specific user."""
        user_events = [e for e in self.events if e.user_id == user_id]

        if not user_events:
            return {"error": "User not found"}

        # Usage patterns
        event_types = defaultdict(int)
        for event in user_events:
            event_types[event.event_type] += 1

        # Session patterns
        sessions = self.user_sessions.get(user_id, [])
        avg_events_per_session = len(user_events) / len(sessions) if sessions else 0

        return {
            "total_events": len(user_events),
            "event_types": dict(event_types),
            "sessions_count": len(sessions),
            "avg_events_per_session": avg_events_per_session,
            "first_seen": min(e.timestamp for e in user_events).isoformat(),
            "last_seen": max(e.timestamp for e in user_events).isoformat()
        }

    def export_data(self, filepath: str):
        """Export analytics data."""
        data = {
            "events": [
                {
                    "event_type": e.event_type,
                    "user_id": e.user_id,
                    "session_id": e.session_id,
                    "timestamp": e.timestamp.isoformat(),
                    "metadata": e.metadata,
                    "duration_ms": e.duration_ms
                }
                for e in self.events
            ],
            "export_timestamp": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Analytics data exported to {filepath}")

# Example usage
def demo_analytics():
    """Demo analytics dashboard."""
    dashboard = AnalyticsDashboard()

    # Simulate events
    for i in range(10):
        event = UsageEvent(
            event_type="code_explanation" if i % 2 == 0 else "code_generation",
            user_id=f"user_{i % 3}",
            session_id=f"session_{i % 2}",
            timestamp=datetime.now(),
            duration_ms=500 + i * 50
        )
        dashboard.track_event(event)

    # Get metrics
    metrics = dashboard.get_metrics(days=1)
    print(f"Total users: {metrics.total_users}")
    print(f"Avg session duration: {metrics.avg_session_duration:.2f}s")
    print(f"Top features: {metrics.top_features}")

if __name__ == "__main__":
    demo_analytics()
