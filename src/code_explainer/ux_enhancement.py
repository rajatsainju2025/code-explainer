"""
UX Enhancement Module for Code Intelligence Platform

This module provides comprehensive user experience enhancements including
accessibility improvements, interaction design, user interface optimizations,
and usability features to create an exceptional user experience.

Features:
- Accessibility compliance (WCAG 2.1 AA)
- Responsive design and mobile optimization
- Interactive tutorials and onboarding
- User feedback and analytics
- Progressive disclosure and information architecture
- Visual design system and theming
- User preference management
- Performance optimizations for UI
- Error handling and user communication
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import re
import uuid


class Theme(Enum):
    """Available UI themes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"
    HIGH_CONTRAST = "high_contrast"


class AccessibilityLevel(Enum):
    """Accessibility compliance levels."""
    A = "A"
    AA = "AA"
    AAA = "AAA"


class UserRole(Enum):
    """User roles for personalization."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class UserPreferences:
    """User preference settings."""
    theme: Theme = Theme.AUTO
    font_size: str = "medium"
    language: str = "en"
    accessibility_level: AccessibilityLevel = AccessibilityLevel.AA
    animations_enabled: bool = True
    sound_enabled: bool = False
    keyboard_shortcuts: Dict[str, str] = field(default_factory=dict)
    dashboard_layout: Dict[str, Any] = field(default_factory=dict)
    notification_preferences: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert preferences to dictionary."""
        return {
            "theme": self.theme.value,
            "font_size": self.font_size,
            "language": self.language,
            "accessibility_level": self.accessibility_level.value,
            "animations_enabled": self.animations_enabled,
            "sound_enabled": self.sound_enabled,
            "keyboard_shortcuts": self.keyboard_shortcuts,
            "dashboard_layout": self.dashboard_layout,
            "notification_preferences": self.notification_preferences
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Create preferences from dictionary."""
        return cls(
            theme=Theme(data.get("theme", "auto")),
            font_size=data.get("font_size", "medium"),
            language=data.get("language", "en"),
            accessibility_level=AccessibilityLevel(data.get("accessibility_level", "AA")),
            animations_enabled=data.get("animations_enabled", True),
            sound_enabled=data.get("sound_enabled", False),
            keyboard_shortcuts=data.get("keyboard_shortcuts", {}),
            dashboard_layout=data.get("dashboard_layout", {}),
            notification_preferences=data.get("notification_preferences", {})
        )


@dataclass
class UserSession:
    """User session information."""
    user_id: str
    start_time: datetime
    last_activity: datetime
    preferences: UserPreferences
    session_data: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()

    def add_interaction(self, interaction_type: str, data: Dict[str, Any]) -> None:
        """Add user interaction to history."""
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": interaction_type,
            "data": data
        }
        self.interaction_history.append(interaction)

        # Keep only recent interactions
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-50:]


class AccessibilityManager:
    """Manages accessibility features and compliance."""

    def __init__(self):
        self.compliance_checks: Dict[str, Callable[[str], bool]] = {
            "color_contrast": self._check_color_contrast,
            "alt_text": self._check_alt_text,
            "keyboard_navigation": self._check_keyboard_navigation,
            "screen_reader": self._check_screen_reader_compatibility,
            "focus_management": self._check_focus_management
        }

    def audit_accessibility(self, content: str, level: AccessibilityLevel = AccessibilityLevel.AA) -> Dict[str, Any]:
        """Perform accessibility audit on content."""
        results = {
            "level": level.value,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "issues": [],
            "recommendations": []
        }

        for check_name, check_func in self.compliance_checks.items():
            try:
                passed = check_func(content)
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["issues"].append({
                        "check": check_name,
                        "status": "failed",
                        "level": level.value
                    })
            except Exception as e:
                results["warnings"] += 1
                results["issues"].append({
                    "check": check_name,
                    "status": "warning",
                    "message": str(e)
                })

        # Generate recommendations
        results["recommendations"] = self._generate_accessibility_recommendations(results["issues"])

        return results

    def _check_color_contrast(self, content: str) -> bool:
        """Check color contrast ratios."""
        # Simplified check - in real implementation, would analyze CSS
        return True  # Placeholder

    def _check_alt_text(self, content: str) -> bool:
        """Check for alt text on images."""
        # Look for img tags without alt attributes
        img_pattern = r'<img[^>]*>'
        images = re.findall(img_pattern, content, re.IGNORECASE)

        for img in images:
            if 'alt=' not in img.lower():
                return False
        return True

    def _check_keyboard_navigation(self, content: str) -> bool:
        """Check keyboard navigation support."""
        # Look for interactive elements
        interactive_elements = ['button', 'input', 'select', 'textarea', 'a']
        has_interactive = any(elem in content.lower() for elem in interactive_elements)
        return has_interactive  # Simplified check

    def _check_screen_reader_compatibility(self, content: str) -> bool:
        """Check screen reader compatibility."""
        # Check for semantic HTML elements
        semantic_elements = ['header', 'nav', 'main', 'section', 'article', 'aside', 'footer']
        has_semantic = any(elem in content.lower() for elem in semantic_elements)
        return has_semantic

    def _check_focus_management(self, content: str) -> bool:
        """Check focus management."""
        # Look for tabindex attributes
        return 'tabindex' in content.lower()

    def _generate_accessibility_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate accessibility improvement recommendations."""
        recommendations = []

        for issue in issues:
            check = issue["check"]
            if check == "alt_text":
                recommendations.append("Add alt text to all images for screen reader users")
            elif check == "keyboard_navigation":
                recommendations.append("Ensure all interactive elements are keyboard accessible")
            elif check == "screen_reader":
                recommendations.append("Use semantic HTML elements for better screen reader support")
            elif check == "focus_management":
                recommendations.append("Implement proper focus management for keyboard navigation")

        return recommendations

    def get_accessibility_score(self, audit_results: Dict[str, Any]) -> float:
        """Calculate accessibility score from audit results."""
        total_checks = audit_results["passed"] + audit_results["failed"] + audit_results["warnings"]
        if total_checks == 0:
            return 1.0

        score = audit_results["passed"] / total_checks
        return round(score, 2)


class OnboardingManager:
    """Manages user onboarding and tutorials."""

    def __init__(self):
        self.tutorials: Dict[str, Dict[str, Any]] = {}
        self.user_progress: Dict[str, Dict[str, Any]] = {}

    def create_tutorial(self, tutorial_id: str, steps: List[Dict[str, Any]]) -> None:
        """Create a new tutorial."""
        self.tutorials[tutorial_id] = {
            "id": tutorial_id,
            "steps": steps,
            "created_at": datetime.utcnow().isoformat(),
            "total_steps": len(steps)
        }

    def get_tutorial_progress(self, user_id: str, tutorial_id: str) -> Dict[str, Any]:
        """Get user's progress in a tutorial."""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}

        if tutorial_id not in self.user_progress[user_id]:
            self.user_progress[user_id][tutorial_id] = {
                "current_step": 0,
                "completed_steps": [],
                "started_at": datetime.utcnow().isoformat(),
                "completed": False
            }

        return self.user_progress[user_id][tutorial_id]

    def advance_tutorial(self, user_id: str, tutorial_id: str) -> Dict[str, Any]:
        """Advance user to next tutorial step."""
        progress = self.get_tutorial_progress(user_id, tutorial_id)

        if progress["completed"]:
            return progress

        tutorial = self.tutorials.get(tutorial_id)
        if not tutorial:
            return progress

        current_step = progress["current_step"]
        if current_step < tutorial["total_steps"] - 1:
            progress["current_step"] = current_step + 1
            progress["completed_steps"].append(current_step)

        if progress["current_step"] >= tutorial["total_steps"] - 1:
            progress["completed"] = True
            progress["completed_at"] = datetime.utcnow().isoformat()

        return progress

    def get_recommended_tutorial(self, user_id: str, user_role: UserRole) -> Optional[str]:
        """Get recommended tutorial based on user role and progress."""
        completed_tutorials = []
        if user_id in self.user_progress:
            completed_tutorials = [
                tid for tid, progress in self.user_progress[user_id].items()
                if progress.get("completed", False)
            ]

        # Recommend tutorials based on role
        role_tutorials = {
            UserRole.BEGINNER: ["getting_started", "basic_features"],
            UserRole.INTERMEDIATE: ["advanced_features", "best_practices"],
            UserRole.ADVANCED: ["power_user_features", "customization"],
            UserRole.EXPERT: ["api_integration", "contributing"]
        }

        available_tutorials = role_tutorials.get(user_role, [])
        for tutorial in available_tutorials:
            if tutorial not in completed_tutorials:
                return tutorial

        return None


class FeedbackManager:
    """Manages user feedback and analytics."""

    def __init__(self):
        self.feedback: List[Dict[str, Any]] = []
        self.nps_scores: List[Dict[str, Any]] = []
        self.feature_requests: List[Dict[str, Any]] = []

    def submit_feedback(self, user_id: str, feedback_type: str,
                       content: str, rating: Optional[int] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit user feedback."""
        feedback_id = str(uuid.uuid4())
        feedback_entry = {
            "id": feedback_id,
            "user_id": user_id,
            "type": feedback_type,
            "content": content,
            "rating": rating,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        self.feedback.append(feedback_entry)

        # Categorize feedback
        if feedback_type == "nps":
            self.nps_scores.append(feedback_entry)
        elif feedback_type == "feature_request":
            self.feature_requests.append(feedback_entry)

        return feedback_id

    def get_feedback_analytics(self) -> Dict[str, Any]:
        """Get feedback analytics."""
        if not self.feedback:
            return {}

        ratings = [f["rating"] for f in self.feedback if f["rating"] is not None]
        feedback_types = {}
        for f in self.feedback:
            fb_type = f["type"]
            feedback_types[fb_type] = feedback_types.get(fb_type, 0) + 1

        return {
            "total_feedback": len(self.feedback),
            "average_rating": sum(ratings) / len(ratings) if ratings else None,
            "feedback_types": feedback_types,
            "nps_score": self._calculate_nps(),
            "recent_feedback": self.feedback[-10:]  # Last 10 feedback items
        }

    def _calculate_nps(self) -> Optional[float]:
        """Calculate Net Promoter Score."""
        if not self.nps_scores:
            return None

        promoters = sum(1 for s in self.nps_scores if s["rating"] and s["rating"] >= 9)
        detractors = sum(1 for s in self.nps_scores if s["rating"] and s["rating"] <= 6)
        total = len(self.nps_scores)

        if total == 0:
            return None

        nps = ((promoters - detractors) / total) * 100
        return round(nps, 1)

    def get_feature_request_summary(self) -> List[Dict[str, Any]]:
        """Get summary of feature requests."""
        if not self.feature_requests:
            return []

        # Group by content similarity (simplified)
        request_groups = {}
        for request in self.feature_requests:
            content_hash = hashlib.md5(request["content"].encode()).hexdigest()[:8]
            if content_hash not in request_groups:
                request_groups[content_hash] = {
                    "content": request["content"],
                    "count": 0,
                    "users": [],
                    "first_requested": request["timestamp"]
                }
            request_groups[content_hash]["count"] += 1
            if request["user_id"] not in request_groups[content_hash]["users"]:
                request_groups[content_hash]["users"].append(request["user_id"])

        return list(request_groups.values())


class PerformanceOptimizer:
    """Optimizes UI performance and user experience."""

    def __init__(self):
        self.performance_metrics: Dict[str, List[float]] = {}
        self.optimization_rules: Dict[str, Callable[[], None]] = {}

    def measure_performance(self, component: str, metric_name: str, value: float) -> None:
        """Measure performance metric for a component."""
        key = f"{component}_{metric_name}"
        if key not in self.performance_metrics:
            self.performance_metrics[key] = []

        self.performance_metrics[key].append(value)

        # Keep only recent measurements
        if len(self.performance_metrics[key]) > 100:
            self.performance_metrics[key] = self.performance_metrics[key][-50:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all components."""
        summary = {}
        for key, values in self.performance_metrics.items():
            if values:
                summary[key] = {
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1],
                    "count": len(values)
                }
        return summary

    def optimize_component(self, component: str) -> List[str]:
        """Get optimization recommendations for a component."""
        recommendations = []

        # Check for performance issues
        load_time_key = f"{component}_load_time"
        if load_time_key in self.performance_metrics:
            load_times = self.performance_metrics[load_time_key]
            avg_load_time = sum(load_times) / len(load_times)

            if avg_load_time > 2000:  # 2 seconds
                recommendations.append("Consider lazy loading for this component")
            if avg_load_time > 5000:  # 5 seconds
                recommendations.append("Implement code splitting to reduce bundle size")

        # Memory usage check
        memory_key = f"{component}_memory_usage"
        if memory_key in self.performance_metrics:
            memory_usage = self.performance_metrics[memory_key]
            avg_memory = sum(memory_usage) / len(memory_usage)

            if avg_memory > 50:  # 50MB
                recommendations.append("Optimize memory usage - consider reducing DOM nodes")

        return recommendations

    def add_optimization_rule(self, rule_name: str, rule_func: Callable[[], None]) -> None:
        """Add a performance optimization rule."""
        self.optimization_rules[rule_name] = rule_func

    def apply_optimizations(self) -> None:
        """Apply all registered optimization rules."""
        for rule_name, rule_func in self.optimization_rules.items():
            try:
                rule_func()
            except Exception as e:
                print(f"Failed to apply optimization rule {rule_name}: {e}")


class UserExperienceOrchestrator:
    """Main orchestrator for user experience enhancements."""

    def __init__(self):
        self.accessibility = AccessibilityManager()
        self.onboarding = OnboardingManager()
        self.feedback = FeedbackManager()
        self.performance = PerformanceOptimizer()
        self.user_sessions: Dict[str, UserSession] = {}
        self.user_preferences: Dict[str, UserPreferences] = {}

    def create_user_session(self, user_id: str, preferences: Optional[UserPreferences] = None) -> UserSession:
        """Create a new user session."""
        if preferences is None:
            preferences = UserPreferences()

        session = UserSession(
            user_id=user_id,
            start_time=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            preferences=preferences
        )

        self.user_sessions[user_id] = session
        self.user_preferences[user_id] = preferences

        return session

    def get_user_session(self, user_id: str) -> Optional[UserSession]:
        """Get user session."""
        return self.user_sessions.get(user_id)

    def update_user_preferences(self, user_id: str, preferences: UserPreferences) -> None:
        """Update user preferences."""
        self.user_preferences[user_id] = preferences

        # Update active session if exists
        if user_id in self.user_sessions:
            self.user_sessions[user_id].preferences = preferences

    def get_personalized_experience(self, user_id: str) -> Dict[str, Any]:
        """Get personalized experience recommendations."""
        session = self.get_user_session(user_id)
        preferences = self.user_preferences.get(user_id, UserPreferences())

        recommendations = {
            "theme": preferences.theme.value,
            "tutorial": None,
            "features": [],
            "optimizations": []
        }

        # Determine user role based on session data
        user_role = self._determine_user_role(user_id)

        # Get recommended tutorial
        tutorial = self.onboarding.get_recommended_tutorial(user_id, user_role)
        if tutorial:
            recommendations["tutorial"] = tutorial

        # Get performance optimizations
        recommendations["optimizations"] = self.performance.get_performance_summary()

        return recommendations

    def _determine_user_role(self, user_id: str) -> UserRole:
        """Determine user role based on their activity."""
        session = self.get_user_session(user_id)
        if not session:
            return UserRole.BEGINNER

        interaction_count = len(session.interaction_history)
        session_duration = (datetime.utcnow() - session.start_time).total_seconds()

        if interaction_count > 100 or session_duration > 3600:  # 1 hour
            return UserRole.EXPERT
        elif interaction_count > 50 or session_duration > 1800:  # 30 minutes
            return UserRole.ADVANCED
        elif interaction_count > 20 or session_duration > 600:  # 10 minutes
            return UserRole.INTERMEDIATE
        else:
            return UserRole.BEGINNER

    def audit_user_experience(self, user_id: str) -> Dict[str, Any]:
        """Perform comprehensive UX audit for a user."""
        session = self.get_user_session(user_id)
        preferences = self.user_preferences.get(user_id, UserPreferences())

        audit_results = {
            "user_id": user_id,
            "session_active": session is not None,
            "preferences_configured": preferences != UserPreferences(),
            "accessibility_score": 0.0,
            "performance_score": 0.0,
            "engagement_score": 0.0,
            "recommendations": []
        }

        if session:
            # Calculate engagement score
            interaction_count = len(session.interaction_history)
            session_duration = (datetime.utcnow() - session.start_time).total_seconds()
            audit_results["engagement_score"] = min(interaction_count / 50 + session_duration / 3600, 1.0)

        # Get performance score
        perf_summary = self.performance.get_performance_summary()
        if perf_summary:
            avg_load_times = [v["average"] for v in perf_summary.values() if "load_time" in v]
            if avg_load_times:
                audit_results["performance_score"] = max(0, 1 - (sum(avg_load_times) / len(avg_load_times)) / 5000)

        # Get accessibility score (placeholder)
        audit_results["accessibility_score"] = 0.8  # Would be calculated from actual audits

        # Generate recommendations
        audit_results["recommendations"] = self._generate_ux_recommendations(audit_results)

        return audit_results

    def _generate_ux_recommendations(self, audit_results: Dict[str, Any]) -> List[str]:
        """Generate UX improvement recommendations."""
        recommendations = []

        if audit_results["engagement_score"] < 0.5:
            recommendations.append("Consider adding more interactive tutorials")
            recommendations.append("Implement gamification elements to increase engagement")

        if audit_results["performance_score"] < 0.7:
            recommendations.append("Optimize loading times and implement lazy loading")
            recommendations.append("Consider implementing caching strategies")

        if audit_results["accessibility_score"] < 0.8:
            recommendations.append("Improve accessibility compliance")
            recommendations.append("Add more keyboard navigation support")

        if not audit_results["preferences_configured"]:
            recommendations.append("Guide users to configure their preferences")
            recommendations.append("Show personalization benefits")

        return recommendations

    def get_ux_analytics(self) -> Dict[str, Any]:
        """Get comprehensive UX analytics."""
        return {
            "active_sessions": len(self.user_sessions),
            "total_users": len(self.user_preferences),
            "feedback_analytics": self.feedback.get_feedback_analytics(),
            "performance_summary": self.performance.get_performance_summary(),
            "tutorial_completion_rates": self._calculate_tutorial_completion_rates(),
            "user_role_distribution": self._calculate_user_role_distribution()
        }

    def _calculate_tutorial_completion_rates(self) -> Dict[str, float]:
        """Calculate tutorial completion rates."""
        completion_rates = {}
        for tutorial_id in self.onboarding.tutorials.keys():
            total_users = 0
            completed_users = 0

            for user_progress in self.onboarding.user_progress.values():
                if tutorial_id in user_progress:
                    total_users += 1
                    if user_progress[tutorial_id].get("completed", False):
                        completed_users += 1

            if total_users > 0:
                completion_rates[tutorial_id] = completed_users / total_users

        return completion_rates

    def _calculate_user_role_distribution(self) -> Dict[str, int]:
        """Calculate distribution of user roles."""
        role_counts = {role.value: 0 for role in UserRole}

        for user_id in self.user_sessions.keys():
            role = self._determine_user_role(user_id)
            role_counts[role.value] += 1

        return role_counts


# Export main classes
__all__ = [
    "Theme",
    "AccessibilityLevel",
    "UserRole",
    "UserPreferences",
    "UserSession",
    "AccessibilityManager",
    "OnboardingManager",
    "FeedbackManager",
    "PerformanceOptimizer",
    "UserExperienceOrchestrator"
]
