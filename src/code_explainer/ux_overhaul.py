"""
User Experience Overhaul Module for Code Intelligence Platform

This module provides comprehensive UX improvements including modern UI components,
user feedback systems, personalization, accessibility features, and intuitive
interaction patterns for the code intelligence platform.

Features:
- Modern UI component library with dark/light themes
- User feedback and rating systems
- Personalization and user preferences
- Accessibility compliance (WCAG 2.1)
- Interactive tutorials and onboarding
- Progressive disclosure and information architecture
- Real-time notifications and status updates
"""

import json
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import os
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class Theme(Enum):
    """Available UI themes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"  # Follows system preference


class ComponentType(Enum):
    """Types of UI components."""
    BUTTON = "button"
    INPUT = "input"
    CARD = "card"
    MODAL = "modal"
    TOAST = "toast"
    PROGRESS_BAR = "progress_bar"
    CODE_EDITOR = "code_editor"
    CHART = "chart"


@dataclass
class UserPreferences:
    """User personalization preferences."""
    theme: Theme = Theme.AUTO
    language: str = "en"
    font_size: str = "medium"
    code_theme: str = "vs-dark"
    notifications_enabled: bool = True
    auto_save: bool = True
    keyboard_shortcuts: Dict[str, str] = field(default_factory=dict)
    dashboard_layout: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackItem:
    """User feedback item."""
    id: str
    user_id: str
    type: str  # "bug", "feature", "improvement", "praise"
    title: str
    description: str
    rating: Optional[int] = None  # 1-5 stars
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # "pending", "reviewed", "implemented"
    tags: List[str] = field(default_factory=list)


class UIComponent(ABC):
    """Abstract base class for UI components."""

    def __init__(self, component_id: str, properties: Dict[str, Any]):
        self.component_id = component_id
        self.properties = properties
        self.event_handlers: Dict[str, Callable] = {}

    @abstractmethod
    def render(self) -> Dict[str, Any]:
        """Render component to UI representation."""
        pass

    def add_event_handler(self, event: str, handler: Callable) -> None:
        """Add event handler to component."""
        self.event_handlers[event] = handler

    def trigger_event(self, event: str, *args, **kwargs) -> None:
        """Trigger component event."""
        if event in self.event_handlers:
            self.event_handlers[event](*args, **kwargs)


class ButtonComponent(UIComponent):
    """Modern button component with multiple styles."""

    def __init__(self, component_id: str, label: str, style: str = "primary",
                 size: str = "medium", disabled: bool = False):
        super().__init__(component_id, {
            "label": label,
            "style": style,
            "size": size,
            "disabled": disabled
        })

    def render(self) -> Dict[str, Any]:
        """Render button component."""
        return {
            "type": "button",
            "id": self.component_id,
            "properties": self.properties,
            "accessibility": {
                "role": "button",
                "aria-label": self.properties["label"]
            }
        }


class CodeEditorComponent(UIComponent):
    """Advanced code editor component."""

    def __init__(self, component_id: str, language: str, theme: str = "vs-dark",
                 read_only: bool = False, line_numbers: bool = True):
        super().__init__(component_id, {
            "language": language,
            "theme": theme,
            "read_only": read_only,
            "line_numbers": line_numbers,
            "value": ""
        })

    def set_value(self, value: str) -> None:
        """Set editor content."""
        self.properties["value"] = value

    def get_value(self) -> str:
        """Get editor content."""
        return self.properties["value"]

    def render(self) -> Dict[str, Any]:
        """Render code editor component."""
        return {
            "type": "code_editor",
            "id": self.component_id,
            "properties": self.properties,
            "accessibility": {
                "role": "textbox",
                "aria-multiline": True,
                "aria-label": f"Code editor for {self.properties['language']}"
            }
        }


class ProgressIndicatorComponent(UIComponent):
    """Progress indicator with multiple display modes."""

    def __init__(self, component_id: str, mode: str = "linear",
                 show_percentage: bool = True, animated: bool = True):
        super().__init__(component_id, {
            "mode": mode,
            "progress": 0,
            "show_percentage": show_percentage,
            "animated": animated
        })

    def set_progress(self, progress: float) -> None:
        """Set progress value (0-100)."""
        self.properties["progress"] = max(0, min(100, progress))

    def render(self) -> Dict[str, Any]:
        """Render progress indicator."""
        return {
            "type": "progress_indicator",
            "id": self.component_id,
            "properties": self.properties,
            "accessibility": {
                "role": "progressbar",
                "aria-valuenow": self.properties["progress"],
                "aria-valuemin": 0,
                "aria-valuemax": 100
            }
        }


class NotificationSystem:
    """Real-time notification system."""

    def __init__(self):
        self.notifications: List[Dict[str, Any]] = []
        self.listeners: List[Callable] = []

    def add_notification(self, title: str, message: str, type: str = "info",
                        duration: Optional[int] = None, actions: Optional[List[Dict[str, str]]] = None) -> str:
        """Add a new notification."""
        notification_id = hashlib.md5(f"{title}{message}{time.time()}".encode()).hexdigest()[:8]

        notification = {
            "id": notification_id,
            "title": title,
            "message": message,
            "type": type,  # "info", "success", "warning", "error"
            "timestamp": time.time(),
            "duration": duration,
            "actions": actions or [],
            "read": False
        }

        self.notifications.append(notification)
        self._notify_listeners(notification)

        return notification_id

    def mark_as_read(self, notification_id: str) -> None:
        """Mark notification as read."""
        for notification in self.notifications:
            if notification["id"] == notification_id:
                notification["read"] = True
                break

    def get_unread_count(self) -> int:
        """Get count of unread notifications."""
        return len([n for n in self.notifications if not n["read"]])

    def get_notifications(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get notifications, optionally limited."""
        notifications = sorted(self.notifications, key=lambda x: x["timestamp"], reverse=True)
        return notifications[:limit] if limit else notifications

    def add_listener(self, listener: Callable) -> None:
        """Add notification listener."""
        self.listeners.append(listener)

    def _notify_listeners(self, notification: Dict[str, Any]) -> None:
        """Notify all listeners of new notification."""
        for listener in self.listeners:
            try:
                listener(notification)
            except Exception as e:
                logger.error(f"Notification listener error: {e}")


class UserFeedbackSystem:
    """Comprehensive user feedback collection and management."""

    def __init__(self, storage_path: str = ".user_feedback"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.feedback_items: List[FeedbackItem] = []
        self._load_feedback()

    def submit_feedback(self, user_id: str, feedback_type: str, title: str,
                       description: str, rating: Optional[int] = None,
                       tags: Optional[List[str]] = None) -> str:
        """Submit user feedback."""
        feedback_id = hashlib.md5(f"{user_id}{title}{time.time()}".encode()).hexdigest()

        feedback = FeedbackItem(
            id=feedback_id,
            user_id=user_id,
            type=feedback_type,
            title=title,
            description=description,
            rating=rating,
            tags=tags or []
        )

        self.feedback_items.append(feedback)
        self._save_feedback()

        return feedback_id

    def get_feedback(self, user_id: Optional[str] = None,
                    feedback_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get feedback items with optional filtering."""
        items = self.feedback_items

        if user_id:
            items = [f for f in items if f.user_id == user_id]

        if feedback_type:
            items = [f for f in items if f.type == feedback_type]

        return [self._feedback_to_dict(f) for f in items]

    def update_feedback_status(self, feedback_id: str, status: str) -> bool:
        """Update feedback status."""
        for feedback in self.feedback_items:
            if feedback.id == feedback_id:
                feedback.status = status
                self._save_feedback()
                return True
        return False

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        total = len(self.feedback_items)
        if total == 0:
            return {"total_feedback": 0}

        ratings = [f.rating for f in self.feedback_items if f.rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else None

        type_counts = {}
        for feedback in self.feedback_items:
            type_counts[feedback.type] = type_counts.get(feedback.type, 0) + 1

        return {
            "total_feedback": total,
            "average_rating": avg_rating,
            "type_distribution": type_counts,
            "status_distribution": {
                status: len([f for f in self.feedback_items if f.status == status])
                for status in ["pending", "reviewed", "implemented"]
            }
        }

    def _feedback_to_dict(self, feedback: FeedbackItem) -> Dict[str, Any]:
        """Convert feedback item to dictionary."""
        return {
            "id": feedback.id,
            "user_id": feedback.user_id,
            "type": feedback.type,
            "title": feedback.title,
            "description": feedback.description,
            "rating": feedback.rating,
            "timestamp": feedback.timestamp,
            "status": feedback.status,
            "tags": feedback.tags
        }

    def _save_feedback(self) -> None:
        """Save feedback to storage."""
        data = [self._feedback_to_dict(f) for f in self.feedback_items]
        with open(os.path.join(self.storage_path, "feedback.json"), 'w') as f:
            json.dump(data, f, indent=2)

    def _load_feedback(self) -> None:
        """Load feedback from storage."""
        feedback_file = os.path.join(self.storage_path, "feedback.json")
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r') as f:
                    data = json.load(f)
                self.feedback_items = [
                    FeedbackItem(
                        id=item["id"],
                        user_id=item["user_id"],
                        type=item["type"],
                        title=item["title"],
                        description=item["description"],
                        rating=item.get("rating"),
                        timestamp=item["timestamp"],
                        status=item.get("status", "pending"),
                        tags=item.get("tags", [])
                    )
                    for item in data
                ]
            except Exception as e:
                logger.error(f"Failed to load feedback: {e}")


class PersonalizationEngine:
    """User personalization and preference management."""

    def __init__(self, storage_path: str = ".user_prefs"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.user_preferences: Dict[str, UserPreferences] = {}
        self._load_preferences()

    def get_user_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences, creating defaults if needed."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreferences()
            self._save_preferences()
        return self.user_preferences[user_id]

    def update_user_preferences(self, user_id: str, updates: Dict[str, Any]) -> None:
        """Update user preferences."""
        prefs = self.get_user_preferences(user_id)

        for key, value in updates.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)

        self._save_preferences()

    def get_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Get personalized recommendations based on user behavior."""
        prefs = self.get_user_preferences(user_id)

        recommendations = {
            "suggested_themes": [prefs.theme.value],
            "keyboard_shortcuts": prefs.keyboard_shortcuts,
            "dashboard_layout": prefs.dashboard_layout
        }

        # Add theme recommendations
        if prefs.theme == Theme.AUTO:
            recommendations["suggested_themes"].extend(["dark", "light"])

        return recommendations

    def _save_preferences(self) -> None:
        """Save user preferences to storage."""
        data = {}
        for user_id, prefs in self.user_preferences.items():
            data[user_id] = {
                "theme": prefs.theme.value,
                "language": prefs.language,
                "font_size": prefs.font_size,
                "code_theme": prefs.code_theme,
                "notifications_enabled": prefs.notifications_enabled,
                "auto_save": prefs.auto_save,
                "keyboard_shortcuts": prefs.keyboard_shortcuts,
                "dashboard_layout": prefs.dashboard_layout
            }

        with open(os.path.join(self.storage_path, "preferences.json"), 'w') as f:
            json.dump(data, f, indent=2)

    def _load_preferences(self) -> None:
        """Load user preferences from storage."""
        prefs_file = os.path.join(self.storage_path, "preferences.json")
        if os.path.exists(prefs_file):
            try:
                with open(prefs_file, 'r') as f:
                    data = json.load(f)

                for user_id, prefs_data in data.items():
                    self.user_preferences[user_id] = UserPreferences(
                        theme=Theme(prefs_data.get("theme", "auto")),
                        language=prefs_data.get("language", "en"),
                        font_size=prefs_data.get("font_size", "medium"),
                        code_theme=prefs_data.get("code_theme", "vs-dark"),
                        notifications_enabled=prefs_data.get("notifications_enabled", True),
                        auto_save=prefs_data.get("auto_save", True),
                        keyboard_shortcuts=prefs_data.get("keyboard_shortcuts", {}),
                        dashboard_layout=prefs_data.get("dashboard_layout", {})
                    )
            except Exception as e:
                logger.error(f"Failed to load preferences: {e}")


class AccessibilityManager:
    """Accessibility compliance and features."""

    def __init__(self):
        self.screen_reader_enabled = False
        self.high_contrast_mode = False
        self.keyboard_navigation = True
        self.focus_management = True

    def enable_screen_reader_support(self) -> None:
        """Enable screen reader support."""
        self.screen_reader_enabled = True

    def enable_high_contrast_mode(self) -> None:
        """Enable high contrast mode."""
        self.high_contrast_mode = True

    def get_accessibility_settings(self) -> Dict[str, Any]:
        """Get current accessibility settings."""
        return {
            "screen_reader_enabled": self.screen_reader_enabled,
            "high_contrast_mode": self.high_contrast_mode,
            "keyboard_navigation": self.keyboard_navigation,
            "focus_management": self.focus_management
        }

    def apply_accessibility_overrides(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Apply accessibility overrides to component."""
        if self.screen_reader_enabled:
            component.setdefault("accessibility", {})
            component["accessibility"]["aria-live"] = "polite"

        if self.high_contrast_mode:
            component.setdefault("styling", {})
            component["styling"]["contrast"] = "high"

        return component


class TutorialSystem:
    """Interactive tutorial and onboarding system."""

    def __init__(self):
        self.tutorials: Dict[str, Dict[str, Any]] = {}
        self.user_progress: Dict[str, Dict[str, Any]] = {}

    def create_tutorial(self, tutorial_id: str, title: str, steps: List[Dict[str, Any]]) -> None:
        """Create a new tutorial."""
        self.tutorials[tutorial_id] = {
            "title": title,
            "steps": steps,
            "total_steps": len(steps)
        }

    def get_tutorial_progress(self, user_id: str, tutorial_id: str) -> Dict[str, Any]:
        """Get user's progress in a tutorial."""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}

        if tutorial_id not in self.user_progress[user_id]:
            self.user_progress[user_id][tutorial_id] = {
                "current_step": 0,
                "completed": False,
                "started_at": time.time()
            }

        progress = self.user_progress[user_id][tutorial_id]
        tutorial = self.tutorials.get(tutorial_id, {})

        return {
            "tutorial_id": tutorial_id,
            "title": tutorial.get("title", ""),
            "current_step": progress["current_step"],
            "total_steps": tutorial.get("total_steps", 0),
            "completed": progress["completed"],
            "progress_percentage": (progress["current_step"] / tutorial.get("total_steps", 1)) * 100
        }

    def advance_tutorial(self, user_id: str, tutorial_id: str) -> Dict[str, Any]:
        """Advance user to next tutorial step."""
        progress = self.get_tutorial_progress(user_id, tutorial_id)

        if progress["completed"]:
            return progress

        current_step = progress["current_step"]
        total_steps = progress["total_steps"]

        if current_step < total_steps:
            self.user_progress[user_id][tutorial_id]["current_step"] = current_step + 1

            if current_step + 1 >= total_steps:
                self.user_progress[user_id][tutorial_id]["completed"] = True
                self.user_progress[user_id][tutorial_id]["completed_at"] = time.time()

        return self.get_tutorial_progress(user_id, tutorial_id)

    def get_available_tutorials(self) -> List[Dict[str, Any]]:
        """Get list of available tutorials."""
        return [
            {
                "id": tutorial_id,
                "title": tutorial.get("title", ""),
                "steps": tutorial.get("total_steps", 0)
            }
            for tutorial_id, tutorial in self.tutorials.items()
        ]


class UXOrchestrator:
    """Main orchestrator for UX overhaul features."""

    def __init__(self):
        self.notification_system = NotificationSystem()
        self.feedback_system = UserFeedbackSystem()
        self.personalization = PersonalizationEngine()
        self.accessibility = AccessibilityManager()
        self.tutorials = TutorialSystem()
        self.components: Dict[str, UIComponent] = {}

    def create_component(self, component_type: ComponentType, component_id: str,
                        **kwargs) -> UIComponent:
        """Create a UI component."""
        if component_type == ComponentType.BUTTON:
            component = ButtonComponent(component_id, **kwargs)
        elif component_type == ComponentType.CODE_EDITOR:
            component = CodeEditorComponent(component_id, **kwargs)
        elif component_type == ComponentType.PROGRESS_BAR:
            component = ProgressIndicatorComponent(component_id, **kwargs)
        else:
            raise ValueError(f"Unsupported component type: {component_type}")

        self.components[component_id] = component
        return component

    def render_ui(self, user_id: str) -> Dict[str, Any]:
        """Render complete UI for user."""
        preferences = self.personalization.get_user_preferences(user_id)

        ui_config = {
            "theme": preferences.theme.value,
            "language": preferences.language,
            "components": [comp.render() for comp in self.components.values()],
            "notifications": self.notification_system.get_notifications(limit=10),
            "accessibility": self.accessibility.get_accessibility_settings(),
            "tutorials": self.tutorials.get_available_tutorials()
        }

        return ui_config

    def get_user_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get personalized user dashboard."""
        preferences = self.personalization.get_user_preferences(user_id)
        feedback_stats = self.feedback_system.get_feedback_stats()
        tutorial_progress = [
            self.tutorials.get_tutorial_progress(user_id, tutorial_id)
            for tutorial_id in self.tutorials.tutorials.keys()
        ]

        return {
            "user_id": user_id,
            "preferences": {
                "theme": preferences.theme.value,
                "language": preferences.language,
                "notifications_enabled": preferences.notifications_enabled
            },
            "feedback_stats": feedback_stats,
            "tutorial_progress": tutorial_progress,
            "unread_notifications": self.notification_system.get_unread_count(),
            "recommendations": self.personalization.get_recommendations(user_id)
        }


# Export main classes
__all__ = [
    "Theme",
    "ComponentType",
    "UserPreferences",
    "FeedbackItem",
    "UIComponent",
    "ButtonComponent",
    "CodeEditorComponent",
    "ProgressIndicatorComponent",
    "NotificationSystem",
    "UserFeedbackSystem",
    "PersonalizationEngine",
    "AccessibilityManager",
    "TutorialSystem",
    "UXOrchestrator"
]
