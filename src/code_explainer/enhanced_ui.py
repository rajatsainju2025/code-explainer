"""Enhanced user experience and interface improvements."""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class ThemeMode(Enum):
    """UI theme modes."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class InteractionMode(Enum):
    """User interaction modes."""
    INTERACTIVE = "interactive"
    BATCH = "batch"
    API = "api"
    CLI = "cli"


@dataclass
class UserPreferences:
    """User preferences and settings."""
    theme: ThemeMode = ThemeMode.AUTO
    language: str = "en"
    interaction_mode: InteractionMode = InteractionMode.INTERACTIVE
    auto_save: bool = True
    show_tooltips: bool = True
    enable_animations: bool = True
    default_strategy: str = "enhanced_rag"
    max_history: int = 100
    custom_shortcuts: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "theme": self.theme.value,
            "language": self.language,
            "interaction_mode": self.interaction_mode.value,
            "auto_save": self.auto_save,
            "show_tooltips": self.show_tooltips,
            "enable_animations": self.enable_animations,
            "default_strategy": self.default_strategy,
            "max_history": self.max_history,
            "custom_shortcuts": self.custom_shortcuts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        """Create from dictionary."""
        return cls(
            theme=ThemeMode(data.get("theme", "auto")),
            language=data.get("language", "en"),
            interaction_mode=InteractionMode(data.get("interaction_mode", "interactive")),
            auto_save=data.get("auto_save", True),
            show_tooltips=data.get("show_tooltips", True),
            enable_animations=data.get("enable_animations", True),
            default_strategy=data.get("default_strategy", "enhanced_rag"),
            max_history=data.get("max_history", 100),
            custom_shortcuts=data.get("custom_shortcuts", {})
        )


@dataclass
class ProgressInfo:
    """Progress tracking information."""
    current: int
    total: int
    message: str = ""
    percentage: float = 0.0
    estimated_time_remaining: Optional[float] = None
    
    def __post_init__(self):
        if self.total > 0:
            self.percentage = (self.current / self.total) * 100


class ProgressTracker:
    """Progress tracking for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            description: Operation description
        """
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = time.time()
        self.step_times: List[float] = []
        self.callbacks: List[Callable[[ProgressInfo], None]] = []
        
    def add_callback(self, callback: Callable[[ProgressInfo], None]) -> None:
        """Add progress callback.
        
        Args:
            callback: Callback function
        """
        self.callbacks.append(callback)
    
    def update(self, step_increment: int = 1, message: str = "") -> ProgressInfo:
        """Update progress.
        
        Args:
            step_increment: Number of steps to increment
            message: Status message
            
        Returns:
            Current progress info
        """
        self.current_step += step_increment
        self.step_times.append(time.time())
        
        # Calculate estimated time remaining
        estimated_time = None
        if len(self.step_times) > 1:
            avg_step_time = (self.step_times[-1] - self.step_times[0]) / len(self.step_times)
            remaining_steps = self.total_steps - self.current_step
            estimated_time = avg_step_time * remaining_steps
        
        progress = ProgressInfo(
            current=self.current_step,
            total=self.total_steps,
            message=message or self.description,
            estimated_time_remaining=estimated_time
        )
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
        
        return progress
    
    def finish(self, message: str = "Complete") -> ProgressInfo:
        """Mark progress as finished.
        
        Args:
            message: Completion message
            
        Returns:
            Final progress info
        """
        self.current_step = self.total_steps
        return self.update(0, message)


class UserInterface:
    """Enhanced user interface manager."""
    
    def __init__(self, preferences: Optional[UserPreferences] = None):
        """Initialize UI manager.
        
        Args:
            preferences: User preferences
        """
        self.preferences = preferences or UserPreferences()
        self.history: List[Dict[str, Any]] = []
        self.shortcuts: Dict[str, Callable] = {}
        self.progress_trackers: Dict[str, ProgressTracker] = {}
        
    def print_welcome(self) -> None:
        """Print welcome message."""
        welcome_text = f"""
╔══════════════════════════════════════════════════════════════╗
║                     Code Explainer v2.0                     ║
║              Enhanced AI-Powered Code Analysis              ║
╚══════════════════════════════════════════════════════════════╝

Welcome! Current theme: {self.preferences.theme.value}
Type 'help' for available commands or 'settings' to configure preferences.
        """
        self.print_formatted(welcome_text, style="info")
    
    def print_formatted(
        self,
        text: str,
        style: str = "normal",
        indent: int = 0,
        prefix: str = ""
    ) -> None:
        """Print formatted text with styling.
        
        Args:
            text: Text to print
            style: Style type (normal, info, warning, error, success)
            indent: Indentation level
            prefix: Text prefix
        """
        # Color codes for different styles
        colors = {
            "normal": "",
            "info": "\033[94m",      # Blue
            "warning": "\033[93m",   # Yellow
            "error": "\033[91m",     # Red
            "success": "\033[92m",   # Green
            "reset": "\033[0m"
        }
        
        # Apply theme-based adjustments
        if self.preferences.theme == ThemeMode.DARK:
            # Adjust colors for dark theme
            colors["info"] = "\033[96m"  # Cyan
        
        color = colors.get(style, "")
        reset = colors["reset"]
        indent_str = "  " * indent
        
        # Format text
        formatted_text = f"{indent_str}{prefix}{color}{text}{reset}"
        print(formatted_text)
    
    def get_user_input(
        self,
        prompt: str,
        default: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        validate: Optional[Callable[[str], bool]] = None
    ) -> str:
        """Get user input with enhancements.
        
        Args:
            prompt: Input prompt
            default: Default value
            suggestions: Input suggestions
            validate: Validation function
            
        Returns:
            User input
        """
        # Show suggestions if available
        if suggestions and self.preferences.show_tooltips:
            self.print_formatted(f"Suggestions: {', '.join(suggestions)}", style="info", indent=1)
        
        # Build prompt
        full_prompt = prompt
        if default:
            full_prompt += f" [{default}]"
        full_prompt += ": "
        
        while True:
            try:
                user_input = input(full_prompt).strip()
                
                # Use default if empty
                if not user_input and default:
                    user_input = default
                
                # Validate input
                if validate and not validate(user_input):
                    self.print_formatted("Invalid input. Please try again.", style="error")
                    continue
                
                # Add to history
                self.add_to_history("input", {"prompt": prompt, "response": user_input})
                
                return user_input
                
            except KeyboardInterrupt:
                self.print_formatted("\nOperation cancelled by user.", style="warning")
                return ""
            except EOFError:
                return ""
    
    def show_menu(
        self,
        title: str,
        options: List[Tuple[str, str]],
        allow_cancel: bool = True
    ) -> Optional[str]:
        """Show interactive menu.
        
        Args:
            title: Menu title
            options: List of (key, description) tuples
            allow_cancel: Whether to allow cancellation
            
        Returns:
            Selected option key or None if cancelled
        """
        self.print_formatted(f"\n{title}", style="info")
        self.print_formatted("=" * len(title), style="info")
        
        # Show options
        for key, description in options:
            self.print_formatted(f"{key}) {description}")
        
        if allow_cancel:
            self.print_formatted("q) Cancel/Quit")
        
        # Get selection
        valid_keys = [key for key, _ in options]
        if allow_cancel:
            valid_keys.append("q")
        
        def validate_choice(choice: str) -> bool:
            return choice.lower() in [k.lower() for k in valid_keys]
        
        choice = self.get_user_input(
            "Select option",
            validate=validate_choice
        )
        
        if choice.lower() == "q" and allow_cancel:
            return None
        
        return choice
    
    def show_progress(
        self,
        operation_id: str,
        total_steps: int,
        description: str = "Processing"
    ) -> ProgressTracker:
        """Create and show progress tracker.
        
        Args:
            operation_id: Unique operation identifier
            total_steps: Total number of steps
            description: Operation description
            
        Returns:
            Progress tracker instance
        """
        tracker = ProgressTracker(total_steps, description)
        
        # Add console callback
        def console_callback(progress: ProgressInfo):
            if self.preferences.enable_animations:
                # Animated progress bar
                bar_width = 30
                filled = int(bar_width * progress.percentage / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                eta_str = ""
                if progress.estimated_time_remaining:
                    eta_str = f" ETA: {progress.estimated_time_remaining:.1f}s"
                
                # Clear line and show progress
                print(f"\r{progress.message}: [{bar}] {progress.percentage:.1f}%{eta_str}", end="", flush=True)
            else:
                # Simple text progress
                print(f"{progress.message}: {progress.current}/{progress.total} ({progress.percentage:.1f}%)")
        
        tracker.add_callback(console_callback)
        self.progress_trackers[operation_id] = tracker
        
        return tracker
    
    def hide_progress(self, operation_id: str) -> None:
        """Hide progress tracker.
        
        Args:
            operation_id: Operation identifier
        """
        if operation_id in self.progress_trackers:
            del self.progress_trackers[operation_id]
            if self.preferences.enable_animations:
                print()  # New line after progress bar
    
    def add_to_history(self, action: str, data: Dict[str, Any]) -> None:
        """Add entry to user history.
        
        Args:
            action: Action type
            data: Action data
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data
        }
        
        self.history.append(entry)
        
        # Trim history if too long
        if len(self.history) > self.preferences.max_history:
            self.history = self.history[-self.preferences.max_history:]
    
    def show_history(self, limit: int = 10) -> None:
        """Show user history.
        
        Args:
            limit: Maximum number of entries to show
        """
        if not self.history:
            self.print_formatted("No history available.", style="info")
            return
        
        self.print_formatted("Recent History:", style="info")
        self.print_formatted("-" * 40, style="info")
        
        recent_history = self.history[-limit:]
        for entry in recent_history:
            timestamp = entry["timestamp"][:19]  # Remove microseconds
            action = entry["action"]
            self.print_formatted(f"{timestamp} - {action}", indent=1)
    
    def register_shortcut(self, key: str, handler: Callable) -> None:
        """Register keyboard shortcut.
        
        Args:
            key: Shortcut key
            handler: Handler function
        """
        self.shortcuts[key] = handler
        
        # Add to custom shortcuts if not default
        if key not in ["help", "quit", "settings"]:
            self.preferences.custom_shortcuts[key] = handler.__name__
    
    def handle_shortcuts(self, input_text: str) -> bool:
        """Handle keyboard shortcuts.
        
        Args:
            input_text: User input text
            
        Returns:
            True if shortcut was handled
        """
        if input_text in self.shortcuts:
            try:
                self.shortcuts[input_text]()
                return True
            except Exception as e:
                self.print_formatted(f"Shortcut error: {e}", style="error")
                return True
        
        return False
    
    def show_settings(self) -> None:
        """Show user settings interface."""
        while True:
            settings_options = [
                ("1", f"Theme: {self.preferences.theme.value}"),
                ("2", f"Language: {self.preferences.language}"),
                ("3", f"Interaction Mode: {self.preferences.interaction_mode.value}"),
                ("4", f"Auto-save: {self.preferences.auto_save}"),
                ("5", f"Show Tooltips: {self.preferences.show_tooltips}"),
                ("6", f"Enable Animations: {self.preferences.enable_animations}"),
                ("7", f"Default Strategy: {self.preferences.default_strategy}"),
                ("8", f"Max History: {self.preferences.max_history}"),
                ("s", "Save settings"),
                ("r", "Reset to defaults")
            ]
            
            choice = self.show_menu("Settings", settings_options)
            
            if choice is None or choice.lower() == "q":
                break
            elif choice == "1":
                self._change_theme()
            elif choice == "2":
                self._change_language()
            elif choice == "3":
                self._change_interaction_mode()
            elif choice == "4":
                self.preferences.auto_save = not self.preferences.auto_save
            elif choice == "5":
                self.preferences.show_tooltips = not self.preferences.show_tooltips
            elif choice == "6":
                self.preferences.enable_animations = not self.preferences.enable_animations
            elif choice == "7":
                self._change_default_strategy()
            elif choice == "8":
                self._change_max_history()
            elif choice == "s":
                self.save_preferences()
                self.print_formatted("Settings saved!", style="success")
            elif choice == "r":
                self.preferences = UserPreferences()
                self.print_formatted("Settings reset to defaults!", style="success")
    
    def _change_theme(self) -> None:
        """Change theme setting."""
        theme_options = [
            ("1", "Light"),
            ("2", "Dark"), 
            ("3", "Auto")
        ]
        
        choice = self.show_menu("Select Theme", theme_options)
        theme_map = {"1": ThemeMode.LIGHT, "2": ThemeMode.DARK, "3": ThemeMode.AUTO}
        
        if choice in theme_map:
            self.preferences.theme = theme_map[choice]
            self.print_formatted(f"Theme changed to {self.preferences.theme.value}", style="success")
    
    def _change_language(self) -> None:
        """Change language setting."""
        lang = self.get_user_input("Enter language code (e.g., en, es, fr)", 
                                  default=self.preferences.language)
        if lang:
            self.preferences.language = lang
            self.print_formatted(f"Language changed to {lang}", style="success")
    
    def _change_interaction_mode(self) -> None:
        """Change interaction mode."""
        mode_options = [
            ("1", "Interactive"),
            ("2", "Batch"),
            ("3", "API"),
            ("4", "CLI")
        ]
        
        choice = self.show_menu("Select Interaction Mode", mode_options)
        mode_map = {
            "1": InteractionMode.INTERACTIVE,
            "2": InteractionMode.BATCH,
            "3": InteractionMode.API,
            "4": InteractionMode.CLI
        }
        
        if choice in mode_map:
            self.preferences.interaction_mode = mode_map[choice]
            self.print_formatted(f"Mode changed to {self.preferences.interaction_mode.value}", style="success")
    
    def _change_default_strategy(self) -> None:
        """Change default strategy."""
        strategies = ["basic", "enhanced_rag", "multi_modal", "semantic"]
        
        strategy_options = [(str(i+1), s) for i, s in enumerate(strategies)]
        choice = self.show_menu("Select Default Strategy", strategy_options)
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(strategies):
                self.preferences.default_strategy = strategies[idx]
                self.print_formatted(f"Default strategy changed to {strategies[idx]}", style="success")
        except (ValueError, IndexError):
            self.print_formatted("Invalid choice", style="error")
    
    def _change_max_history(self) -> None:
        """Change max history setting."""
        def validate_number(value: str) -> bool:
            try:
                num = int(value)
                return 10 <= num <= 1000
            except ValueError:
                return False
        
        history_str = self.get_user_input(
            "Enter max history (10-1000)",
            default=str(self.preferences.max_history),
            validate=validate_number
        )
        
        if history_str:
            self.preferences.max_history = int(history_str)
            self.print_formatted(f"Max history changed to {self.preferences.max_history}", style="success")
    
    def save_preferences(self, file_path: str = "user_preferences.json") -> None:
        """Save user preferences to file.
        
        Args:
            file_path: Preferences file path
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.preferences.to_dict(), f, indent=2)
            logger.info(f"Preferences saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")
            self.print_formatted(f"Failed to save preferences: {e}", style="error")
    
    def load_preferences(self, file_path: str = "user_preferences.json") -> None:
        """Load user preferences from file.
        
        Args:
            file_path: Preferences file path
        """
        try:
            if os.path.exists(file_path):
                with open(file_path) as f:
                    data = json.load(f)
                self.preferences = UserPreferences.from_dict(data)
                logger.info(f"Preferences loaded from {file_path}")
            else:
                logger.info("No preferences file found, using defaults")
        except Exception as e:
            logger.error(f"Failed to load preferences: {e}")
            self.print_formatted(f"Failed to load preferences: {e}", style="warning")


def create_enhanced_ui() -> UserInterface:
    """Create enhanced UI instance with default settings.
    
    Returns:
        UserInterface instance
    """
    ui = UserInterface()
    ui.load_preferences()
    
    # Register default shortcuts
    ui.register_shortcut("help", lambda: ui.print_formatted("Available commands: explain, analyze, settings, history, quit"))
    ui.register_shortcut("quit", lambda: sys.exit(0))
    ui.register_shortcut("settings", ui.show_settings)
    ui.register_shortcut("history", lambda: ui.show_history())
    
    return ui
