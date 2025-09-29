"""Plugin architecture and extensibility framework."""

import os
import sys
import importlib
import inspect
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type, Protocol
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import threading
from functools import wraps

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Plugin types."""
    EXPLANATION_STRATEGY = "explanation_strategy"
    OUTPUT_FORMATTER = "output_formatter"
    DATA_SOURCE = "data_source"
    SECURITY_VALIDATOR = "security_validator"
    CODE_ANALYZER = "code_analyzer"
    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"
    UI_COMPONENT = "ui_component"
    MIDDLEWARE = "middleware"


@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    python_version: str = ">=3.8"
    entry_point: str = ""
    config_schema: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "python_version": self.python_version,
            "entry_point": self.entry_point,
            "config_schema": self.config_schema,
            "enabled": self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            author=data["author"],
            plugin_type=PluginType(data["plugin_type"]),
            dependencies=data.get("dependencies", []),
            python_version=data.get("python_version", ">=3.8"),
            entry_point=data.get("entry_point", ""),
            config_schema=data.get("config_schema", {}),
            enabled=data.get("enabled", True)
        )


class PluginInterface(Protocol):
    """Base plugin interface."""

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        ...

    def get_name(self) -> str:
        """Get plugin name."""
        ...

    def get_version(self) -> str:
        """Get plugin version."""
        ...


class BasePlugin(ABC):
    """Base plugin class."""

    def __init__(self):
        """Initialize base plugin."""
        self.config: Dict[str, Any] = {}
        self.metadata: Optional[PluginMetadata] = None
        self._initialized = False

    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get plugin description."""
        pass

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration.

        Args:
            config: Plugin configuration
        """
        self.config = config
        self._initialized = True
        logger.info(f"Plugin {self.get_name()} initialized")

    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid
        """
        # Override in subclasses for specific validation
        return True


class ExplanationStrategyPlugin(BasePlugin):
    """Base class for explanation strategy plugins."""

    @abstractmethod
    def explain(self, code: str, context: Dict[str, Any]) -> str:
        """Explain code using this strategy.

        Args:
            code: Code to explain
            context: Additional context

        Returns:
            Explanation text
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages.

        Returns:
            List of language names
        """
        pass


class OutputFormatterPlugin(BasePlugin):
    """Base class for output formatter plugins."""

    @abstractmethod
    def format(self, content: str, format_type: str, options: Dict[str, Any]) -> str:
        """Format content.

        Args:
            content: Content to format
            format_type: Format type (html, markdown, plain, etc.)
            options: Formatting options

        Returns:
            Formatted content
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported format types.

        Returns:
            List of format names
        """
        pass


class SecurityValidatorPlugin(BasePlugin):
    """Base class for security validator plugins."""

    @abstractmethod
    def validate(self, code: str) -> Dict[str, Any]:
        """Validate code security.

        Args:
            code: Code to validate

        Returns:
            Validation result
        """
        pass


class PluginManager:
    """Plugin management system."""

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """Initialize plugin manager.

        Args:
            plugin_dirs: List of plugin directories
        """
        self.plugin_dirs = plugin_dirs or ["plugins", "~/.code_explainer/plugins"]
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover available plugins.

        Returns:
            List of discovered plugin metadata
        """
        discovered = []

        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir).expanduser()

            if not plugin_path.exists():
                continue

            logger.info(f"Discovering plugins in {plugin_path}")

            # Look for plugin.json files
            for plugin_json in plugin_path.rglob("plugin.json"):
                try:
                    with open(plugin_json) as f:
                        plugin_data = json.load(f)

                    metadata = PluginMetadata.from_dict(plugin_data)
                    discovered.append(metadata)

                    logger.info(f"Discovered plugin: {metadata.name} v{metadata.version}")

                except Exception as e:
                    logger.error(f"Failed to load plugin metadata from {plugin_json}: {e}")

        return discovered

    def load_plugin(self, metadata: PluginMetadata) -> bool:
        """Load a plugin.

        Args:
            metadata: Plugin metadata

        Returns:
            True if loaded successfully
        """
        if not metadata.enabled:
            logger.info(f"Plugin {metadata.name} is disabled, skipping")
            return False

        try:
            with self._lock:
                # Check dependencies
                for dep in metadata.dependencies:
                    if dep not in self.plugins:
                        logger.error(f"Plugin {metadata.name} requires {dep} which is not loaded")
                        return False

                # Import plugin module
                if metadata.entry_point:
                    module_path, class_name = metadata.entry_point.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    plugin_class = getattr(module, class_name)

                    # Create plugin instance
                    plugin = plugin_class()

                    # Validate plugin interface
                    if not isinstance(plugin, BasePlugin):
                        logger.error(f"Plugin {metadata.name} does not inherit from BasePlugin")
                        return False

                    # Initialize plugin
                    config = self._get_plugin_config(metadata.name)
                    plugin.metadata = metadata
                    plugin.initialize(config)

                    # Register plugin
                    self.plugins[metadata.name] = plugin
                    self.plugin_metadata[metadata.name] = metadata

                    logger.info(f"Successfully loaded plugin: {metadata.name}")
                    return True
                else:
                    logger.error(f"Plugin {metadata.name} has no entry point")
                    return False

        except Exception as e:
            logger.error(f"Failed to load plugin {metadata.name}: {e}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin.

        Args:
            plugin_name: Name of plugin to unload

        Returns:
            True if unloaded successfully
        """
        try:
            with self._lock:
                if plugin_name in self.plugins:
                    plugin = self.plugins[plugin_name]

                    # Call cleanup if available
                    if hasattr(plugin, 'cleanup') and callable(getattr(plugin, 'cleanup')):
                        try:
                            plugin.cleanup()  # type: ignore
                        except Exception as e:
                            logger.error(f"Plugin cleanup failed for {plugin_name}: {e}")

                    # Remove from registry
                    del self.plugins[plugin_name]
                    del self.plugin_metadata[plugin_name]

                    logger.info(f"Unloaded plugin: {plugin_name}")
                    return True
                else:
                    logger.warning(f"Plugin {plugin_name} not found")
                    return False

        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False

    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin.

        Args:
            plugin_name: Name of plugin to reload

        Returns:
            True if reloaded successfully
        """
        if plugin_name not in self.plugin_metadata:
            logger.error(f"Plugin {plugin_name} not found")
            return False

        metadata = self.plugin_metadata[plugin_name]

        # Unload and reload
        self.unload_plugin(plugin_name)
        return self.load_plugin(metadata)

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin instance or None
        """
        return self.plugins.get(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get plugins by type.

        Args:
            plugin_type: Plugin type

        Returns:
            List of plugins of the specified type
        """
        result = []
        for plugin_name, plugin in self.plugins.items():
            metadata = self.plugin_metadata.get(plugin_name)
            if metadata and metadata.plugin_type == plugin_type:
                result.append(plugin)
        return result

    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all plugins with their status.

        Returns:
            Dictionary of plugin information
        """
        result = {}

        for plugin_name, metadata in self.plugin_metadata.items():
            is_loaded = plugin_name in self.plugins
            result[plugin_name] = {
                "metadata": metadata.to_dict(),
                "loaded": is_loaded,
                "initialized": self.plugins[plugin_name].is_initialized() if is_loaded else False
            }

        return result

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a hook callback.

        Args:
            hook_name: Hook name
            callback: Callback function
        """
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)

    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Call hook callbacks.

        Args:
            hook_name: Hook name
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            List of callback results
        """
        results = []

        if hook_name in self.hooks:
            for callback in self.hooks[hook_name]:
                try:
                    result = callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Hook {hook_name} callback failed: {e}")

        return results

    def _get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Get plugin configuration.

        Args:
            plugin_name: Plugin name

        Returns:
            Plugin configuration
        """
        config_file = Path(f"config/plugins/{plugin_name}.json")

        if config_file.exists():
            try:
                with open(config_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config for plugin {plugin_name}: {e}")

        return {}

    def auto_load_plugins(self) -> None:
        """Automatically discover and load all available plugins."""
        discovered = self.discover_plugins()

        # Sort by dependencies (simple topological sort)
        loaded = set()
        remaining = discovered.copy()

        while remaining:
            progress = False

            for metadata in remaining.copy():
                # Check if all dependencies are loaded
                deps_satisfied = all(dep in loaded for dep in metadata.dependencies)

                if deps_satisfied:
                    if self.load_plugin(metadata):
                        loaded.add(metadata.name)
                    remaining.remove(metadata)
                    progress = True

            if not progress:
                # Circular dependency or missing dependency
                for metadata in remaining:
                    logger.error(f"Cannot load plugin {metadata.name} - dependency issues")
                break


# Decorator for plugin hooks
def plugin_hook(hook_name: str):
    """Decorator to register a function as a plugin hook.

    Args:
        hook_name: Hook name
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get plugin manager from context (simplified)
            plugin_manager = getattr(wrapper, '_plugin_manager', None)
            if plugin_manager:
                # Call pre-hooks
                plugin_manager.call_hook(f"before_{hook_name}", *args, **kwargs)

                # Call original function
                result = func(*args, **kwargs)

                # Call post-hooks
                plugin_manager.call_hook(f"after_{hook_name}", result, *args, **kwargs)

                return result
            else:
                return func(*args, **kwargs)

        # Store hook name as attribute (type ignore for dynamic attribute)
        setattr(wrapper, 'hook_name', hook_name)
        return wrapper

    return decorator


# Example plugin implementations
class MarkdownFormatterPlugin(OutputFormatterPlugin):
    """Example markdown formatter plugin."""

    def get_name(self) -> str:
        return "markdown_formatter"

    def get_version(self) -> str:
        return "1.0.0"

    def get_description(self) -> str:
        return "Formats output as Markdown"

    def format(self, content: str, format_type: str, options: Dict[str, Any]) -> str:
        if format_type == "markdown":
            # Add markdown formatting
            lines = content.split('\n')
            formatted_lines = []

            for line in lines:
                if line.startswith('# '):
                    formatted_lines.append(f"## {line[2:]}")  # Convert to h2
                elif line.strip() and not line.startswith(' '):
                    formatted_lines.append(f"**{line}**")  # Bold headers
                else:
                    formatted_lines.append(line)

            return '\n'.join(formatted_lines)

        return content

    def get_supported_formats(self) -> List[str]:
        return ["markdown", "md"]


class BasicSecurityPlugin(SecurityValidatorPlugin):
    """Example basic security validator plugin."""

    def get_name(self) -> str:
        return "basic_security"

    def get_version(self) -> str:
        return "1.0.0"

    def get_description(self) -> str:
        return "Basic security validation for common patterns"

    def validate(self, code: str) -> Dict[str, Any]:
        issues = []

        # Check for dangerous patterns
        dangerous_patterns = [
            ("eval(", "Use of eval() can be dangerous"),
            ("exec(", "Use of exec() can be dangerous"),
            ("__import__", "Dynamic imports can be risky"),
            ("subprocess", "Subprocess usage detected"),
            ("os.system", "System command execution")
        ]

        for pattern, message in dangerous_patterns:
            if pattern in code:
                issues.append({
                    "type": "security_warning",
                    "pattern": pattern,
                    "message": message
                })

        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "risk_level": "high" if len(issues) > 2 else "medium" if issues else "low"
        }


def create_plugin_manager() -> PluginManager:
    """Create and configure plugin manager.

    Returns:
        Configured plugin manager
    """
    manager = PluginManager()

    # Auto-discover and load plugins
    manager.auto_load_plugins()

    return manager
