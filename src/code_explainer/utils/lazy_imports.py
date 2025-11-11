"""Lazy import utilities for reducing startup time and memory usage."""

import sys
from types import ModuleType
from typing import Any, Optional


class LazyModule(ModuleType):
    """A module that lazy-loads its attributes on first access."""
    
    def __init__(self, name: str, loader_func):
        """Initialize lazy module.
        
        Args:
            name: Module name
            loader_func: Callable that returns the actual module
        """
        super().__init__(name)
        self._loader_func = loader_func
        self._loaded = False
        self._real_module: Optional[ModuleType] = None
    
    def __getattr__(self, name: str) -> Any:
        """Load module on first attribute access."""
        if not self._loaded:
            self._real_module = self._loader_func()
            self._loaded = True
        
        if self._real_module is None:
            raise ImportError(f"Failed to load module for {self.__name__}")
        
        return getattr(self._real_module, name)
    
    def __dir__(self):
        """List attributes of the real module."""
        if not self._loaded:
            self._real_module = self._loader_func()
            self._loaded = True
        
        return dir(self._real_module) if self._real_module else []


def lazy_import(module_name: str, real_module_name: Optional[str] = None) -> ModuleType:
    """Create a lazy import for a module.
    
    Usage:
        torch = lazy_import("torch")
        x = torch.tensor([1, 2, 3])  # Actually imports torch here
    
    Args:
        module_name: Name to register in sys.modules
        real_module_name: Actual module to import (defaults to module_name)
    
    Returns:
        A lazy loading module proxy
    """
    if real_module_name is None:
        real_module_name = module_name
    
    def loader():
        """Load the actual module."""
        __import__(real_module_name)
        return sys.modules[real_module_name]
    
    lazy = LazyModule(module_name, loader)
    sys.modules[module_name] = lazy
    return lazy


class lazy_property:
    """Decorator for lazy evaluation of properties."""
    
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.__doc__ = func.__doc__
    
    def __get__(self, obj, objtype=None):
        """Compute and cache property value on first access."""
        if obj is None:
            return self
        
        # Compute value and cache it
        value = self.func(obj)
        # Store computed value in instance __dict__
        setattr(obj, self.name, value)
        return value
