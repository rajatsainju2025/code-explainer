"""Global model instance singleton for efficient model state management.

Prevents redundant model initialization and provides centralized
model lifecycle management across the application.
"""

import threading
from typing import Optional, Dict, Any
from weakref import WeakValueDictionary
import logging

logger = logging.getLogger(__name__)


class ModelInstanceManager:
    """Manages global model instances with lazy initialization and cleanup."""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = WeakValueDictionary()
        self._global_lock = threading.Lock()
    
    def get_or_create(self, model_key: str, factory_func, *args, **kwargs) -> Any:
        """Get or create a model instance with thread-safe lazy loading.
        
        Args:
            model_key: Unique identifier for the model
            factory_func: Function to create the model if not cached
            *args, **kwargs: Arguments to pass to factory_func
            
        Returns:
            Model instance (cached or newly created)
        """
        # Fast path: check if already loaded
        if model_key in self._models:
            logger.debug(f"Using cached model: {model_key}")
            return self._models[model_key]
        
        # Get or create lock for this model
        with self._global_lock:
            if model_key not in self._locks:
                self._locks[model_key] = threading.Lock()
            lock = self._locks[model_key]
        
        # Thread-safe model creation
        with lock:
            # Double-check pattern to avoid race conditions
            if model_key in self._models:
                return self._models[model_key]
            
            logger.info(f"Initializing model: {model_key}")
            model = factory_func(*args, **kwargs)
            self._models[model_key] = model
            return model
    
    def release(self, model_key: str) -> bool:
        """Release a model instance from cache.
        
        Args:
            model_key: Unique identifier for the model
            
        Returns:
            True if model was released, False if not found
        """
        if model_key in self._models:
            with self._global_lock:
                if model_key in self._models:
                    del self._models[model_key]
                    logger.info(f"Released model: {model_key}")
                    return True
        return False
    
    def clear_all(self) -> int:
        """Clear all cached models.
        
        Returns:
            Number of models cleared
        """
        with self._global_lock:
            count = len(self._models)
            self._models.clear()
            logger.info(f"Cleared {count} cached models")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cached models.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._global_lock:
            return {
                "total_cached_models": len(self._models),
                "model_keys": list(self._models.keys())
            }


# Global singleton instance
_model_manager = ModelInstanceManager()


def get_model_manager() -> ModelInstanceManager:
    """Get the global model manager singleton.
    
    Returns:
        Global ModelInstanceManager instance
    """
    return _model_manager
