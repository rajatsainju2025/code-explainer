"""Optimized error handling with pre-built response templates.

This module reduces overhead from error handling by pre-building
common error responses and reducing traceback collection overhead.
"""

import sys
import traceback
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
import threading


@dataclass
class ErrorTemplate:
    """Pre-built error response template."""
    status_code: int
    error_type: str
    message: str
    
    def build_response(self, request_id: Optional[str] = None,
                      detail: Optional[str] = None) -> Dict[str, Any]:
        """Build error response from template.
        
        Args:
            request_id: Request ID for tracing
            detail: Additional detail message
            
        Returns:
            Error response dict
        """
        response = {
            'error': self.error_type,
            'message': self.message,
            'status_code': self.status_code,
        }
        
        if detail:
            response['detail'] = detail
        
        if request_id:
            response['request_id'] = request_id
        
        return response


class ErrorTemplateRegistry:
    """Registry of pre-built error templates."""
    
    __slots__ = ('_templates', '_lock')
    
    def __init__(self):
        """Initialize error template registry with common errors."""
        self._templates: Dict[str, ErrorTemplate] = {
            'validation_error': ErrorTemplate(
                status_code=400,
                error_type='ValidationError',
                message='Input validation failed'
            ),
            'not_found': ErrorTemplate(
                status_code=404,
                error_type='NotFoundError',
                message='Resource not found'
            ),
            'unauthorized': ErrorTemplate(
                status_code=401,
                error_type='UnauthorizedError',
                message='Authentication required'
            ),
            'forbidden': ErrorTemplate(
                status_code=403,
                error_type='ForbiddenError',
                message='Access denied'
            ),
            'timeout': ErrorTemplate(
                status_code=408,
                error_type='TimeoutError',
                message='Request timeout'
            ),
            'conflict': ErrorTemplate(
                status_code=409,
                error_type='ConflictError',
                message='Resource conflict'
            ),
            'internal_error': ErrorTemplate(
                status_code=500,
                error_type='InternalServerError',
                message='Internal server error'
            ),
            'service_unavailable': ErrorTemplate(
                status_code=503,
                error_type='ServiceUnavailableError',
                message='Service temporarily unavailable'
            ),
        }
        self._lock = threading.RLock()
    
    def get_template(self, key: str) -> Optional[ErrorTemplate]:
        """Get error template by key."""
        with self._lock:
            return self._templates.get(key)
    
    def register_template(self, key: str, template: ErrorTemplate) -> None:
        """Register a new error template."""
        with self._lock:
            self._templates[key] = template
    
    def build_error_response(self, key: str, request_id: Optional[str] = None,
                            detail: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Build error response from template.
        
        Args:
            key: Template key
            request_id: Request ID
            detail: Detail message
            
        Returns:
            Error response dict or None if template not found
        """
        template = self.get_template(key)
        if template is None:
            return None
        
        return template.build_response(request_id, detail)


class FastExceptionHandler:
    """Optimized exception handler with reduced overhead."""
    
    __slots__ = ('_template_registry', '_max_traceback_frames')
    
    def __init__(self, max_traceback_frames: int = 3):
        """Initialize exception handler.
        
        Args:
            max_traceback_frames: Maximum frames to include in traceback
        """
        self._template_registry = ErrorTemplateRegistry()
        self._max_traceback_frames = max_traceback_frames
    
    def handle_exception(self, exc: Exception, request_id: Optional[str] = None,
                        include_traceback: bool = False) -> Dict[str, Any]:
        """Handle exception with minimal overhead.
        
        Args:
            exc: Exception to handle
            request_id: Request ID for tracing
            include_traceback: Whether to include traceback (expensive)
            
        Returns:
            Error response dict
        """
        exc_type = type(exc).__name__
        exc_message = str(exc)
        
        response = {
            'error': exc_type,
            'message': exc_message,
            'status_code': 500,
        }
        
        if request_id:
            response['request_id'] = request_id
        
        # Include abbreviated traceback if requested (expensive operation)
        if include_traceback:
            tb_lines = traceback.format_exc().split('\n')
            # Limit to last N lines to reduce overhead
            response['traceback'] = '\n'.join(tb_lines[-self._max_traceback_frames:])
        
        return response
    
    def fast_error_response(self, error_key: str, request_id: Optional[str] = None,
                           detail: Optional[str] = None) -> Dict[str, Any]:
        """Get error response from template (very fast).
        
        Args:
            error_key: Error template key
            request_id: Request ID
            detail: Detail message
            
        Returns:
            Pre-built error response
        """
        response = self._template_registry.build_error_response(error_key, request_id, detail)
        return response or {'error': 'UnknownError', 'status_code': 500}


class ContextualErrorHandler:
    """Error handler with request context awareness."""
    
    __slots__ = ('_fast_handler', '_context_stack')
    
    def __init__(self):
        """Initialize contextual error handler."""
        self._fast_handler = FastExceptionHandler()
        self._context_stack: List[Dict[str, Any]] = []
    
    def push_context(self, context_name: str, **kwargs) -> None:
        """Push error context for better error reporting.
        
        Args:
            context_name: Name of the context
            **kwargs: Context data
        """
        context = {'name': context_name, 'data': kwargs}
        self._context_stack.append(context)
    
    def pop_context(self) -> Optional[Dict[str, Any]]:
        """Pop error context."""
        if self._context_stack:
            return self._context_stack.pop()
        return None
    
    def get_context_info(self) -> Dict[str, Any]:
        """Get current context information for error reporting."""
        if not self._context_stack:
            return {}
        
        current_context = self._context_stack[-1]
        return {
            'context': current_context['name'],
            'context_data': current_context['data']
        }
    
    def handle_with_context(self, exc: Exception, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle exception with context information.
        
        Args:
            exc: Exception to handle
            request_id: Request ID
            
        Returns:
            Error response with context
        """
        response = self._fast_handler.handle_exception(exc, request_id)
        context_info = self.get_context_info()
        
        if context_info:
            response['context'] = context_info
        
        return response


class ErrorMetrics:
    """Track error metrics for monitoring."""
    
    __slots__ = ('_error_counts', '_lock', '_last_errors')
    
    def __init__(self, max_last_errors: int = 100):
        """Initialize error metrics.
        
        Args:
            max_last_errors: Maximum recent errors to keep
        """
        self._error_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._last_errors: List[Tuple[str, float]] = []  # (error_type, timestamp)
        self._max_last_errors = max_last_errors
    
    def record_error(self, error_type: str) -> None:
        """Record an error occurrence.
        
        Args:
            error_type: Type of error
        """
        import time
        
        with self._lock:
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
            
            # Keep last N errors
            self._last_errors.append((error_type, time.time()))
            if len(self._last_errors) > self._max_last_errors:
                self._last_errors.pop(0)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            total_errors = sum(self._error_counts.values())
            
            return {
                'total_errors': total_errors,
                'error_counts': self._error_counts.copy(),
                'recent_errors': len(self._last_errors),
            }
    
    def get_most_common_errors(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get most common errors.
        
        Args:
            limit: Number of top errors to return
            
        Returns:
            List of (error_type, count) tuples
        """
        with self._lock:
            sorted_errors = sorted(
                self._error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_errors[:limit]


# Global singleton instances
_fast_exception_handler: Optional[FastExceptionHandler] = None
_error_metrics: Optional[ErrorMetrics] = None

_handler_init_lock = threading.RLock()


def get_fast_exception_handler() -> FastExceptionHandler:
    """Get singleton fast exception handler."""
    global _fast_exception_handler
    
    if _fast_exception_handler is None:
        with _handler_init_lock:
            if _fast_exception_handler is None:
                _fast_exception_handler = FastExceptionHandler()
    
    return _fast_exception_handler


def get_error_metrics() -> ErrorMetrics:
    """Get singleton error metrics."""
    global _error_metrics
    
    if _error_metrics is None:
        with _handler_init_lock:
            if _error_metrics is None:
                _error_metrics = ErrorMetrics()
    
    return _error_metrics
