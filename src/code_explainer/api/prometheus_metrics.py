"""Prometheus metrics exporter for API."""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from typing import Dict, Any


# Define Prometheus metrics
request_count = Counter(
    'code_explainer_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'code_explainer_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

model_inference_duration = Histogram(
    'code_explainer_model_inference_seconds',
    'Model inference duration in seconds'
)

cache_hits = Counter(
    'code_explainer_cache_hits_total',
    'Total number of cache hits'
)

cache_misses = Counter(
    'code_explainer_cache_misses_total',
    'Total number of cache misses'
)

active_requests = Gauge(
    'code_explainer_active_requests',
    'Number of requests currently being processed'
)

model_loaded = Gauge(
    'code_explainer_model_loaded',
    'Whether the model is loaded (1=loaded, 0=not loaded)'
)


class PrometheusMetrics:
    """Prometheus metrics collector and exporter."""
    
    @staticmethod
    def record_request(method: str, endpoint: str, status_code: int, duration: float):
        """Record a completed request.
        
        Args:
            method: HTTP method
            endpoint: API endpoint path
            status_code: HTTP status code
            duration: Request duration in seconds
        """
        request_count.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    @staticmethod
    def record_model_inference(duration: float):
        """Record model inference time.
        
        Args:
            duration: Inference duration in seconds
        """
        model_inference_duration.observe(duration)
    
    @staticmethod
    def record_cache_hit():
        """Record a cache hit."""
        cache_hits.inc()
    
    @staticmethod
    def record_cache_miss():
        """Record a cache miss."""
        cache_misses.inc()
    
    @staticmethod
    def inc_active_requests():
        """Increment active requests counter."""
        active_requests.inc()
    
    @staticmethod
    def dec_active_requests():
        """Decrement active requests counter."""
        active_requests.dec()
    
    @staticmethod
    def set_model_loaded(loaded: bool):
        """Set model loaded status.
        
        Args:
            loaded: Whether model is loaded
        """
        model_loaded.set(1 if loaded else 0)
    
    @staticmethod
    def export_metrics() -> Response:
        """Export metrics in Prometheus format.
        
        Returns:
            Response with Prometheus metrics
        """
        metrics_data = generate_latest()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    
    @staticmethod
    def get_current_metrics() -> Dict[str, Any]:
        """Get current metrics as dictionary.
        
        Returns:
            Dictionary of current metric values
        """
        # Note: This is a simplified view. Full metrics are in Prometheus format.
        return {
            "requests_total": request_count._value.sum(),
            "cache_hits_total": cache_hits._value.get(),
            "cache_misses_total": cache_misses._value.get(),
            "active_requests": active_requests._value.get(),
            "model_loaded": model_loaded._value.get() == 1,
        }


# Global instance
prometheus_metrics = PrometheusMetrics()
