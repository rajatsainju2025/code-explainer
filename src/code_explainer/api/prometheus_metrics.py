"""Prometheus metrics exporter for API."""

from prometheus_client import (
    generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, 
    Gauge, REGISTRY
)
from fastapi import Response
import time
from functools import wraps


# Request metrics
request_count = Counter(
    'code_explainer_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status_code']
)

request_duration = Histogram(
    'code_explainer_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint', 'method'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# Model inference metrics
inference_duration = Histogram(
    'code_explainer_inference_duration_seconds',
    'Model inference duration',
    ['model_name'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

inference_errors = Counter(
    'code_explainer_inference_errors_total',
    'Total inference errors',
    ['model_name', 'error_type']
)

# Cache metrics
cache_hits = Counter(
    'code_explainer_cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'code_explainer_cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

# Database metrics
db_queries = Histogram(
    'code_explainer_db_query_duration_seconds',
    'Database query duration',
    ['operation'],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5)
)

db_errors = Counter(
    'code_explainer_db_errors_total',
    'Total database errors',
    ['operation', 'error_type']
)

# Application metrics
active_requests = Gauge(
    'code_explainer_active_requests',
    'Currently active requests'
)


class PrometheusMetrics:
    """Prometheus metrics exporter."""

    @staticmethod
    def export_metrics() -> Response:
        """Export metrics in Prometheus format.

        Returns:
            Response with Prometheus metrics
        """
        metrics_data = generate_latest(REGISTRY)
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
# Global instance
prometheus_metrics = PrometheusMetrics()


def track_request_metrics(endpoint: str, method: str):
    """Decorator to track HTTP request metrics."""
    def decorator(f):
        @wraps(f)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            active_requests.inc()
            status = 200
            try:
                result = await f(*args, **kwargs)
                if hasattr(result, 'status_code'):
                    status = result.status_code
                return result
            except Exception as e:
                status = 500
                request_count.labels(endpoint, method, status).inc()
                raise
            finally:
                duration = time.time() - start_time
                request_duration.labels(endpoint, method).observe(duration)
                request_count.labels(endpoint, method, status).inc()
                active_requests.dec()
        
        @wraps(f)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            active_requests.inc()
            status = 200
            try:
                result = f(*args, **kwargs)
                if hasattr(result, 'status_code'):
                    status = result.status_code
                return result
            except Exception as e:
                status = 500
                request_count.labels(endpoint, method, status).inc()
                raise
            finally:
                duration = time.time() - start_time
                request_duration.labels(endpoint, method).observe(duration)
                request_count.labels(endpoint, method, status).inc()
                active_requests.dec()
        
        import inspect
        if inspect.iscoroutinefunction(f):
            return async_wrapper
        return sync_wrapper
    
    return decorator
