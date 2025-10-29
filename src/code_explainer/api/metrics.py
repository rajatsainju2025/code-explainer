"""Metrics tracking for API endpoints."""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime, timedelta


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    
    request_id: str
    endpoint: str
    start_time: float
    end_time: float = 0.0
    status_code: int = 200
    error: str = ""
    
    @property
    def duration(self) -> float:
        """Get request duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class MetricsCollector:
    """Thread-safe metrics collector for API requests."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of requests to keep in history
        """
        self._lock = threading.RLock()
        self._requests: List[RequestMetrics] = []
        self._max_history = max_history
        
        # Aggregated metrics
        self._total_requests = 0
        self._endpoint_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._response_times: Dict[str, List[float]] = defaultdict(list)
        
        # Model-specific metrics
        self._model_inference_times: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0
    
    def start_request(self, request_id: str, endpoint: str) -> RequestMetrics:
        """Start tracking a new request.
        
        Args:
            request_id: Unique request identifier
            endpoint: API endpoint being accessed
            
        Returns:
            RequestMetrics object for this request
        """
        metrics = RequestMetrics(
            request_id=request_id,
            endpoint=endpoint,
            start_time=time.time()
        )
        
        with self._lock:
            self._total_requests += 1
            self._endpoint_counts[endpoint] += 1
        
        return metrics
    
    def end_request(
        self,
        metrics: RequestMetrics,
        status_code: int = 200,
        error: str = ""
    ) -> None:
        """End tracking for a request.
        
        Args:
            metrics: RequestMetrics object to complete
            status_code: HTTP status code
            error: Error message if request failed
        """
        metrics.end_time = time.time()
        metrics.status_code = status_code
        metrics.error = error
        
        with self._lock:
            # Add to history
            self._requests.append(metrics)
            
            # Trim history if needed
            if len(self._requests) > self._max_history:
                self._requests = self._requests[-self._max_history:]
            
            # Update aggregated metrics
            self._response_times[metrics.endpoint].append(metrics.duration)
            
            # Keep only recent response times
            if len(self._response_times[metrics.endpoint]) > 1000:
                self._response_times[metrics.endpoint] = \
                    self._response_times[metrics.endpoint][-1000:]
            
            # Track errors
            if status_code >= 400:
                self._error_counts[f"{status_code}"] += 1
    
    def record_model_inference(self, duration: float) -> None:
        """Record model inference time.
        
        Args:
            duration: Inference duration in seconds
        """
        with self._lock:
            self._model_inference_times.append(duration)
            
            # Keep only recent inference times
            if len(self._model_inference_times) > 1000:
                self._model_inference_times = self._model_inference_times[-1000:]
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._cache_misses += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics.
        
        Returns:
            Dictionary of metrics
        """
        with self._lock:
            # Calculate average response time
            all_times = []
            for times in self._response_times.values():
                all_times.extend(times)
            
            avg_response_time = sum(all_times) / len(all_times) if all_times else 0.0
            
            # Calculate cache hit rate
            total_cache_requests = self._cache_hits + self._cache_misses
            cache_hit_rate = (
                self._cache_hits / total_cache_requests
                if total_cache_requests > 0
                else 0.0
            )
            
            # Calculate average model inference time
            avg_inference = (
                sum(self._model_inference_times) / len(self._model_inference_times)
                if self._model_inference_times
                else 0.0
            )
            
            # Get recent requests (last hour)
            cutoff_time = time.time() - 3600
            recent_requests = [
                r for r in self._requests
                if r.start_time >= cutoff_time
            ]
            
            return {
                "total_requests": self._total_requests,
                "recent_requests_1h": len(recent_requests),
                "average_response_time": round(avg_response_time, 4),
                "cache_hit_rate": round(cache_hit_rate, 4),
                "model_inference_time": round(avg_inference, 4),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "endpoint_counts": dict(self._endpoint_counts),
                "error_counts": dict(self._error_counts),
                "p50_response_time": self._calculate_percentile(all_times, 0.50),
                "p95_response_time": self._calculate_percentile(all_times, 0.95),
                "p99_response_time": self._calculate_percentile(all_times, 0.99),
            }
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile from values.
        
        Args:
            values: List of values
            percentile: Percentile to calculate (0.0-1.0)
            
        Returns:
            Percentile value
        """
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        index = min(index, len(sorted_values) - 1)
        return round(sorted_values[index], 4)
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._requests.clear()
            self._total_requests = 0
            self._endpoint_counts.clear()
            self._error_counts.clear()
            self._response_times.clear()
            self._model_inference_times.clear()
            self._cache_hits = 0
            self._cache_misses = 0


# Global metrics collector instance
_metrics_collector: MetricsCollector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    return _metrics_collector
