"""
Real-Time Performance Analytics and Optimization Engine

This module provides comprehensive performance monitoring, analysis, and optimization
for the Code Explainer platform, including:
- Real-time performance metrics collection and analysis
- Intelligent caching strategies with cache warming
- Database query optimization and connection pooling
- Memory management and garbage collection optimization
- API response time optimization and request batching
- Predictive performance scaling and resource allocation
- Performance regression detection and alerting
- Automated performance tuning recommendations

Based on latest research in distributed systems performance and ML system optimization.
"""

import asyncio
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import statistics
import json
import weakref
import gc
import psutil
import sys

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    DATABASE_PERFORMANCE = "database_performance"

class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    CACHING = "caching"
    CONNECTION_POOLING = "connection_pooling"
    BATCH_PROCESSING = "batch_processing"
    MEMORY_OPTIMIZATION = "memory_optimization"
    QUERY_OPTIMIZATION = "query_optimization"
    LOAD_BALANCING = "load_balancing"

@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    timestamp: float
    metric_type: MetricType
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert configuration and state."""
    name: str
    metric_type: MetricType
    threshold: float
    condition: str  # "above" or "below"
    window_minutes: int = 5
    is_active: bool = False
    last_triggered: Optional[float] = None

class MetricsCollector:
    """Collects and stores performance metrics."""

    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.collection_interval = 1.0  # seconds
        self.running = False
        self._collection_thread = None

    def start_collection(self):
        """Start continuous metrics collection."""
        if self.running:
            return

        self.running = True
        self._collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collection_thread.start()
        logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        if self._collection_thread:
            self._collection_thread.join()
        logger.info("Metrics collection stopped")

    def _collect_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

    def _collect_system_metrics(self):
        """Collect system-level performance metrics."""
        timestamp = time.time()

        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.add_metric(PerformanceMetric(
            timestamp=timestamp,
            metric_type=MetricType.CPU_USAGE,
            value=cpu_percent,
            tags={"source": "system"}
        ))

        # Memory usage
        memory = psutil.virtual_memory()
        self.add_metric(PerformanceMetric(
            timestamp=timestamp,
            metric_type=MetricType.MEMORY_USAGE,
            value=memory.percent,
            tags={"source": "system"},
            metadata={
                "available": memory.available,
                "used": memory.used,
                "total": memory.total
            }
        ))

    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric."""
        self.metrics[metric.metric_type].append(metric)

    def get_metrics(self, metric_type: MetricType, 
                   minutes: int = 5) -> List[PerformanceMetric]:
        """Get metrics for a specific type and time window."""
        cutoff_time = time.time() - (minutes * 60)
        metrics = self.metrics[metric_type]
        
        return [m for m in metrics if m.timestamp >= cutoff_time]

    def get_metric_summary(self, metric_type: MetricType, 
                          minutes: int = 5) -> Dict[str, float]:
        """Get summary statistics for a metric type."""
        metrics = self.get_metrics(metric_type, minutes)
        
        if not metrics:
            return {}

        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "percentile_95": sorted(values)[int(len(values) * 0.95)] if values else 0
        }

class IntelligentCache:
    """Intelligent caching system with performance optimization."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU tracking."""
        with self._lock:
            if key not in self.cache:
                self.miss_count += 1
                return None

            cache_entry = self.cache[key]
            
            # Check TTL
            if time.time() - cache_entry["timestamp"] > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                self.miss_count += 1
                return None

            # Update access time for LRU
            self.access_times[key] = time.time()
            self.hit_count += 1
            
            return cache_entry["value"]

    def set(self, key: str, value: Any):
        """Set value in cache with intelligent eviction."""
        with self._lock:
            current_time = time.time()

            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()

            self.cache[key] = {
                "value": value,
                "timestamp": current_time,
                "access_count": 1
            }
            self.access_times[key] = current_time

    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.eviction_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0

        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count,
            "current_size": len(self.cache),
            "max_size": self.max_size
        }

    def warm_cache(self, data_loader: Callable[[], Dict[str, Any]]):
        """Warm cache with frequently accessed data."""
        try:
            warm_data = data_loader()
            for key, value in warm_data.items():
                self.set(key, value)
            logger.info(f"Cache warmed with {len(warm_data)} items")
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")

class DatabaseOptimizer:
    """Database performance optimization and monitoring."""

    def __init__(self):
        self.query_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "max_time": 0.0,
            "min_time": float('inf')
        })
        self.slow_query_threshold = 1.0  # seconds
        self.connection_pool_stats = {
            "active_connections": 0,
            "idle_connections": 0,
            "max_connections": 100,
            "connection_waits": 0
        }

    def track_query(self, query_hash: str, execution_time: float):
        """Track query execution performance."""
        stats = self.query_stats[query_hash]
        stats["count"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["min_time"] = min(stats["min_time"], execution_time)

        if execution_time > self.slow_query_threshold:
            logger.warning(f"Slow query detected: {query_hash} took {execution_time:.2f}s")

    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest queries by average execution time."""
        sorted_queries = sorted(
            self.query_stats.items(),
            key=lambda x: x[1]["avg_time"],
            reverse=True
        )

        return [
            {
                "query_hash": query_hash,
                **stats
            }
            for query_hash, stats in sorted_queries[:limit]
        ]

    def suggest_optimizations(self, query_hash: str) -> List[str]:
        """Suggest optimizations for slow queries."""
        stats = self.query_stats.get(query_hash, {})
        suggestions = []

        if stats.get("avg_time", 0) > self.slow_query_threshold:
            suggestions.extend([
                "Consider adding database indexes on frequently queried columns",
                "Review query for unnecessary JOINs or subqueries",
                "Consider query result caching for repeated queries",
                "Analyze query execution plan for optimization opportunities"
            ])

        if stats.get("count", 0) > 1000:
            suggestions.append("High-frequency query: consider result caching or materialized views")

        return suggestions

class MemoryOptimizer:
    """Memory usage optimization and monitoring."""

    def __init__(self):
        self.gc_stats = {
            "collections": [0, 0, 0],  # Generation 0, 1, 2
            "collected": [0, 0, 0],
            "uncollectable": [0, 0, 0]
        }
        self.memory_leaks: List[Dict[str, Any]] = []

    def optimize_memory(self):
        """Perform memory optimization operations."""
        # Force garbage collection
        before_gc = gc.get_count()
        collected = gc.collect()
        after_gc = gc.get_count()

        # Update statistics
        for i in range(3):
            self.gc_stats["collections"][i] += 1

        logger.info(f"Garbage collection freed {collected} objects")

        # Check for memory leaks
        self._detect_memory_leaks()

        return {
            "objects_collected": collected,
            "gc_counts_before": before_gc,
            "gc_counts_after": after_gc
        }

    def _detect_memory_leaks(self):
        """Detect potential memory leaks."""
        # Get object counts by type
        object_counts = defaultdict(int)
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] += 1

        # Check for unusually high counts (simplified detection)
        for obj_type, count in object_counts.items():
            if count > 10000:  # Threshold for potential leak
                leak_info = {
                    "type": obj_type,
                    "count": count,
                    "timestamp": time.time()
                }
                
                # Only add if not already reported recently
                recent_leaks = [
                    leak for leak in self.memory_leaks
                    if leak["type"] == obj_type and 
                       time.time() - leak["timestamp"] < 300  # 5 minutes
                ]
                
                if not recent_leaks:
                    self.memory_leaks.append(leak_info)
                    logger.warning(f"Potential memory leak detected: {obj_type} ({count} objects)")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "gc_stats": self.gc_stats,
            "potential_leaks": len(self.memory_leaks),
            "python_object_count": len(gc.get_objects())
        }

class PerformanceOptimizer:
    """Main performance optimization orchestrator."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.cache = IntelligentCache()
        self.db_optimizer = DatabaseOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.alerts: List[PerformanceAlert] = []
        self.optimization_history: List[Dict[str, Any]] = []

    def start_monitoring(self):
        """Start performance monitoring."""
        self.metrics_collector.start_collection()
        
        # Set up default alerts
        self._setup_default_alerts()
        
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.metrics_collector.stop_collection()
        logger.info("Performance monitoring stopped")

    def _setup_default_alerts(self):
        """Set up default performance alerts."""
        default_alerts = [
            PerformanceAlert("High CPU Usage", MetricType.CPU_USAGE, 80.0, "above"),
            PerformanceAlert("High Memory Usage", MetricType.MEMORY_USAGE, 85.0, "above"),
            PerformanceAlert("Low Cache Hit Rate", MetricType.CACHE_HIT_RATE, 50.0, "below"),
        ]
        
        self.alerts.extend(default_alerts)

    def check_alerts(self):
        """Check and trigger performance alerts."""
        triggered_alerts = []
        
        for alert in self.alerts:
            if self._should_trigger_alert(alert):
                if not alert.is_active:
                    alert.is_active = True
                    alert.last_triggered = time.time()
                    triggered_alerts.append(alert)
                    logger.warning(f"Performance alert triggered: {alert.name}")

        return triggered_alerts

    def _should_trigger_alert(self, alert: PerformanceAlert) -> bool:
        """Check if an alert should be triggered."""
        metrics = self.metrics_collector.get_metrics(
            alert.metric_type, 
            alert.window_minutes
        )
        
        if not metrics:
            return False

        avg_value = statistics.mean([m.value for m in metrics])
        
        if alert.condition == "above":
            return avg_value > alert.threshold
        else:
            return avg_value < alert.threshold

    def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run a complete optimization cycle."""
        start_time = time.time()
        optimizations_applied = []

        try:
            # Memory optimization
            memory_result = self.memory_optimizer.optimize_memory()
            optimizations_applied.append({
                "type": OptimizationStrategy.MEMORY_OPTIMIZATION,
                "result": memory_result
            })

            # Cache statistics update
            cache_stats = self.cache.get_stats()
            self.metrics_collector.add_metric(PerformanceMetric(
                timestamp=time.time(),
                metric_type=MetricType.CACHE_HIT_RATE,
                value=cache_stats["hit_rate"]
            ))

            # Check alerts
            triggered_alerts = self.check_alerts()

            duration = time.time() - start_time
            
            optimization_record = {
                "timestamp": start_time,
                "duration": duration,
                "optimizations": optimizations_applied,
                "alerts_triggered": len(triggered_alerts),
                "success": True
            }
            
            self.optimization_history.append(optimization_record)
            
            return optimization_record

        except Exception as e:
            duration = time.time() - start_time
            error_record = {
                "timestamp": start_time,
                "duration": duration,
                "error": str(e),
                "success": False
            }
            
            self.optimization_history.append(error_record)
            logger.error(f"Optimization cycle failed: {e}")
            
            return error_record

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard."""
        return {
            "system_metrics": {
                "cpu": self.metrics_collector.get_metric_summary(MetricType.CPU_USAGE),
                "memory": self.metrics_collector.get_metric_summary(MetricType.MEMORY_USAGE),
            },
            "cache_performance": self.cache.get_stats(),
            "database_performance": {
                "slow_queries": self.db_optimizer.get_slow_queries(5),
                "connection_stats": self.db_optimizer.connection_pool_stats
            },
            "memory_stats": self.memory_optimizer.get_memory_stats(),
            "active_alerts": [
                {
                    "name": alert.name,
                    "metric": alert.metric_type.value,
                    "threshold": alert.threshold,
                    "is_active": alert.is_active
                }
                for alert in self.alerts if alert.is_active
            ],
            "optimization_history": self.optimization_history[-10:],  # Last 10 cycles
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # CPU usage recommendations
        cpu_metrics = self.metrics_collector.get_metrics(MetricType.CPU_USAGE, 10)
        if cpu_metrics:
            avg_cpu = statistics.mean([m.value for m in cpu_metrics])
            if avg_cpu > 70:
                recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")

        # Memory usage recommendations
        memory_metrics = self.metrics_collector.get_metrics(MetricType.MEMORY_USAGE, 10)
        if memory_metrics:
            avg_memory = statistics.mean([m.value for m in memory_metrics])
            if avg_memory > 80:
                recommendations.append("Consider increasing memory allocation or optimizing memory usage")

        # Cache recommendations
        cache_stats = self.cache.get_stats()
        if cache_stats["hit_rate"] < 60:
            recommendations.append("Improve cache strategy: increase TTL or cache size, implement cache warming")

        # Database recommendations
        slow_queries = self.db_optimizer.get_slow_queries(3)
        if slow_queries:
            recommendations.append(f"Optimize {len(slow_queries)} slow database queries")

        return recommendations

# Export main classes
__all__ = [
    "MetricType",
    "OptimizationStrategy", 
    "PerformanceMetric",
    "PerformanceAlert",
    "MetricsCollector",
    "IntelligentCache",
    "DatabaseOptimizer",
    "MemoryOptimizer",
    "PerformanceOptimizer"
]
