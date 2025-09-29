"""Real-time monitoring and alerting system for code explanation operations."""

import logging
import time
import threading
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MetricEvent:
    """Represents a metric event."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Defines an alert rule."""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    window_seconds: int = 60
    callback: Optional[Callable[[List[MetricEvent]], None]] = None
    enabled: bool = True


class RealTimeMetrics:
    """Real-time metrics collection and monitoring."""

    def __init__(self, max_events: int = 10000, alert_check_interval: int = 5):
        """Initialize real-time metrics.

        Args:
            max_events: Maximum events to keep in memory
            alert_check_interval: Seconds between alert checks
        """
        self.max_events = max_events
        self.alert_check_interval = alert_check_interval

        # Thread-safe data structures
        self._events = deque(maxlen=max_events)
        self._counters = defaultdict(float)
        self._gauges = defaultdict(float)
        self._histograms = defaultdict(list)
        self._alert_rules = {}
        self._subscribers = defaultdict(list)

        # Threading
        self._lock = threading.RLock()
        self._alert_thread = None
        self._running = False

        # Start alert monitoring
        self.start_monitoring()

    def record_event(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a metric event.

        Args:
            name: Event name
            value: Event value
            tags: Optional tags for filtering/grouping
            metadata: Optional additional metadata
        """
        event = MetricEvent(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metadata=metadata or {}
        )

        with self._lock:
            self._events.append(event)

            # Update aggregated metrics
            self._counters[name] += value
            self._gauges[name] = value
            self._histograms[name].append(value)

            # Keep histogram size manageable
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]

        # Notify subscribers
        self._notify_subscribers(name, event)

    def increment_counter(self, name: str, value: float = 1.0, **tags) -> None:
        """Increment a counter metric."""
        self.record_event(name, value, tags)

    def set_gauge(self, name: str, value: float, **tags) -> None:
        """Set a gauge metric."""
        self.record_event(name, value, tags)

    def time_operation(self, operation_name: str, **tags):
        """Context manager for timing operations."""
        class TimingContext:
            def __init__(self, metrics_instance, name, tags):
                self.metrics = metrics_instance
                self.name = name
                self.tags = tags
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.start_time:
                    duration = time.time() - self.start_time
                    self.metrics.record_event(
                        f"{self.name}_duration",
                        duration * 1000,  # Convert to milliseconds
                        self.tags,
                        {"operation": self.name, "success": exc_type is None}
                    )

        return TimingContext(self, operation_name, tags)

    def get_metrics_summary(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """Get metrics summary.

        Args:
            time_window: Time window in seconds (None for all time)

        Returns:
            Metrics summary
        """
        cutoff_time = time.time() - time_window if time_window else 0

        with self._lock:
            # Filter events by time window
            recent_events = [
                event for event in self._events
                if event.timestamp >= cutoff_time
            ]

            # Group events by name
            event_groups = defaultdict(list)
            for event in recent_events:
                event_groups[event.name].append(event)

            summary = {}
            for name, events in event_groups.items():
                values = [event.value for event in events]

                if values:
                    summary[name] = {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "median": statistics.median(values),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99),
                        "recent_events": len(events),
                        "rate_per_sec": len(events) / time_window if time_window else 0
                    }

                    if len(values) > 1:
                        summary[name]["std"] = statistics.stdev(values)

            return summary

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: Alert rule to add
        """
        with self._lock:
            self._alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_name: str) -> None:
        """Remove an alert rule.

        Args:
            rule_name: Name of rule to remove
        """
        with self._lock:
            if rule_name in self._alert_rules:
                del self._alert_rules[rule_name]
                logger.info(f"Removed alert rule: {rule_name}")

    def subscribe_to_metric(self, metric_name: str, callback: Callable[[MetricEvent], None]) -> None:
        """Subscribe to metric events.

        Args:
            metric_name: Name of metric to subscribe to
            callback: Callback function to call on events
        """
        with self._lock:
            self._subscribers[metric_name].append(callback)

    def get_current_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        alerts = []
        current_time = time.time()

        with self._lock:
            for rule_name, rule in self._alert_rules.items():
                if not rule.enabled:
                    continue

                # Get events in window
                window_start = current_time - rule.window_seconds
                window_events = [
                    event for event in self._events
                    if (event.name == rule.metric_name and
                        event.timestamp >= window_start)
                ]

                if not window_events:
                    continue

                # Check condition
                values = [event.value for event in window_events]
                current_value = statistics.mean(values)  # Use average for window

                triggered = self._check_condition(current_value, rule.condition, rule.threshold)

                if triggered:
                    alerts.append({
                        "rule_name": rule_name,
                        "metric_name": rule.metric_name,
                        "current_value": current_value,
                        "threshold": rule.threshold,
                        "condition": rule.condition,
                        "window_seconds": rule.window_seconds,
                        "event_count": len(window_events),
                        "timestamp": current_time
                    })

        return alerts

    def start_monitoring(self) -> None:
        """Start the monitoring thread."""
        if self._running:
            return

        self._running = True
        self._alert_thread = threading.Thread(target=self._alert_loop, daemon=True)
        self._alert_thread.start()
        logger.info("Started real-time monitoring")

    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self._running = False
        if self._alert_thread:
            self._alert_thread.join()
        logger.info("Stopped real-time monitoring")

    def export_metrics(self, filepath: str, format: str = "json") -> None:
        """Export metrics to file.

        Args:
            filepath: Output file path
            format: Export format ("json" or "csv")
        """
        summary = self.get_metrics_summary()

        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": summary,
                    "total_events": len(self._events)
                }, f, indent=2)
        elif format.lower() == "csv":
            import csv
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "count", "sum", "avg", "min", "max", "p95", "p99"])
                for name, stats in summary.items():
                    writer.writerow([
                        name, stats["count"], stats["sum"], stats["avg"],
                        stats["min"], stats["max"], stats["p95"], stats["p99"]
                    ])

        logger.info(f"Exported metrics to {filepath}")

    def _notify_subscribers(self, metric_name: str, event: MetricEvent) -> None:
        """Notify subscribers of new events."""
        callbacks = self._subscribers.get(metric_name, [])
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in metric subscriber callback: {e}")

    def _alert_loop(self) -> None:
        """Main alert monitoring loop."""
        while self._running:
            try:
                alerts = self.get_current_alerts()

                for alert in alerts:
                    rule = self._alert_rules.get(alert["rule_name"])
                    if rule and rule.callback:
                        try:
                            # Get relevant events for callback
                            window_start = time.time() - rule.window_seconds
                            relevant_events = [
                                event for event in self._events
                                if (event.name == rule.metric_name and
                                    event.timestamp >= window_start)
                            ]
                            rule.callback(relevant_events)
                        except Exception as e:
                            logger.error(f"Error in alert callback for {rule.name}: {e}")

                time.sleep(self.alert_check_interval)

            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                time.sleep(self.alert_check_interval)

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if alert condition is met."""
        if condition == "gt":
            return value > threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.001  # Float comparison
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = min(lower_index + 1, len(sorted_values) - 1)
            weight = index - lower_index
            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_monitoring()


class PerformanceDashboard:
    """Dashboard for monitoring code explanation performance."""

    def __init__(self, metrics: RealTimeMetrics):
        """Initialize dashboard with metrics instance."""
        self.metrics = metrics
        self._setup_default_alerts()

    def _setup_default_alerts(self) -> None:
        """Setup default performance alerts."""
        # High latency alert
        self.metrics.add_alert_rule(AlertRule(
            name="high_latency",
            metric_name="explanation_duration",
            condition="gt",
            threshold=5000,  # 5 seconds in milliseconds
            window_seconds=60,
            callback=self._high_latency_alert
        ))

        # High error rate alert
        self.metrics.add_alert_rule(AlertRule(
            name="high_error_rate",
            metric_name="explanation_error",
            condition="gt",
            threshold=5,  # More than 5 errors per minute
            window_seconds=60,
            callback=self._high_error_rate_alert
        ))

        # Memory usage alert
        self.metrics.add_alert_rule(AlertRule(
            name="high_memory_usage",
            metric_name="memory_usage_mb",
            condition="gt",
            threshold=1000,  # 1GB
            window_seconds=30,
            callback=self._high_memory_alert
        ))

    def _high_latency_alert(self, events: List[MetricEvent]) -> None:
        """Handle high latency alert."""
        avg_latency = statistics.mean([e.value for e in events])
        logger.warning(f"High latency detected: {avg_latency:.2f}ms average over {len(events)} requests")

    def _high_error_rate_alert(self, events: List[MetricEvent]) -> None:
        """Handle high error rate alert."""
        error_count = len(events)
        logger.error(f"High error rate detected: {error_count} errors in the last minute")

    def _high_memory_alert(self, events: List[MetricEvent]) -> None:
        """Handle high memory usage alert."""
        if events:
            current_memory = events[-1].value
            logger.warning(f"High memory usage detected: {current_memory:.2f}MB")

    def print_dashboard(self) -> None:
        """Print a text-based dashboard."""
        summary = self.metrics.get_metrics_summary(time_window=300)  # Last 5 minutes
        alerts = self.metrics.get_current_alerts()

        print("\n" + "="*60)
        print("📊 CODE EXPLAINER PERFORMANCE DASHBOARD")
        print("="*60)
        print(f"⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if summary:
            print("\n📈 METRICS (Last 5 minutes):")
            for name, stats in summary.items():
                print(f"  {name}:")
                print(f"    Count: {stats['count']}")
                print(f"    Average: {stats['avg']:.2f}")
                print(f"    P95: {stats['p95']:.2f}")
                print(f"    Rate: {stats['rate_per_sec']:.2f}/sec")

        if alerts:
            print("\n🚨 ACTIVE ALERTS:")
            for alert in alerts:
                print(f"  ⚠️  {alert['rule_name']}: {alert['current_value']:.2f} {alert['condition']} {alert['threshold']}")
        else:
            print("\n✅ No active alerts")

        print("="*60)


# Global metrics instance
_global_metrics = None

def get_metrics() -> RealTimeMetrics:
    """Get global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = RealTimeMetrics()
    return _global_metrics


def monitor_explanation_performance(func):
    """Decorator to monitor explanation performance."""
    def wrapper(*args, **kwargs):
        metrics = get_metrics()

        with metrics.time_operation("explanation"):
            try:
                result = func(*args, **kwargs)
                metrics.increment_counter("explanation_success")
                return result
            except Exception as e:
                metrics.increment_counter("explanation_error")
                metrics.record_event("explanation_error_detail", 1, metadata={"error": str(e)})
                raise

    return wrapper


def main():
    """Example usage of real-time metrics."""
    metrics = RealTimeMetrics()
    dashboard = PerformanceDashboard(metrics)

    # Simulate some metric events
    for i in range(50):
        metrics.record_event("explanation_duration", 100 + i * 10)
        metrics.record_event("memory_usage_mb", 500 + i * 5)
        time.sleep(0.1)

    # Print dashboard
    dashboard.print_dashboard()

    # Export metrics
    metrics.export_metrics("metrics_export.json")

    # Stop monitoring
    metrics.stop_monitoring()


if __name__ == "__main__":
    main()
