"""
Monitoring and Analytics Module for Code Intelligence Platform

This module provides comprehensive monitoring, analytics, and observability
capabilities to ensure system reliability, performance optimization, and
data-driven insights for the code intelligence platform.

Features:
- Real-time metrics collection and visualization
- Distributed tracing and performance monitoring
- Log aggregation and analysis
- Alerting and incident management
- Analytics dashboard and reporting
- Performance profiling and optimization
- User behavior analytics
- System health monitoring
- Custom metrics and KPIs
"""

import time
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import statistics
import asyncio
from collections import defaultdict, deque
import psutil
import platform
import uuid

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Represents a monitoring metric."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    description: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class Alert:
    """Represents an alert."""
    name: str
    message: str
    severity: AlertSeverity
    condition: str
    value: Union[int, float]
    threshold: Union[int, float]
    timestamp: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class LogEntry:
    """Represents a log entry."""
    level: LogLevel
    message: str
    timestamp: Optional[datetime] = None
    source: str = ""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.trace_id is None:
            self.trace_id = str(uuid.uuid4())


class MetricsCollector:
    """Collects and manages metrics."""

    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._lock = threading.Lock()
        self.max_history = 1000

    def record_metric(self, metric: Metric) -> None:
        """Record a metric."""
        with self._lock:
            if len(self.metrics[metric.name]) >= self.max_history:
                self.metrics[metric.name].pop(0)
            self.metrics[metric.name].append(metric)

    def get_metric(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[Metric]:
        """Get the latest metric by name and optional tags."""
        with self._lock:
            if name not in self.metrics:
                return None

            metrics = self.metrics[name]
            if not metrics:
                return None

            if tags:
                # Filter by tags
                filtered = [m for m in metrics if all(m.tags.get(k) == v for k, v in tags.items())]
                return filtered[-1] if filtered else None

            return metrics[-1]

    def get_metrics_history(self, name: str, duration: timedelta = timedelta(hours=1)) -> List[Metric]:
        """Get metrics history for the specified duration."""
        with self._lock:
            if name not in self.metrics:
                return []

            cutoff = datetime.utcnow() - duration
            return [m for m in self.metrics[name] if m.timestamp and m.timestamp >= cutoff]

    def get_metric_stats(self, name: str, duration: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get statistical summary of metrics."""
        metrics = self.get_metrics_history(name, duration)
        if not metrics:
            return {}

        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "latest": values[-1],
            "oldest": values[0]
        }


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Callable[[], bool]] = {}
        self.notification_handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

    def add_alert_rule(self, name: str, condition: Callable[[], bool], severity: AlertSeverity,
                      message: str, threshold: Union[int, float]) -> None:
        """Add an alert rule."""
        def rule():
            if condition():
                alert = Alert(
                    name=name,
                    message=message,
                    severity=severity,
                    condition=str(condition),
                    value=0,  # Will be set when triggered
                    threshold=threshold
                )
                self.trigger_alert(alert)
                return True
            return False

        with self._lock:
            self.alert_rules[name] = rule

    def trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert and notify handlers."""
        with self._lock:
            self.alerts.append(alert)

            # Notify all handlers
            for handler in self.notification_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert notification handler failed: {e}")

    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add a notification handler."""
        with self._lock:
            self.notification_handlers.append(handler)

    def resolve_alert(self, alert_name: str) -> None:
        """Resolve an alert."""
        with self._lock:
            for alert in self.alerts:
                if alert.name == alert_name and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    break

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self.alerts if not alert.resolved]

    def get_alert_history(self, duration: timedelta = timedelta(days=7)) -> List[Alert]:
        """Get alert history for the specified duration."""
        cutoff = datetime.utcnow() - duration
        return [alert for alert in self.alerts if alert.timestamp and alert.timestamp >= cutoff]


class LogAggregator:
    """Aggregates and manages logs."""

    def __init__(self):
        self.logs: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        self.filters: List[Callable[[LogEntry], bool]] = []

    def log(self, entry: LogEntry) -> None:
        """Add a log entry."""
        with self._lock:
            # Apply filters
            if all(f(entry) for f in self.filters):
                self.logs.append(entry)

    def add_filter(self, filter_func: Callable[[LogEntry], bool]) -> None:
        """Add a log filter."""
        with self._lock:
            self.filters.append(filter_func)

    def get_logs(self, level: Optional[LogLevel] = None,
                source: Optional[str] = None,
                duration: timedelta = timedelta(hours=1)) -> List[LogEntry]:
        """Get filtered logs."""
        with self._lock:
            cutoff = datetime.utcnow() - duration
            logs = [log for log in self.logs if log.timestamp >= cutoff]

            if level:
                logs = [log for log in logs if log.level == level]
            if source:
                logs = [log for log in logs if log.source == source]

            return logs

    def search_logs(self, query: str, duration: timedelta = timedelta(hours=24)) -> List[LogEntry]:
        """Search logs by query string."""
        with self._lock:
            cutoff = datetime.utcnow() - duration
            return [log for log in self.logs
                   if log.timestamp >= cutoff and query.lower() in log.message.lower()]


class PerformanceProfiler:
    """Performance profiling and monitoring."""

    def __init__(self):
        self.profiles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start_profile(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a performance profile."""
        profile_id = str(uuid.uuid4())
        with self._lock:
            self.active_profiles[profile_id] = {
                "name": name,
                "start_time": time.time(),
                "metadata": metadata or {},
                "checkpoints": []
            }
        return profile_id

    def add_checkpoint(self, profile_id: str, checkpoint_name: str) -> None:
        """Add a checkpoint to a profile."""
        with self._lock:
            if profile_id in self.active_profiles:
                profile = self.active_profiles[profile_id]
                checkpoint_time = time.time()
                start_time = profile["start_time"]

                profile["checkpoints"].append({
                    "name": checkpoint_name,
                    "timestamp": checkpoint_time,
                    "elapsed": checkpoint_time - start_time
                })

    def end_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """End a performance profile."""
        with self._lock:
            if profile_id not in self.active_profiles:
                return None

            profile = self.active_profiles.pop(profile_id)
            end_time = time.time()
            profile["end_time"] = end_time
            profile["total_duration"] = end_time - profile["start_time"]

            # Store completed profile
            if len(self.profiles[profile["name"]]) >= 100:
                self.profiles[profile["name"]].pop(0)
            self.profiles[profile["name"]].append(profile)

            return profile

    def get_profile_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a profile type."""
        with self._lock:
            if name not in self.profiles:
                return {}

            profiles = self.profiles[name]
            if not profiles:
                return {}

            durations = [p["total_duration"] for p in profiles]
            return {
                "count": len(profiles),
                "avg_duration": statistics.mean(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "median_duration": statistics.median(durations),
                "latest_duration": durations[-1]
            }


class SystemMonitor:
    """System resource monitoring."""

    def __init__(self):
        self.system_info = self._get_system_info()
        self.baseline_metrics = {}

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total
        }

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "percentage": mem.percent
        }

    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage statistics."""
        disk = psutil.disk_usage('/')
        return {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percentage": disk.percent
        }

    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        net = psutil.net_io_counters()
        return {
            "bytes_sent": net.bytes_sent,
            "bytes_recv": net.bytes_recv,
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv
        }

    def get_process_info(self) -> Dict[str, Any]:
        """Get current process information."""
        process = psutil.Process()
        return {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(),
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "status": process.status()
        }

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_usage": self.get_cpu_usage(),
            "memory": self.get_memory_usage(),
            "disk": self.get_disk_usage(),
            "network": self.get_network_stats(),
            "process": self.get_process_info(),
            "system_info": self.system_info
        }


class AnalyticsEngine:
    """Analytics and insights engine."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.kpis: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def track_event(self, event_type: str, user_id: Optional[str] = None,
                   properties: Optional[Dict[str, Any]] = None) -> None:
        """Track a user or system event."""
        event = {
            "event_type": event_type,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "properties": properties or {}
        }

        with self._lock:
            self.events.append(event)

            # Maintain only recent events
            if len(self.events) > 10000:
                self.events = self.events[-5000:]

    def start_user_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start tracking a user session."""
        with self._lock:
            self.user_sessions[user_id] = {
                "start_time": datetime.utcnow(),
                "metadata": metadata or {},
                "events": [],
                "active": True
            }

    def end_user_session(self, user_id: str) -> None:
        """End a user session."""
        with self._lock:
            if user_id in self.user_sessions:
                session = self.user_sessions[user_id]
                session["end_time"] = datetime.utcnow()
                session["duration"] = (session["end_time"] - session["start_time"]).total_seconds()
                session["active"] = False

    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get analytics for a specific user."""
        with self._lock:
            user_events = [e for e in self.events if e["user_id"] == user_id]
            session = self.user_sessions.get(user_id, {})

            return {
                "total_events": len(user_events),
                "event_types": list(set(e["event_type"] for e in user_events)),
                "session_duration": session.get("duration", 0),
                "session_active": session.get("active", False),
                "last_activity": max((e["timestamp"] for e in user_events), default=None)
            }

    def get_system_analytics(self, duration: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Get system-wide analytics."""
        cutoff = datetime.utcnow() - duration

        with self._lock:
            recent_events = [e for e in self.events
                           if datetime.fromisoformat(e["timestamp"]) >= cutoff]

            event_types = defaultdict(int)
            for event in recent_events:
                event_types[event["event_type"]] += 1

            return {
                "total_events": len(recent_events),
                "unique_users": len(set(e["user_id"] for e in recent_events if e["user_id"])),
                "event_distribution": dict(event_types),
                "events_per_day": len(recent_events) / max(duration.days, 1),
                "most_common_event": max(event_types.items(), key=lambda x: x[1], default=(None, 0))[0]
            }

    def calculate_kpis(self) -> Dict[str, Any]:
        """Calculate key performance indicators."""
        analytics = self.get_system_analytics()

        # Example KPIs
        kpis = {
            "user_engagement": analytics["events_per_day"],
            "system_utilization": analytics["total_events"] / max(analytics["unique_users"], 1),
            "event_diversity": len(analytics["event_distribution"]),
            "timestamp": datetime.utcnow().isoformat()
        }

        with self._lock:
            self.kpis = kpis

        return kpis


class MonitoringDashboard:
    """Real-time monitoring dashboard."""

    def __init__(self, metrics_collector: MetricsCollector,
                 alert_manager: AlertManager,
                 analytics_engine: AnalyticsEngine):
        self.metrics = metrics_collector
        self.alerts = alert_manager
        self.analytics = analytics_engine
        self.dashboard_data: Dict[str, Any] = {}

    def update_dashboard(self) -> None:
        """Update dashboard data."""
        self.dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self._get_key_metrics(),
            "alerts": self._get_alert_summary(),
            "analytics": self.analytics.calculate_kpis(),
            "system_health": self._get_system_health()
        }

    def _get_key_metrics(self) -> Dict[str, Any]:
        """Get key metrics for dashboard."""
        key_metrics = {}
        metric_names = ["cpu_usage", "memory_usage", "request_count", "error_rate"]

        for name in metric_names:
            metric = self.metrics.get_metric(name)
            if metric:
                key_metrics[name] = {
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat() if metric.timestamp else datetime.utcnow().isoformat()
                }

        return key_metrics

    def _get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary for dashboard."""
        active_alerts = self.alerts.get_active_alerts()
        severity_counts = defaultdict(int)

        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1

        return {
            "total_active": len(active_alerts),
            "by_severity": dict(severity_counts),
            "critical_alerts": [a.message for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        }

    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_score = 100
        issues = []

        # Check for critical alerts
        critical_alerts = [a for a in self.alerts.get_active_alerts()
                          if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            health_score -= len(critical_alerts) * 20
            issues.extend([a.message for a in critical_alerts])

        # Check key metrics
        cpu_metric = self.metrics.get_metric("cpu_usage")
        if cpu_metric and cpu_metric.value > 90:
            health_score -= 10
            issues.append("High CPU usage detected")

        memory_metric = self.metrics.get_metric("memory_usage")
        if memory_metric and memory_metric.value > 90:
            health_score -= 10
            issues.append("High memory usage detected")

        return {
            "score": max(0, health_score),
            "status": "healthy" if health_score >= 80 else "warning" if health_score >= 60 else "critical",
            "issues": issues
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        return self.dashboard_data


class MonitoringOrchestrator:
    """Main orchestrator for monitoring and analytics."""

    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.log_aggregator = LogAggregator()
        self.performance_profiler = PerformanceProfiler()
        self.system_monitor = SystemMonitor()
        self.analytics_engine = AnalyticsEngine()
        self.dashboard = MonitoringDashboard(
            self.metrics_collector,
            self.alert_manager,
            self.analytics_engine
        )
        self._monitoring_thread: Optional[threading.Thread] = None
        self._running = False

    def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self._running:
            return

        self._running = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()

        logger.info("Monitoring system started")

    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self._running = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Monitoring system stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect system metrics
                system_metrics = self.system_monitor.collect_system_metrics()

                # Record key metrics
                self.metrics_collector.record_metric(Metric(
                    name="cpu_usage",
                    value=system_metrics["cpu_usage"],
                    metric_type=MetricType.GAUGE,
                    description="CPU usage percentage"
                ))

                self.metrics_collector.record_metric(Metric(
                    name="memory_usage",
                    value=system_metrics["memory"]["percentage"],
                    metric_type=MetricType.GAUGE,
                    description="Memory usage percentage"
                ))

                # Update dashboard
                self.dashboard.update_dashboard()

                # Check alert rules
                for rule_name, rule_func in self.alert_manager.alert_rules.items():
                    rule_func()

                time.sleep(30)  # Collect metrics every 30 seconds

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error

    def setup_default_alerts(self) -> None:
        """Set up default alert rules."""
        # High CPU usage alert
        self.alert_manager.add_alert_rule(
            name="high_cpu_usage",
            condition=lambda: (self.metrics_collector.get_metric("cpu_usage") or Metric("", 0, MetricType.GAUGE)).value > 90,
            severity=AlertSeverity.WARNING,
            message="CPU usage is above 90%",
            threshold=90
        )

        # High memory usage alert
        self.alert_manager.add_alert_rule(
            name="high_memory_usage",
            condition=lambda: (self.metrics_collector.get_metric("memory_usage") or Metric("", 0, MetricType.GAUGE)).value > 90,
            severity=AlertSeverity.ERROR,
            message="Memory usage is above 90%",
            threshold=90
        )

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get overall monitoring system status."""
        return {
            "running": self._running,
            "metrics_collected": len(self.metrics_collector.metrics),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "logs_aggregated": len(self.log_aggregator.logs),
            "active_profiles": len(self.performance_profiler.active_profiles),
            "dashboard_data": self.dashboard.get_dashboard_data()
        }


# Export main classes
__all__ = [
    "MetricType",
    "AlertSeverity",
    "LogLevel",
    "Metric",
    "Alert",
    "LogEntry",
    "MetricsCollector",
    "AlertManager",
    "LogAggregator",
    "PerformanceProfiler",
    "SystemMonitor",
    "AnalyticsEngine",
    "MonitoringDashboard",
    "MonitoringOrchestrator"
]
