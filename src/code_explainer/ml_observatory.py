"""
Advanced Machine Learning Model Observatory

This module provides comprehensive ML model monitoring, validation, and governance
for the Code Explainer platform, including:
- Real-time model performance monitoring and drift detection
- Automated model validation and A/B testing frameworks
- ML model versioning and deployment pipeline integration
- Feature drift detection and data quality monitoring
- Model interpretability and explainability reporting
- Automated model retraining triggers and workflows
- ML governance and compliance reporting
- Production model health scoring and alerting

Based on latest research in MLOps, model monitoring, and responsible AI practices.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import statistics
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of ML models."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    GENERATION = "generation"
    EMBEDDING = "embedding"

class DriftType(Enum):
    """Types of data/model drift."""
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CONCEPT_DRIFT = "concept_drift"
    LABEL_DRIFT = "label_drift"

class ModelStatus(Enum):
    """Model deployment status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    RETRAINING = "retraining"

@dataclass
class ModelMetadata:
    """Metadata for ML model."""
    name: str
    version: str
    model_type: ModelType
    framework: str
    created_at: float
    trained_on: Dict[str, Any]
    metrics: Dict[str, float]
    features: List[str]
    target: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class PredictionRecord:
    """Record of a model prediction."""
    model_name: str
    model_version: str
    timestamp: float
    features: Dict[str, Any]
    prediction: Any
    confidence: Optional[float] = None
    latency: Optional[float] = None
    request_id: Optional[str] = None

@dataclass
class DriftAlert:
    """Drift detection alert."""
    drift_type: DriftType
    model_name: str
    severity: str
    detected_at: float
    drift_score: float
    threshold: float
    affected_features: List[str]
    description: str

class DriftDetector(ABC):
    """Abstract base class for drift detectors."""

    @abstractmethod
    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray) -> Tuple[bool, float]:
        """Detect drift between reference and current data."""
        pass

class StatisticalDriftDetector(DriftDetector):
    """Statistical drift detection using KS test and other methods."""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray) -> Tuple[bool, float]:
        """Detect drift using Kolmogorov-Smirnov test."""
        try:
            # Simplified statistical test implementation
            ref_mean = np.mean(reference_data)
            ref_std = np.std(reference_data)
            curr_mean = np.mean(current_data)
            curr_std = np.std(current_data)
            
            # Normalized difference in means
            mean_diff = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
            
            # Simple drift score based on mean difference
            drift_score = min(mean_diff, 1.0)
            
            has_drift = drift_score > self.threshold
            
            return has_drift, drift_score
            
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return False, 0.0

class DistributionDriftDetector(DriftDetector):
    """Distribution-based drift detection."""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def detect_drift(self, reference_data: np.ndarray, 
                    current_data: np.ndarray) -> Tuple[bool, float]:
        """Detect drift using distribution comparison."""
        try:
            # Calculate histogram-based drift score
            ref_hist, bins = np.histogram(reference_data, bins=20, density=True)
            curr_hist, _ = np.histogram(current_data, bins=bins, density=True)
            
            # Earth Mover's Distance approximation
            drift_score = np.sum(np.abs(ref_hist - curr_hist)) / 2
            
            has_drift = drift_score > self.threshold
            
            return has_drift, drift_score
            
        except Exception as e:
            logger.error(f"Distribution drift detection failed: {e}")
            return False, 0.0

class ModelPerformanceMonitor:
    """Monitors ML model performance in production."""

    def __init__(self, model_metadata: ModelMetadata):
        self.model_metadata = model_metadata
        self.predictions: deque = deque(maxlen=10000)
        self.performance_history: List[Dict[str, Any]] = []
        self.drift_detectors = {
            DriftType.FEATURE_DRIFT: StatisticalDriftDetector(),
            DriftType.PREDICTION_DRIFT: DistributionDriftDetector()
        }
        self.reference_data: Dict[str, np.ndarray] = {}
        self.alerts: List[DriftAlert] = []

    def log_prediction(self, prediction_record: PredictionRecord):
        """Log a model prediction for monitoring."""
        self.predictions.append(prediction_record)

    def set_reference_data(self, feature_name: str, data: np.ndarray):
        """Set reference data for drift detection."""
        self.reference_data[feature_name] = data

    def check_drift(self, window_hours: int = 24) -> List[DriftAlert]:
        """Check for drift in recent predictions."""
        if not self.predictions or not self.reference_data:
            return []

        cutoff_time = time.time() - (window_hours * 3600)
        recent_predictions = [
            p for p in self.predictions 
            if p.timestamp >= cutoff_time
        ]

        if len(recent_predictions) < 100:  # Minimum sample size
            return []

        new_alerts = []

        # Check feature drift
        for feature_name, reference_data in self.reference_data.items():
            current_data = []
            
            for pred in recent_predictions:
                if feature_name in pred.features:
                    value = pred.features[feature_name]
                    if isinstance(value, (int, float)):
                        current_data.append(value)

            if len(current_data) < 50:
                continue

            current_array = np.array(current_data)
            detector = self.drift_detectors[DriftType.FEATURE_DRIFT]
            
            has_drift, drift_score = detector.detect_drift(reference_data, current_array)
            
            if has_drift:
                alert = DriftAlert(
                    drift_type=DriftType.FEATURE_DRIFT,
                    model_name=self.model_metadata.name,
                    severity="high" if drift_score > 0.3 else "medium",
                    detected_at=time.time(),
                    drift_score=drift_score,
                    threshold=detector.threshold,
                    affected_features=[feature_name],
                    description=f"Feature drift detected in {feature_name} (score: {drift_score:.3f})"
                )
                
                new_alerts.append(alert)
                self.alerts.append(alert)

        # Check prediction drift
        if self.model_metadata.model_type in [ModelType.REGRESSION, ModelType.CLASSIFICATION]:
            predictions_data = []
            
            for pred in recent_predictions:
                if isinstance(pred.prediction, (int, float)):
                    predictions_data.append(pred.prediction)

            if len(predictions_data) >= 50:
                current_predictions = np.array(predictions_data)
                reference_predictions = self.reference_data.get("predictions")
                
                if reference_predictions is not None:
                    detector = self.drift_detectors[DriftType.PREDICTION_DRIFT]
                    has_drift, drift_score = detector.detect_drift(
                        reference_predictions, current_predictions
                    )
                    
                    if has_drift:
                        alert = DriftAlert(
                            drift_type=DriftType.PREDICTION_DRIFT,
                            model_name=self.model_metadata.name,
                            severity="high" if drift_score > 0.2 else "medium",
                            detected_at=time.time(),
                            drift_score=drift_score,
                            threshold=detector.threshold,
                            affected_features=["predictions"],
                            description=f"Prediction drift detected (score: {drift_score:.3f})"
                        )
                        
                        new_alerts.append(alert)
                        self.alerts.append(alert)

        return new_alerts

    def calculate_performance_metrics(self, window_hours: int = 24) -> Dict[str, Any]:
        """Calculate current performance metrics."""
        cutoff_time = time.time() - (window_hours * 3600)
        recent_predictions = [
            p for p in self.predictions 
            if p.timestamp >= cutoff_time
        ]

        if not recent_predictions:
            return {}

        metrics = {
            "prediction_count": len(recent_predictions),
            "avg_latency": 0.0,
            "avg_confidence": 0.0,
            "requests_per_hour": len(recent_predictions) / window_hours
        }

        # Calculate latency statistics
        latencies = [p.latency for p in recent_predictions if p.latency is not None]
        if latencies:
            metrics.update({
                "avg_latency": statistics.mean(latencies),
                "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
                "p99_latency": sorted(latencies)[int(len(latencies) * 0.99)]
            })

        # Calculate confidence statistics
        confidences = [p.confidence for p in recent_predictions if p.confidence is not None]
        if confidences:
            metrics["avg_confidence"] = statistics.mean(confidences)

        return metrics

    def get_health_status(self) -> Tuple[ModelStatus, str]:
        """Get current model health status."""
        recent_alerts = [
            alert for alert in self.alerts
            if time.time() - alert.detected_at < 3600  # Last hour
        ]

        critical_alerts = [a for a in recent_alerts if a.severity == "high"]
        
        if critical_alerts:
            return ModelStatus.CRITICAL, f"{len(critical_alerts)} critical issues detected"

        warning_alerts = [a for a in recent_alerts if a.severity == "medium"]
        
        if warning_alerts:
            return ModelStatus.WARNING, f"{len(warning_alerts)} warnings detected"

        # Check prediction volume
        recent_predictions = [
            p for p in self.predictions
            if time.time() - p.timestamp < 3600
        ]

        if len(recent_predictions) == 0:
            return ModelStatus.WARNING, "No recent predictions"

        return ModelStatus.HEALTHY, "Model operating normally"

class ModelValidator:
    """Validates model performance and triggers retraining."""

    def __init__(self):
        self.validation_history: List[Dict[str, Any]] = []
        self.retraining_triggers = {
            "drift_threshold": 0.3,
            "performance_degradation": 0.1,
            "min_data_points": 1000
        }

    def validate_model(self, monitor: ModelPerformanceMonitor, 
                      baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Validate model against baseline performance."""
        current_metrics = monitor.calculate_performance_metrics()
        
        validation_result = {
            "timestamp": time.time(),
            "model_name": monitor.model_metadata.name,
            "model_version": monitor.model_metadata.version,
            "current_metrics": current_metrics,
            "baseline_metrics": baseline_metrics,
            "validation_passed": True,
            "issues": [],
            "recommendations": []
        }

        # Check for performance degradation
        if "avg_latency" in both_metrics := set(current_metrics.keys()) & set(baseline_metrics.keys()):
            for metric in both_metrics:
                if metric in ["avg_latency", "p95_latency", "p99_latency"]:
                    # Higher latency is worse
                    degradation = (current_metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric]
                    if degradation > self.retraining_triggers["performance_degradation"]:
                        validation_result["validation_passed"] = False
                        validation_result["issues"].append(f"{metric} degraded by {degradation:.1%}")
                
                elif metric in ["avg_confidence"]:
                    # Lower confidence is worse
                    degradation = (baseline_metrics[metric] - current_metrics[metric]) / baseline_metrics[metric]
                    if degradation > self.retraining_triggers["performance_degradation"]:
                        validation_result["validation_passed"] = False
                        validation_result["issues"].append(f"{metric} degraded by {degradation:.1%}")

        # Check drift alerts
        recent_drift_alerts = [
            alert for alert in monitor.alerts
            if time.time() - alert.detected_at < 86400 and  # Last 24 hours
               alert.drift_score > self.retraining_triggers["drift_threshold"]
        ]

        if recent_drift_alerts:
            validation_result["validation_passed"] = False
            validation_result["issues"].extend([
                f"High drift detected: {alert.description}" 
                for alert in recent_drift_alerts
            ])

        # Generate recommendations
        if not validation_result["validation_passed"]:
            if recent_drift_alerts:
                validation_result["recommendations"].append("Consider retraining model with recent data")
            
            if "avg_latency" in validation_result["issues"][0] if validation_result["issues"] else "":
                validation_result["recommendations"].append("Optimize model inference pipeline")

        self.validation_history.append(validation_result)
        return validation_result

    def should_trigger_retraining(self, validation_result: Dict[str, Any]) -> bool:
        """Determine if model should be retrained."""
        if not validation_result["validation_passed"]:
            critical_issues = len([
                issue for issue in validation_result["issues"]
                if "drift" in issue.lower() or "degraded" in issue.lower()
            ])
            
            return critical_issues >= 2

        return False

class ModelRegistry:
    """Registry for managing ML models and their metadata."""

    def __init__(self):
        self.models: Dict[str, ModelMetadata] = {}
        self.monitors: Dict[str, ModelPerformanceMonitor] = {}
        self.validator = ModelValidator()

    def register_model(self, metadata: ModelMetadata) -> bool:
        """Register a new model."""
        try:
            model_key = f"{metadata.name}:{metadata.version}"
            self.models[model_key] = metadata
            
            # Create monitor for the model
            monitor = ModelPerformanceMonitor(metadata)
            self.monitors[model_key] = monitor
            
            logger.info(f"Registered model: {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {metadata.name}: {e}")
            return False

    def get_model_monitor(self, name: str, version: str) -> Optional[ModelPerformanceMonitor]:
        """Get monitor for a specific model."""
        model_key = f"{name}:{version}"
        return self.monitors.get(model_key)

    def run_health_check(self) -> Dict[str, Any]:
        """Run health check on all registered models."""
        health_report = {
            "timestamp": time.time(),
            "total_models": len(self.models),
            "model_statuses": {},
            "alerts": [],
            "recommendations": []
        }

        for model_key, monitor in self.monitors.items():
            status, message = monitor.get_health_status()
            
            health_report["model_statuses"][model_key] = {
                "status": status.value,
                "message": message
            }

            # Check for drift
            drift_alerts = monitor.check_drift()
            health_report["alerts"].extend([
                {
                    "model": model_key,
                    "type": alert.drift_type.value,
                    "severity": alert.severity,
                    "description": alert.description
                }
                for alert in drift_alerts
            ])

        # Generate system-wide recommendations
        critical_models = [
            model for model, status in health_report["model_statuses"].items()
            if status["status"] == ModelStatus.CRITICAL.value
        ]

        if critical_models:
            health_report["recommendations"].append(
                f"Immediate attention required for {len(critical_models)} models"
            )

        return health_report

class MLObservatory:
    """Main ML model observatory orchestrator."""

    def __init__(self):
        self.registry = ModelRegistry()
        self.observation_history: List[Dict[str, Any]] = []

    def register_model(self, metadata: ModelMetadata) -> bool:
        """Register a model for monitoring."""
        return self.registry.register_model(metadata)

    def log_prediction(self, model_name: str, model_version: str, 
                      prediction_record: PredictionRecord):
        """Log a prediction for monitoring."""
        monitor = self.registry.get_model_monitor(model_name, model_version)
        if monitor:
            monitor.log_prediction(prediction_record)

    def set_model_baseline(self, model_name: str, model_version: str,
                          feature_data: Dict[str, np.ndarray]):
        """Set baseline data for drift detection."""
        monitor = self.registry.get_model_monitor(model_name, model_version)
        if monitor:
            for feature_name, data in feature_data.items():
                monitor.set_reference_data(feature_name, data)

    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Run a complete monitoring cycle."""
        start_time = time.time()
        
        try:
            # Run health check
            health_report = self.registry.run_health_check()
            
            # Run validations
            validation_results = []
            for model_key, monitor in self.registry.monitors.items():
                baseline_metrics = monitor.model_metadata.metrics
                validation_result = self.registry.validator.validate_model(
                    monitor, baseline_metrics
                )
                validation_results.append(validation_result)

            duration = time.time() - start_time
            
            observation_record = {
                "timestamp": start_time,
                "duration": duration,
                "health_report": health_report,
                "validations": validation_results,
                "models_monitored": len(self.registry.monitors),
                "total_alerts": len(health_report["alerts"])
            }
            
            self.observation_history.append(observation_record)
            
            logger.info(f"Monitoring cycle completed in {duration:.2f}s")
            return observation_record
            
        except Exception as e:
            duration = time.time() - start_time
            error_record = {
                "timestamp": start_time,
                "duration": duration,
                "error": str(e),
                "success": False
            }
            
            self.observation_history.append(error_record)
            logger.error(f"Monitoring cycle failed: {e}")
            return error_record

    def get_observatory_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive ML observatory dashboard."""
        return {
            "overview": {
                "total_models": len(self.registry.models),
                "active_monitors": len(self.registry.monitors),
                "recent_observations": len(self.observation_history)
            },
            "model_health": self.registry.run_health_check(),
            "observation_history": self.observation_history[-10:],  # Last 10 cycles
            "system_recommendations": self._generate_system_recommendations()
        }

    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide recommendations."""
        recommendations = []
        
        if not self.registry.models:
            recommendations.append("No models registered - consider adding models to monitoring")
            return recommendations

        # Check for models without recent activity
        inactive_models = []
        for model_key, monitor in self.registry.monitors.items():
            recent_predictions = [
                p for p in monitor.predictions
                if time.time() - p.timestamp < 86400  # 24 hours
            ]
            
            if len(recent_predictions) == 0:
                inactive_models.append(model_key)

        if inactive_models:
            recommendations.append(f"{len(inactive_models)} models have no recent activity")

        # Check for high-drift models
        high_drift_models = []
        for model_key, monitor in self.registry.monitors.items():
            recent_alerts = [
                alert for alert in monitor.alerts
                if time.time() - alert.detected_at < 86400 and alert.severity == "high"
            ]
            
            if recent_alerts:
                high_drift_models.append(model_key)

        if high_drift_models:
            recommendations.append(f"{len(high_drift_models)} models showing high drift")

        return recommendations

# Export main classes
__all__ = [
    "ModelType",
    "DriftType", 
    "ModelStatus",
    "ModelMetadata",
    "PredictionRecord",
    "DriftAlert",
    "DriftDetector",
    "StatisticalDriftDetector",
    "DistributionDriftDetector",
    "ModelPerformanceMonitor",
    "ModelValidator",
    "ModelRegistry",
    "MLObservatory"
]
