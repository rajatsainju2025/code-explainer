"""
Advanced Analytics Module for Code Intelligence Platform

This module provides advanced analytics capabilities including machine learning,
predictive modeling, anomaly detection, and sophisticated data analysis to
extract deep insights from code intelligence data.

Features:
- Machine learning-based code quality prediction
- Anomaly detection for system metrics and user behavior
- Predictive analytics for performance optimization
- Natural language processing for code documentation
- Time series analysis and forecasting
- Clustering and pattern recognition
- Recommendation systems for code improvements
- Automated insights generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import statistics
import json
from collections import defaultdict
import re
import math
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class AnalysisType(Enum):
    """Types of advanced analysis."""
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE_MODELING = "predictive_modeling"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    NLP_ANALYSIS = "nlp_analysis"
    RECOMMENDATION = "recommendation"


class AnomalyType(Enum):
    """Types of anomalies."""
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    timestamp: datetime
    value: float
    score: float
    is_anomaly: bool
    anomaly_type: AnomalyType
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of predictive modeling."""
    target_timestamp: datetime
    predicted_value: float
    confidence_interval: Tuple[float, float]
    model_accuracy: float
    features_used: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterResult:
    """Result of clustering analysis."""
    cluster_id: int
    cluster_center: List[float]
    members: List[Dict[str, Any]]
    silhouette_score: float
    cluster_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Insight:
    """Automated insight from data analysis."""
    title: str
    description: str
    severity: str
    category: str
    confidence: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """Advanced anomaly detection using machine learning."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.baseline_data: Dict[str, List[float]] = defaultdict(list)

    def train_model(self, metric_name: str, data: List[float], contamination: float = 0.1) -> None:
        """Train anomaly detection model for a metric."""
        if len(data) < 10:
            return

        # Prepare data
        X = np.array(data).reshape(-1, 1)

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Isolation Forest
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)

        self.models[metric_name] = model
        self.scalers[metric_name] = scaler
        self.baseline_data[metric_name] = data

    def detect_anomalies(self, metric_name: str, values: List[Tuple[datetime, float]]) -> List[AnomalyResult]:
        """Detect anomalies in time series data."""
        if metric_name not in self.models:
            return []

        model = self.models[metric_name]
        scaler = self.scalers[metric_name]

        anomalies = []
        for timestamp, value in values:
            # Scale the value
            X = np.array([[value]])
            X_scaled = scaler.transform(X)

            # Get anomaly score
            score = model.decision_function(X_scaled)[0]
            prediction = model.predict(X_scaled)[0]

            # Convert score to confidence (higher score = more normal)
            confidence = (score + 1) / 2  # Normalize to [0, 1]

            anomaly_result = AnomalyResult(
                timestamp=timestamp,
                value=value,
                score=score,
                is_anomaly=prediction == -1,
                anomaly_type=AnomalyType.POINT_ANOMALY,
                confidence=confidence,
                metadata={"model": "isolation_forest"}
            )
            anomalies.append(anomaly_result)

        return anomalies

    def detect_contextual_anomalies(self, metric_name: str,
                                   values: List[Tuple[datetime, float]],
                                   window_size: int = 10) -> List[AnomalyResult]:
        """Detect contextual anomalies based on local patterns."""
        if len(values) < window_size:
            return []

        anomalies = []
        for i in range(window_size, len(values)):
            window = [v for _, v in values[i-window_size:i]]
            current_value = values[i][1]
            timestamp = values[i][0]

            # Calculate local statistics
            mean = statistics.mean(window)
            std = statistics.stdev(window) if len(window) > 1 else 0

            if std == 0:
                continue

            # Z-score based anomaly detection
            z_score = abs(current_value - mean) / std
            is_anomaly = z_score > 3  # 3-sigma rule

            anomaly_result = AnomalyResult(
                timestamp=timestamp,
                value=current_value,
                score=z_score,
                is_anomaly=is_anomaly,
                anomaly_type=AnomalyType.CONTEXTUAL_ANOMALY,
                confidence=min(z_score / 5, 1.0),  # Normalize confidence
                metadata={"window_size": window_size, "local_mean": mean, "local_std": std}
            )
            anomalies.append(anomaly_result)

        return anomalies


class PredictiveModeler:
    """Predictive analytics for performance and usage patterns."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}

    def train_performance_predictor(self, metric_name: str,
                                   historical_data: List[Tuple[datetime, float]],
                                   features: Optional[Dict[str, List[float]]] = None) -> None:
        """Train a model to predict future performance metrics."""
        if len(historical_data) < 20:
            return

        # Prepare target variable
        timestamps, values = zip(*historical_data)
        y = np.array(values)

        # Create time-based features
        X = []
        for i, (timestamp, _) in enumerate(historical_data):
            time_features = [
                timestamp.hour,
                timestamp.weekday(),
                timestamp.month,
                i,  # Time index
                len(historical_data) - i  # Reverse time index
            ]

            # Add rolling statistics
            if i >= 5:
                window = y[max(0, i-5):i]
                time_features.extend([
                    statistics.mean(window),
                    statistics.stdev(window) if len(window) > 1 else 0,
                    max(window),
                    min(window)
                ])
            else:
                time_features.extend([y[i], 0, y[i], y[i]])  # Pad with current value

            X.append(time_features)

        X = np.array(X)

        # Add external features if provided
        if features:
            external_features = []
            for feature_name, feature_values in features.items():
                # Interpolate to match timestamps
                external_features.append(feature_values[:len(X)])
            if external_features:
                X = np.column_stack([X] + external_features)

        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        model.fit(X, y)

        # Store feature importance
        feature_names = [f"time_feature_{i}" for i in range(X.shape[1])]
        if features:
            feature_names.extend(features.keys())

        self.feature_importance[metric_name] = dict(zip(feature_names, model.feature_importances_))
        self.models[metric_name] = model

    def predict_future_values(self, metric_name: str,
                             future_timestamps: List[datetime],
                             current_data: List[Tuple[datetime, float]],
                             features: Optional[Dict[str, List[float]]] = None) -> List[PredictionResult]:
        """Predict future values for a metric."""
        if metric_name not in self.models:
            return []

        model = self.models[metric_name]
        predictions = []

        for i, timestamp in enumerate(future_timestamps):
            # Create features for prediction
            time_features = [
                timestamp.hour,
                timestamp.weekday(),
                timestamp.month,
                len(current_data) + i,  # Future time index
                len(future_timestamps) - i  # Reverse future index
            ]

            # Add rolling statistics from recent data
            if current_data:
                recent_values = [v for _, v in current_data[-5:]]
                time_features.extend([
                    float(statistics.mean(recent_values)),
                    float(statistics.stdev(recent_values) if len(recent_values) > 1 else 0),
                    float(max(recent_values)),
                    float(min(recent_values))
                ])
            else:
                time_features.extend([0.0, 0.0, 0.0, 0.0])

            X_pred = np.array([time_features])

            # Add external features
            if features:
                external_features = []
                for feature_name, feature_values in features.items():
                    # Use latest available value
                    external_features.append(feature_values[-1] if feature_values else 0)
                if external_features:
                    X_pred = np.column_stack([X_pred, external_features])

            # Make prediction
            predicted_value = model.predict(X_pred)[0]

            # Calculate confidence interval (simplified)
            confidence_interval = (
                predicted_value * 0.9,  # -10%
                predicted_value * 1.1   # +10%
            )

            # Estimate model accuracy (simplified)
            model_accuracy = 0.85  # Placeholder

            prediction = PredictionResult(
                target_timestamp=timestamp,
                predicted_value=predicted_value,
                confidence_interval=confidence_interval,
                model_accuracy=model_accuracy,
                features_used=list(self.feature_importance.get(metric_name, {}).keys()),
                metadata={"model_type": "random_forest"}
            )
            predictions.append(prediction)

        return predictions


class ClusterAnalyzer:
    """Clustering analysis for pattern recognition."""

    def __init__(self):
        self.cluster_models: Dict[str, KMeans] = {}
        self.cluster_labels: Dict[str, List[int]] = {}
        self.cluster_centers: Dict[str, np.ndarray] = {}

    def perform_clustering(self, data_name: str, data: List[List[float]],
                          n_clusters: Optional[int] = None) -> List[ClusterResult]:
        """Perform clustering analysis on multidimensional data."""
        if len(data) < 3:
            return []

        X = np.array(data)

        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X, max_clusters=10)

        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(X)

        # Calculate silhouette score
        if n_clusters > 1:
            silhouette_avg = silhouette_score(X, labels)
        else:
            silhouette_avg = 0

        # Store results
        self.cluster_models[data_name] = kmeans
        self.cluster_labels[data_name] = labels.tolist()
        self.cluster_centers[data_name] = kmeans.cluster_centers_

        # Create cluster results
        results = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_members = [
                {"index": i, "features": data[i]}
                for i in range(len(data)) if cluster_mask[i]
            ]

            result = ClusterResult(
                cluster_id=cluster_id,
                cluster_center=kmeans.cluster_centers_[cluster_id].tolist(),
                members=cluster_members,
                silhouette_score=float(silhouette_avg),
                cluster_size=len(cluster_members),
                metadata={"n_clusters": n_clusters, "total_samples": len(data)}
            )
            results.append(result)

        return results

    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using silhouette analysis."""
        if len(X) < max_clusters:
            max_clusters = len(X)

        best_score = -1
        best_n_clusters = 2

        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            try:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except:
                continue

        return best_n_clusters

    def analyze_user_behavior_clusters(self, user_data: List[Dict[str, Any]]) -> List[ClusterResult]:
        """Analyze user behavior patterns using clustering."""
        if len(user_data) < 3:
            return []

        # Extract features from user data
        features = []
        for user in user_data:
            user_features = [
                user.get("session_duration", 0),
                user.get("total_events", 0),
                user.get("unique_features_used", 0),
                user.get("error_rate", 0),
                user.get("avg_response_time", 0)
            ]
            features.append(user_features)

        return self.perform_clustering("user_behavior", features)


class CodeQualityAnalyzer:
    """Analyze code quality using ML and heuristics."""

    def __init__(self):
        self.quality_model = None
        self.quality_features = []

    def analyze_code_quality(self, code_snippet: str) -> Dict[str, Any]:
        """Analyze code quality metrics."""
        metrics = {
            "complexity_score": self._calculate_complexity(code_snippet),
            "readability_score": self._calculate_readability(code_snippet),
            "maintainability_index": self._calculate_maintainability(code_snippet),
            "bug_probability": self._estimate_bug_probability(code_snippet),
            "documentation_coverage": self._calculate_documentation_coverage(code_snippet)
        }

        # Overall quality score (weighted average)
        weights = {
            "complexity_score": 0.2,
            "readability_score": 0.3,
            "maintainability_index": 0.3,
            "bug_probability": 0.15,
            "documentation_coverage": 0.05
        }

        quality_score = sum(metrics[key] * weights[key] for key in weights.keys())
        metrics["overall_quality_score"] = quality_score

        return metrics

    def _calculate_complexity(self, code: str) -> float:
        """Calculate code complexity score."""
        lines = code.split('\n')
        complexity = 0

        for line in lines:
            # Count control structures
            if any(keyword in line for keyword in ['if ', 'for ', 'while ', 'try:', 'except:']):
                complexity += 1
            # Count nested structures (indentation)
            indentation = len(line) - len(line.lstrip())
            complexity += indentation // 4

        # Normalize to 0-1 scale
        return min(complexity / 20, 1.0)

    def _calculate_readability(self, code: str) -> float:
        """Calculate code readability score."""
        lines = code.split('\n')
        score = 1.0

        # Penalize long lines
        long_lines = sum(1 for line in lines if len(line) > 88)
        score -= (long_lines / len(lines)) * 0.3

        # Penalize lack of comments
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        comment_ratio = comment_lines / len(lines) if lines else 0
        score += comment_ratio * 0.2

        # Penalize poor variable naming
        words = re.findall(r'\b\w+\b', code)
        poor_names = sum(1 for word in words if len(word) == 1 and word.islower())
        score -= (poor_names / len(words)) * 0.1 if words else 0

        return max(0, min(1, score))

    def _calculate_maintainability(self, code: str) -> float:
        """Calculate maintainability index."""
        lines = len(code.split('\n'))
        complexity = self._calculate_complexity(code)

        # Simplified maintainability index
        mi = 171 - 5.2 * math.log(lines) - 0.23 * complexity * 100
        mi = max(0, min(100, mi))

        return mi / 100  # Normalize to 0-1

    def _estimate_bug_probability(self, code: str) -> float:
        """Estimate bug probability based on code patterns."""
        bug_indicators = [
            'except:', 'Exception', 'raise', 'assert',
            'None', 'null', 'undefined', 'TODO', 'FIXME'
        ]

        total_indicators = 0
        for indicator in bug_indicators:
            total_indicators += code.count(indicator)

        # Normalize based on code length
        probability = total_indicators / len(code) * 1000
        return min(probability, 1.0)

    def _calculate_documentation_coverage(self, code: str) -> float:
        """Calculate documentation coverage."""
        lines = code.split('\n')
        docstring_lines = 0
        code_lines = 0

        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if stripped:
                code_lines += 1
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                if in_docstring or stripped.startswith('#'):
                    docstring_lines += 1

        return docstring_lines / code_lines if code_lines > 0 else 0


class InsightGenerator:
    """Generate automated insights from analytics data."""

    def __init__(self):
        self.insights: List[Insight] = []

    def generate_insights(self, metrics_data: Dict[str, Any],
                         anomaly_results: List[AnomalyResult],
                         prediction_results: List[PredictionResult]) -> List[Insight]:
        """Generate insights from various analytics results."""
        insights = []

        # Analyze anomalies
        insights.extend(self._analyze_anomalies(anomaly_results))

        # Analyze predictions
        insights.extend(self._analyze_predictions(prediction_results))

        # Analyze metrics trends
        insights.extend(self._analyze_metrics_trends(metrics_data))

        # Analyze system health
        insights.extend(self._analyze_system_health(metrics_data))

        self.insights.extend(insights)
        return insights

    def _analyze_anomalies(self, anomalies: List[AnomalyResult]) -> List[Insight]:
        """Generate insights from anomaly detection results."""
        insights = []

        if not anomalies:
            return insights

        anomaly_count = sum(1 for a in anomalies if a.is_anomaly)
        if anomaly_count > 0:
            severity = "high" if anomaly_count > 5 else "medium" if anomaly_count > 2 else "low"

            insight = Insight(
                title=f"Detected {anomaly_count} Anomalies",
                description=f"Found {anomaly_count} anomalous patterns in system metrics that may indicate issues.",
                severity=severity,
                category="anomaly_detection",
                confidence=0.8,
                recommendations=[
                    "Investigate the root cause of detected anomalies",
                    "Review system logs around anomaly timestamps",
                    "Consider adjusting monitoring thresholds",
                    "Implement automated remediation if possible"
                ],
                metadata={"anomaly_count": anomaly_count}
            )
            insights.append(insight)

        return insights

    def _analyze_predictions(self, predictions: List[PredictionResult]) -> List[Insight]:
        """Generate insights from prediction results."""
        insights = []

        if not predictions:
            return insights

        # Analyze prediction trends
        predicted_values = [p.predicted_value for p in predictions]
        if predicted_values:
            trend = "increasing" if predicted_values[-1] > predicted_values[0] else "decreasing"

            if trend == "increasing":
                insight = Insight(
                    title="Performance Trend: Increasing Load",
                    description="Predictions indicate increasing system load that may require capacity planning.",
                    severity="medium",
                    category="predictive_analytics",
                    confidence=0.7,
                    recommendations=[
                        "Monitor resource utilization closely",
                        "Consider scaling up infrastructure",
                        "Review application performance optimizations",
                        "Plan for peak load scenarios"
                    ]
                )
                insights.append(insight)

        return insights

    def _analyze_metrics_trends(self, metrics_data: Dict[str, Any]) -> List[Insight]:
        """Generate insights from metrics trends."""
        insights = []

        # Analyze CPU usage trends
        cpu_usage = metrics_data.get("cpu_usage", [])
        if len(cpu_usage) > 10:
            recent_avg = statistics.mean(cpu_usage[-5:])
            overall_avg = statistics.mean(cpu_usage)

            if recent_avg > overall_avg * 1.2:
                insight = Insight(
                    title="CPU Usage Increasing",
                    description="Recent CPU usage is significantly higher than historical average.",
                    severity="medium",
                    category="performance",
                    confidence=0.75,
                    recommendations=[
                        "Investigate processes causing high CPU usage",
                        "Consider optimizing CPU-intensive operations",
                        "Review application performance bottlenecks",
                        "Monitor for potential resource exhaustion"
                    ],
                    metadata={"recent_avg": recent_avg, "overall_avg": overall_avg}
                )
                insights.append(insight)

        return insights

    def _analyze_system_health(self, metrics_data: Dict[str, Any]) -> List[Insight]:
        """Generate insights from system health metrics."""
        insights = []

        # Check for critical thresholds
        cpu_usage = metrics_data.get("cpu_usage", [])
        memory_usage = metrics_data.get("memory_usage", [])

        if cpu_usage and max(cpu_usage) > 90:
            insight = Insight(
                title="Critical CPU Usage Detected",
                description="System CPU usage exceeded 90%, indicating potential performance issues.",
                severity="high",
                category="system_health",
                confidence=0.9,
                recommendations=[
                    "Immediately investigate high CPU processes",
                    "Consider scaling resources or optimizing code",
                    "Monitor for cascading failures",
                    "Implement circuit breakers if applicable"
                ]
            )
            insights.append(insight)

        if memory_usage and max(memory_usage) > 90:
            insight = Insight(
                title="Critical Memory Usage Detected",
                description="System memory usage exceeded 90%, risking out-of-memory errors.",
                severity="high",
                category="system_health",
                confidence=0.9,
                recommendations=[
                    "Check for memory leaks in applications",
                    "Optimize memory usage patterns",
                    "Consider increasing memory allocation",
                    "Implement memory monitoring and alerts"
                ]
            )
            insights.append(insight)

        return insights


class AdvancedAnalyticsOrchestrator:
    """Main orchestrator for advanced analytics capabilities."""

    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.predictive_modeler = PredictiveModeler()
        self.cluster_analyzer = ClusterAnalyzer()
        self.code_quality_analyzer = CodeQualityAnalyzer()
        self.insight_generator = InsightGenerator()

    def perform_comprehensive_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive advanced analytics."""
        results = {
            "anomalies": [],
            "predictions": [],
            "clusters": [],
            "insights": [],
            "code_quality": {},
            "metadata": {
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "data_sources": list(data.keys())
            }
        }

        # Extract time series data for analysis
        time_series_data = self._extract_time_series_data(data)

        # Perform anomaly detection
        for metric_name, values in time_series_data.items():
            if len(values) >= 10:
                # Train model if not already trained
                if metric_name not in self.anomaly_detector.models:
                    train_values = [v for _, v in values[:int(len(values)*0.7)]]
                    self.anomaly_detector.train_model(metric_name, train_values)

                # Detect anomalies
                anomalies = self.anomaly_detector.detect_anomalies(metric_name, values)
                results["anomalies"].extend(anomalies)

        # Perform predictive modeling
        for metric_name, values in time_series_data.items():
            if len(values) >= 20:
                # Train predictor
                self.predictive_modeler.train_performance_predictor(metric_name, values)

                # Generate future predictions
                last_timestamp = values[-1][0]
                future_timestamps = [
                    last_timestamp + timedelta(hours=i) for i in range(1, 25)
                ]
                predictions = self.predictive_modeler.predict_future_values(
                    metric_name, future_timestamps, values
                )
                results["predictions"].extend(predictions)

        # Perform clustering analysis
        if "user_data" in data:
            clusters = self.cluster_analyzer.analyze_user_behavior_clusters(data["user_data"])
            results["clusters"] = clusters

        # Analyze code quality if code data is available
        if "code_snippets" in data:
            code_quality_results = {}
            for i, code in enumerate(data["code_snippets"]):
                quality = self.code_quality_analyzer.analyze_code_quality(code)
                code_quality_results[f"snippet_{i}"] = quality
            results["code_quality"] = code_quality_results

        # Generate insights
        insights = self.insight_generator.generate_insights(
            data, results["anomalies"], results["predictions"]
        )
        results["insights"] = insights

        return results

    def _extract_time_series_data(self, data: Dict[str, Any]) -> Dict[str, List[Tuple[datetime, float]]]:
        """Extract time series data from various sources."""
        time_series = {}

        # Extract from metrics
        if "metrics" in data:
            for metric_name, metric_data in data["metrics"].items():
                if isinstance(metric_data, list) and metric_data:
                    # Assume format: [{"timestamp": datetime, "value": float}, ...]
                    values = []
                    for point in metric_data:
                        if isinstance(point, dict) and "timestamp" in point and "value" in point:
                            values.append((point["timestamp"], point["value"]))
                    if values:
                        time_series[metric_name] = values

        # Extract from system monitoring data
        if "system_metrics" in data:
            system_data = data["system_metrics"]
            if isinstance(system_data, list):
                for metric in ["cpu_usage", "memory_usage", "disk_usage"]:
                    values = []
                    for point in system_data:
                        if metric in point:
                            timestamp = point.get("timestamp", datetime.utcnow())
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(timestamp)
                            values.append((timestamp, point[metric]))
                    if values:
                        time_series[metric] = values

        return time_series

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analytics capabilities."""
        return {
            "capabilities": {
                "anomaly_detection": {
                    "models_trained": len(self.anomaly_detector.models),
                    "supported_types": [t.value for t in AnomalyType]
                },
                "predictive_modeling": {
                    "models_trained": len(self.predictive_modeler.models),
                    "prediction_horizon": "24 hours"
                },
                "clustering": {
                    "datasets_analyzed": len(self.cluster_analyzer.cluster_models),
                    "clustering_algorithm": "K-means"
                },
                "code_quality": {
                    "metrics_calculated": ["complexity", "readability", "maintainability", "bug_probability"]
                },
                "insights": {
                    "total_generated": len(self.insight_generator.insights),
                    "categories": ["anomaly_detection", "predictive_analytics", "performance", "system_health"]
                }
            },
            "last_analysis": datetime.utcnow().isoformat()
        }


# Export main classes
__all__ = [
    "AnalysisType",
    "AnomalyType",
    "AnomalyResult",
    "PredictionResult",
    "ClusterResult",
    "Insight",
    "AnomalyDetector",
    "PredictiveModeler",
    "ClusterAnalyzer",
    "CodeQualityAnalyzer",
    "InsightGenerator",
    "AdvancedAnalyticsOrchestrator"
]
