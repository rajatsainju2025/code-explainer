"""Prometheus metrics exporter for API."""

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response


class PrometheusMetrics:
    """Prometheus metrics exporter."""

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


# Global instance
prometheus_metrics = PrometheusMetrics()
