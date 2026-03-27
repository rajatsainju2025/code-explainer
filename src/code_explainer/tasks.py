"""Celery task queue configuration for async processing and scaling."""

import os
import logging

from celery import Celery, Task
from celery.schedules import crontab

# Use environment variables directly for broker config to avoid import-time
# failures from config_manager requiring API_KEY and other mandatory fields.
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "code_explainer",
    broker=_REDIS_URL,
    backend=_REDIS_URL,
)

# Configuration
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task time limits (hard limit 5 minutes)
    task_time_limit=300,
    # Soft limit 4 minutes (graceful shutdown)
    task_soft_time_limit=240,
    # Task result expires after 1 hour
    result_expires=3600,
    # Retry on failure
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Worker pool settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
)


logger = logging.getLogger("code-explainer.celery")


class CodeExplainerTask(Task):
    """Base task class with custom error handling."""

    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        logger.warning(
            f"Task {task_id} retrying: {str(exc)}",
            extra={"task_id": task_id, "exception": str(exc)}
        )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails permanently."""
        logger.error(
            f"Task {task_id} failed: {str(exc)}",
            extra={"task_id": task_id, "exception": str(exc)}
        )


@app.task(base=CodeExplainerTask, bind=True)
def async_code_explanation(
    self,
    code: str,
    model_name: str = None,
    use_cache: bool = True,
    strategy: str = "vanilla"
) -> dict:
    """Async task for code explanation inference.

    Args:
        self: Celery task instance
        code: Source code to explain
        model_name: Model to use (default: settings.model_name)
        use_cache: Whether to use cache
        strategy: Explanation strategy (default: vanilla)

    Returns:
        Explanation result
    """
    model_name = model_name or os.getenv("MODEL_NAME", "codet5-base")
    logger.info(
        f"Starting async explanation for {model_name}",
        extra={"task_id": self.request.id, "code_length": len(code)}
    )
    
    try:
        # Import here to avoid circular dependency and heavy startup
        from .model.core import CodeExplainer
        
        explainer = CodeExplainer()
        explanation = explainer.explain_code(code, strategy=strategy)
        
        return {
            "task_id": self.request.id,
            "status": "completed",
            "explanation": explanation,
            "model_name": model_name
        }
    except Exception as exc:
        logger.error("Async explanation failed: %s", exc)
        raise


@app.task(base=CodeExplainerTask)
def cleanup_expired_cache():
    """Periodic task to clean up expired cache entries."""
    from .database import DatabaseManager
    logger.info("Running cache cleanup task")
    
    try:
        db = DatabaseManager()
        removed = db.cleanup_expired_cache()
        logger.info("Removed %d expired cache entries", removed)
        return {"removed": removed}
    except Exception as exc:
        logger.error("Cache cleanup failed: %s", exc)
        raise


@app.task(base=CodeExplainerTask)
def generate_metrics_report():
    """Periodic task to generate performance metrics report."""
    from .database import DatabaseManager
    logger.info("Generating metrics report")
    
    try:
        db = DatabaseManager()
        stats = db.get_request_stats(hours=24)
        logger.info("Generated metrics for %s requests", stats.get('total_requests', 0))
        return stats
    except Exception as exc:
        logger.error("Metrics report failed: %s", exc)
        raise


# Periodic task schedule
app.conf.beat_schedule = {
    "cleanup-cache-every-hour": {
        "task": "code_explainer.tasks.cleanup_expired_cache",
        "schedule": crontab(minute=0),  # Run every hour
    },
    "metrics-report-every-6-hours": {
        "task": "code_explainer.tasks.generate_metrics_report",
        "schedule": crontab(minute=0, hour="*/6"),  # Run every 6 hours
    },
}


if __name__ == "__main__":
    app.start()
