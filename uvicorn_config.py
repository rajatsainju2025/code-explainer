"""Production Uvicorn configuration for better performance."""

import multiprocessing


def workers() -> int:
    cpu = multiprocessing.cpu_count()
    # Use min(8, CPUs*2) workers for IO-bound FastAPI
    return max(2, min(8, cpu * 2))


log_level = "info"
access_log = False
timeout_keep_alive = 20
backlog = 2048
limit_concurrency = 100
