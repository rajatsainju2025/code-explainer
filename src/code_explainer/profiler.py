"""
Performance profiling utilities.
"""

from typing import Dict, Any
import time


class Profiler:
    """Performance profiler for code explanation."""

    def __init__(self):
        self.metrics = {}

    def start_profiling(self, name: str):
        """Start profiling a section."""
        self.metrics[name] = {"start_time": time.time()}

    def end_profiling(self, name: str) -> Dict[str, Any]:
        """End profiling and return metrics."""
        if name in self.metrics:
            start_time = self.metrics[name]["start_time"]
            duration = time.time() - start_time
            self.metrics[name]["duration"] = duration
            return self.metrics[name]
        return {}