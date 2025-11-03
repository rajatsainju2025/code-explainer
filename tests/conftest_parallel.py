"""Parallel test execution configuration and utilities."""

import pytest
from typing import List


# Mark slow tests to allow filtering
def pytest_configure(config):
    """Configure pytest with parallel execution settings."""
    config.addinivalue_line(
        "markers", "serial: marks tests as non-parallelizable"
    )


def pytest_collection_modifyitems(config, items: List[pytest.Item]) -> None:
    """Modify test collection to support parallel execution.
    
    Marks tests that use shared resources as serial to prevent race conditions.
    """
    serial_markers = {"serial", "integration"}
    
    for item in items:
        # Mark integration and I/O heavy tests as serial
        if any(marker in item.keywords for marker in serial_markers):
            item.add_marker(pytest.mark.serial)
