"""Device management utilities.

Small wrapper that avoids importing the global `device_manager` at module
import time to reduce startup overhead and circular import risk.
"""

import importlib
from typing import Optional


def get_device(prefer: Optional[str] = None) -> str:
    """Get the best available device for training/inference.

    Returns one of: 'cuda', 'mps', 'cpu'.

    This function imports the `device_manager` lazily to keep module import
    fast and avoid circular import problems during package initialization.
    """
    # Import absolute module path to avoid relative package resolution issues
    # when the package is imported from different contexts (tests, CLI, etc.)
    dm = importlib.import_module('code_explainer.device_manager')
    return dm.get_device_manager().get_optimal_device(prefer).device_type