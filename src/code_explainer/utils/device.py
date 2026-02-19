"""Device management utilities."""

from ..device_manager import device_manager


def get_device() -> str:
    """Get the best available device for training/inference.

    Returns one of: 'cuda', 'mps', 'cpu'.

    This is a convenience wrapper around DeviceManager.
    For new code, consider using DeviceManager directly.
    """
    return device_manager.get_optimal_device().device_type