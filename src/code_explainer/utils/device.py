"""Device management utilities."""


def get_device() -> str:
    """Get the best available device for training/inference.

    This function is maintained for backwards compatibility.
    For new code, consider using DeviceManager directly.
    """
    try:
        from ..device_manager import device_manager
        device_capabilities = device_manager.get_optimal_device()
        return device_capabilities.device_type
    except (ImportError, AttributeError, RuntimeError):
        # Fallback to original logic if DeviceManager fails
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"