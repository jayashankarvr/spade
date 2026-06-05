"""Backend abstraction for CPU/GPU array operations."""

from dataclasses import dataclass
from typing import Any
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


@dataclass
class Backend:
    """Unified interface for array operations on CPU or GPU."""

    name: str
    xp: Any  # numpy or cupy module

    def to_device(self, arr: np.ndarray) -> Any:
        """Move array to device (GPU if cupy, else no-op)."""
        if self.name == "cupy":
            return cp.asarray(arr)
        return arr

    def to_host(self, arr: Any) -> np.ndarray:
        """Move array to CPU."""
        if self.name == "cupy":
            return cp.asnumpy(arr)
        return arr

    def asarray(self, arr: Any) -> Any:
        """Ensure array is on this backend."""
        return self.xp.asarray(arr)

    def zeros(self, shape, dtype=np.float32) -> Any:
        """Create zero array on device."""
        return self.xp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=np.float32) -> Any:
        """Create ones array on device."""
        return self.xp.ones(shape, dtype=dtype)


def get_backend(use_gpu: bool = False) -> Backend:
    """
    Get compute backend.

    Args:
        use_gpu: Whether to use GPU if available

    Returns:
        Backend instance (cupy if GPU requested and available, else numpy)
    """
    if use_gpu and CUPY_AVAILABLE:
        return Backend(name="cupy", xp=cp)
    return Backend(name="numpy", xp=np)
