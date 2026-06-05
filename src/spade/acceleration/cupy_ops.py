"""CuPy-accelerated operations for patch extraction."""

import numpy as np

from spade.exceptions import DependencyError

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def compute_entropy_batch_gpu(patches_gpu, bins: int = 8):
    """
    Vectorized entropy computation on GPU.

    Eliminates the per-patch loop in CPU version by using
    GPU-parallel histogram computation.

    Args:
        patches_gpu: CuPy array of shape (N, H, W, 3) in [0, 1]
        bins: Number of histogram bins

    Returns:
        CuPy array of shape (N,) with entropy values
    """
    if not CUPY_AVAILABLE:
        raise DependencyError("CuPy required for GPU entropy")

    n = len(patches_gpu)
    if n == 0:
        return cp.array([], dtype=cp.float32)

    # Convert to grayscale (ITU-R BT.601)
    gray = (
        0.299 * patches_gpu[:, :, :, 0]
        + 0.587 * patches_gpu[:, :, :, 1]
        + 0.114 * patches_gpu[:, :, :, 2]
    )

    # Flatten spatial dimensions: (N, H*W)
    pixels_per_patch = gray.shape[1] * gray.shape[2]
    gray_flat = gray.reshape(n, -1)

    # Quantize to bins [0, bins-1]
    quantized = cp.floor(gray_flat * bins).astype(cp.int32)
    quantized = cp.clip(quantized, 0, bins - 1)

    # Compute histogram counts for each patch
    # Use one-hot encoding approach for GPU parallelism
    entropies = cp.zeros(n, dtype=cp.float32)

    for b in range(bins):
        counts = cp.sum(quantized == b, axis=1).astype(cp.float32)
        probs = counts / pixels_per_patch
        # Shannon entropy: -sum(p * log2(p)) for p > 0
        mask = probs > 0
        entropies -= cp.where(mask, probs * cp.log2(probs), 0.0)

    return entropies


def resize_image_gpu(image_gpu, scale: float):
    """
    Resize image on GPU using CuPy.

    Args:
        image_gpu: CuPy array of shape (H, W, 3)
        scale: Scale factor (e.g., 0.5 for half size)

    Returns:
        Resized CuPy array
    """
    if not CUPY_AVAILABLE:
        raise DependencyError("CuPy required for GPU resize")

    if scale == 1.0:
        return image_gpu

    from cupyx.scipy.ndimage import zoom

    # zoom with order=1 for bilinear interpolation
    return zoom(image_gpu, (scale, scale, 1), order=1)
