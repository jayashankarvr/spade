"""FAISS-based approximate nearest neighbor search."""

import json
import threading
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from spade.logging_config import get_logger
from spade.exceptions import IndexStoreError as SPADEIndexError, IndexCorruptedError, DependencyError

logger = get_logger("search.index")

try:
    import faiss
    FAISS_AVAILABLE = True
    # Check for GPU support
    try:
        _res = faiss.StandardGpuResources()
        FAISS_GPU_AVAILABLE = True
        del _res
    except (AttributeError, RuntimeError):
        FAISS_GPU_AVAILABLE = False
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False


@dataclass
class SearchResult:
    """Single search result with metadata."""
    index: int
    distance: float
    metadata: Dict[str, Any]


class ANNIndex:
    """
    Approximate nearest neighbor index using FAISS.
    Stores descriptors and associated metadata for retrieval.
    """

    def __init__(self, dim: int = 128, use_gpu: bool = False):
        """
        Args:
            dim: Descriptor dimensionality
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
        """
        if not FAISS_AVAILABLE:
            raise DependencyError("faiss-cpu required: pip install faiss-cpu")

        self.dim = dim
        self.use_gpu = use_gpu
        self._gpu_resources = None
        self._lock = threading.RLock()  # Reentrant lock for thread safety

        # Create base CPU index
        cpu_index = faiss.IndexFlatL2(dim)

        # Move to GPU if requested and available
        if use_gpu:
            if FAISS_GPU_AVAILABLE:
                self._gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
            else:
                logger.warning("FAISS-GPU not available, falling back to CPU")
                self.use_gpu = False
                self.index = cpu_index
        else:
            self.index = cpu_index

        self.metadata: List[Dict[str, Any]] = []

    def add(
        self,
        descriptors: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add descriptors to the index (thread-safe).

        Args:
            descriptors: Array of shape (N, dim)
            metadata_list: Optional metadata for each descriptor
        """
        with self._lock:
            descriptors = np.ascontiguousarray(descriptors.astype(np.float32))
            n = len(descriptors)

            if metadata_list is None:
                metadata_list = [{} for _ in range(n)]

            if len(metadata_list) != n:
                raise SPADEIndexError(f"Metadata length {len(metadata_list)} != descriptors {n}")

            self.index.add(descriptors)
            self.metadata.extend(metadata_list)

    def search(self, query: np.ndarray, k: int = 100, lsh_candidates: int = 1000) -> List[SearchResult]:
        """
        Search for k nearest neighbors (thread-safe).

        Args:
            query: Query descriptor, shape (dim,) or (1, dim)
            k: Number of neighbors
            lsh_candidates: Ignored (for API compatibility with HybridIndex)

        Returns:
            List of SearchResult objects
        """
        with self._lock:
            # lsh_candidates is ignored - only used by HybridIndex
            if query.ndim == 1:
                query = query.reshape(1, -1)
            query = np.ascontiguousarray(query.astype(np.float32))

            k = min(k, self.index.ntotal)
            if k == 0:
                return []

            distances, indices = self.index.search(query, k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:
                    results.append(SearchResult(
                        index=int(idx),
                        distance=float(dist),
                        metadata=self.metadata[idx] if idx < len(self.metadata) else {},
                    ))
            return results

    def save(self, path: str) -> None:
        """
        Save index and metadata to files atomically (thread-safe).

        Uses FAISS binary format for index and JSON for metadata (secure).
        Writes to temporary files first, then atomically renames to prevent corruption.
        """
        with self._lock:
            import tempfile
            import shutil

            index_path = f"{path}.index"
            metadata_path = f"{path}.meta.json"

            # Write to temporary files first
            try:
                # Save index to temporary file
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.index') as tmp_index:
                    tmp_index_path = tmp_index.name

                if self.use_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, tmp_index_path)
                else:
                    faiss.write_index(self.index, tmp_index_path)

                # Save metadata to temporary file
                json_safe_metadata = []
                for meta in self.metadata:
                    safe_meta = {}
                    for k, v in meta.items():
                        if isinstance(v, tuple):
                            safe_meta[k] = [int(x) if hasattr(x, 'item') else x for x in v]
                        elif hasattr(v, 'item'):  # numpy scalar
                            safe_meta[k] = v.item()
                        else:
                            safe_meta[k] = v
                    json_safe_metadata.append(safe_meta)

                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.meta.json') as tmp_meta:
                    tmp_meta_path = tmp_meta.name
                    json.dump(json_safe_metadata, tmp_meta)

                # Atomic rename (POSIX systems guarantee atomicity)
                # On Windows, need to remove target first
                import os
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)

                shutil.move(tmp_index_path, index_path)
                shutil.move(tmp_meta_path, metadata_path)

            except Exception as e:
                # Clean up temp files on failure
                try:
                    if 'tmp_index_path' in locals() and os.path.exists(tmp_index_path):
                        os.remove(tmp_index_path)
                    if 'tmp_meta_path' in locals() and os.path.exists(tmp_meta_path):
                        os.remove(tmp_meta_path)
                except:
                    pass
                raise SPADEIndexError(f"Failed to save index atomically: {str(e)}") from e

    def load(self, path: str) -> None:
        """
        Load index and metadata from files (thread-safe).

        Supports both legacy .npy format and new JSON format.
        """
        with self._lock:
            cpu_index = faiss.read_index(f"{path}.index")

            # Move to GPU if originally configured for GPU
            if self.use_gpu and FAISS_GPU_AVAILABLE:
                if self._gpu_resources is None:
                    self._gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)
            else:
                self.index = cpu_index

            # Try JSON first (secure), fall back to npy (legacy)
            json_path = f"{path}.meta.json"
            npy_path = f"{path}.meta.npy"

            try:
                with open(json_path, 'r') as f:
                    self.metadata = json.load(f)
                # Convert coord lists back to tuples
                for meta in self.metadata:
                    if 'coord' in meta and isinstance(meta['coord'], list):
                        meta['coord'] = tuple(meta['coord'])
            except FileNotFoundError:
                # Legacy format - load with pickle but warn
                logger.warning(
                    "Loading legacy .npy metadata format. "
                    "Re-save index to use secure JSON format."
                )
                logger.warning(
                    "SECURITY WARNING: Legacy .npy format uses pickle which can execute "
                    "arbitrary code. Only load indexes from trusted sources!"
                )
                self.metadata = np.load(npy_path, allow_pickle=True).tolist()

            # Validate that index size matches metadata size
            if self.index.ntotal != len(self.metadata):
                raise IndexCorruptedError(
                    f"Index corruption detected: FAISS index has {self.index.ntotal} vectors "
                    f"but metadata has {len(self.metadata)} entries. Index may be corrupted."
                )

    @property
    def size(self) -> int:
        """Number of vectors in the index (thread-safe)."""
        with self._lock:
            return self.index.ntotal
