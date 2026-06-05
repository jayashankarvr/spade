"""LSH-based pre-filtering for fast approximate search."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
import numpy as np

from spade.exceptions import DependencyError

try:
    from datasketch import MinHashLSH, MinHash
    DATASKETCH_AVAILABLE = True
except ImportError:
    MinHashLSH = None
    MinHash = None
    DATASKETCH_AVAILABLE = False


@dataclass
class LSHConfig:
    """Configuration for LSH pre-filtering."""
    num_perm: int = 128       # Number of permutations for MinHash
    threshold: float = 0.3    # Jaccard similarity threshold (0-1)


class LSHPreFilter:
    """
    LSH-based pre-filter for fast candidate selection.

    Uses MinHash LSH to quickly identify candidate vectors before
    exhaustive search. Converts float descriptors to binary features
    for MinHash compatibility.
    """

    def __init__(self, config: Optional[LSHConfig] = None):
        """
        Args:
            config: LSH configuration
        """
        if not DATASKETCH_AVAILABLE:
            raise DependencyError("datasketch required: pip install datasketch")

        self.config = config or LSHConfig()
        self.lsh = MinHashLSH(
            threshold=self.config.threshold,
            num_perm=self.config.num_perm,
        )
        self._index_to_key: Dict[int, str] = {}
        self._key_to_index: Dict[str, int] = {}
        self._count = 0

    def add(self, descriptors: np.ndarray, start_index: int = 0) -> None:
        """
        Add descriptors to the LSH index.

        Args:
            descriptors: Array of shape (N, dim)
            start_index: Starting index for these descriptors
        """
        for i, desc in enumerate(descriptors):
            idx = start_index + i
            key = f"v{idx}"
            minhash = self._descriptor_to_minhash(desc)
            self.lsh.insert(key, minhash)
            self._index_to_key[idx] = key
            self._key_to_index[key] = idx
            self._count += 1

    def query(self, descriptor: np.ndarray, max_candidates: int = 1000) -> List[int]:
        """
        Query for candidate indices.

        Args:
            descriptor: Query descriptor, shape (dim,)
            max_candidates: Maximum candidates to return

        Returns:
            List of candidate indices (ordered by insertion)
        """
        minhash = self._descriptor_to_minhash(descriptor)
        candidates = self.lsh.query(minhash)

        indices = []
        for key in candidates[:max_candidates]:
            if key in self._key_to_index:
                indices.append(self._key_to_index[key])

        return indices

    def _descriptor_to_minhash(self, descriptor: np.ndarray) -> "MinHash":
        """
        Convert float descriptor to MinHash.

        Uses quantization to create binary features from the descriptor.
        """
        mh = MinHash(num_perm=self.config.num_perm)

        # Quantize descriptor to create discrete features
        # Use sign of each dimension as binary feature
        binary = (descriptor > 0).astype(np.uint8)

        # Also add quantized magnitude features
        magnitudes = np.abs(descriptor)
        quantized = np.digitize(magnitudes, bins=[0.1, 0.3, 0.5, 0.7, 0.9])

        # Create feature strings for hashing
        for i, (b, q) in enumerate(zip(binary, quantized)):
            # Feature: dimension index + sign + magnitude bin
            feature = f"{i}:{b}:{q}"
            mh.update(feature.encode('utf8'))

        return mh

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._count


class HybridIndex:
    """
    Hybrid index combining LSH pre-filtering with FAISS search.

    Uses LSH to quickly identify candidates, then computes exact distances
    for ranking. FAISS stores descriptors; LSH provides fast candidate filtering.
    """

    def __init__(
        self,
        dim: int = 256,
        lsh_config: Optional[LSHConfig] = None,
        use_lsh: bool = True,
    ):
        """
        Args:
            dim: Descriptor dimensionality
            lsh_config: LSH configuration
            use_lsh: Whether to use LSH pre-filtering
        """
        from spade.search.index import ANNIndex

        self.dim = dim
        self.ann_index = ANNIndex(dim=dim)
        self.use_lsh = use_lsh and DATASKETCH_AVAILABLE
        self._lsh_config = lsh_config

        if self.use_lsh:
            self.lsh = LSHPreFilter(lsh_config)
        else:
            self.lsh = None

    def add(
        self,
        descriptors: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add descriptors to both LSH and FAISS indexes.

        Args:
            descriptors: Array of shape (N, dim)
            metadata_list: Optional metadata for each descriptor
        """
        start_idx = self.ann_index.size

        # Add to FAISS (stores descriptors internally)
        self.ann_index.add(descriptors, metadata_list)

        # Add to LSH for fast candidate filtering
        if self.use_lsh and self.lsh is not None:
            self.lsh.add(descriptors, start_idx)

    def search(
        self,
        query: np.ndarray,
        k: int = 100,
        lsh_candidates: int = 1000,
    ) -> List:
        """
        Search for k nearest neighbors.

        If LSH is enabled, first filters candidates then searches within them.

        Args:
            query: Query descriptor
            k: Number of neighbors
            lsh_candidates: Max candidates from LSH

        Returns:
            List of SearchResult objects
        """
        from spade.search.index import SearchResult

        # Direct FAISS search if LSH disabled or small index
        if not self.use_lsh or self.lsh is None or self.ann_index.size < lsh_candidates:
            return self.ann_index.search(query, k)

        # LSH pre-filtering - returns list of indices
        candidate_indices = self.lsh.query(query, lsh_candidates)

        if not candidate_indices:
            # Fall back to full search if LSH returns nothing
            return self.ann_index.search(query, k)

        # Reconstruct candidate descriptors from FAISS
        # Note: IndexFlatL2 supports reconstruct
        query = query.reshape(1, -1).astype(np.float32)

        # Build candidate array maintaining index correspondence
        valid_candidates = []
        valid_indices = []
        for idx in candidate_indices:
            if idx < self.ann_index.size:
                try:
                    desc = self.ann_index.index.reconstruct(idx)
                    valid_candidates.append(desc)
                    valid_indices.append(idx)
                except RuntimeError:
                    # reconstruct not supported, fall back to FAISS search
                    return self.ann_index.search(query.flatten(), k)

        if not valid_candidates:
            return self.ann_index.search(query.flatten(), k)

        candidate_array = np.array(valid_candidates, dtype=np.float32)

        # Compute L2 distances for candidates
        distances = np.sum((candidate_array - query) ** 2, axis=1)

        # Sort and get top k
        sorted_order = np.argsort(distances)[:k]

        results = []
        for order_idx in sorted_order:
            original_idx = valid_indices[order_idx]
            results.append(SearchResult(
                index=original_idx,
                distance=float(distances[order_idx]),
                metadata=self.ann_index.metadata[original_idx] if original_idx < len(self.ann_index.metadata) else {},
            ))

        return results

    def save(self, path: str) -> None:
        """Save hybrid index."""
        self.ann_index.save(path)

    def load(self, path: str) -> None:
        """
        Load hybrid index.

        Note: LSH index is rebuilt from FAISS data for consistency.
        """
        self.ann_index.load(path)

        # Rebuild LSH from FAISS descriptors
        if self.use_lsh:
            self.lsh = LSHPreFilter(self._lsh_config)
            # Reconstruct all descriptors and add to LSH
            for i in range(self.ann_index.size):
                try:
                    desc = self.ann_index.index.reconstruct(i)
                    self.lsh.add(desc.reshape(1, -1), start_index=i)
                except RuntimeError:
                    # If reconstruct fails, disable LSH
                    self.lsh = None
                    self.use_lsh = False
                    break

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.ann_index.size

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """Access metadata from underlying FAISS index."""
        return self.ann_index.metadata
