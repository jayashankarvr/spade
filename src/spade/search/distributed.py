"""Distributed sharded index for large-scale search."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from spade.search.index import ANNIndex, SearchResult
from spade.exceptions import DependencyError, ConfigurationError, IndexError as SPADEIndexError


@dataclass
class ShardInfo:
    """Information about a single shard."""

    shard_id: int
    size: int


class ShardedIndex:
    """
    Sharded index for distributed search.

    Supports two modes:
    1. Local mode: Multiple index files on same machine (parallel search)
    2. Remote mode: Shards on different machines via HTTP

    Index is sharded by image_id hash for even distribution.
    All descriptors from the same image go to the same shard.
    """

    def __init__(
        self,
        dim: int = 256,
        num_shards: int = 4,
        mode: str = "local",
        use_gpu: bool = False,
    ):
        """
        Args:
            dim: Descriptor dimensionality
            num_shards: Number of shards to create
            mode: "local" for in-memory shards, "remote" for HTTP-based
            use_gpu: Use GPU for each shard (local mode only)
        """
        self.dim = dim
        self.num_shards = num_shards
        self.mode = mode
        self.use_gpu = use_gpu

        # Local shards (in-memory)
        self.shards: List[ANNIndex] = []

        # Remote shard endpoints
        self.remote_endpoints: List[str] = []

        if mode == "local":
            self.shards = [
                ANNIndex(dim=dim, use_gpu=use_gpu) for _ in range(num_shards)
            ]

    def _get_shard_id(self, image_id: str) -> int:
        """Compute shard index from image_id hash."""
        return hash(image_id) % self.num_shards

    def add(
        self,
        descriptors: np.ndarray,
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """
        Add descriptors to appropriate shard based on image_id in metadata.

        All descriptors must have the same image_id.

        Args:
            descriptors: Array of shape (N, dim)
            metadata_list: Metadata for each descriptor (must contain image_id)
        """
        if not metadata_list:
            return

        # Get image_id from first metadata entry
        image_id = metadata_list[0].get("image_id", "")
        shard_id = self._get_shard_id(image_id)

        if self.mode == "local":
            self.shards[shard_id].add(descriptors, metadata_list)
        else:
            self._remote_add(shard_id, descriptors, metadata_list)

    def search(
        self,
        query: np.ndarray,
        k: int = 100,
        lsh_candidates: int = 1000,
    ) -> List[SearchResult]:
        """
        Search all shards and merge results.

        Uses parallel search across shards for speed.

        Args:
            query: Query descriptor
            k: Number of results to return
            lsh_candidates: Ignored (for API compatibility)

        Returns:
            Top-k results across all shards, sorted by distance
        """
        if self.mode == "local":
            return self._search_local(query, k)
        else:
            return self._search_remote(query, k)

    def _search_local(self, query: np.ndarray, k: int) -> List[SearchResult]:
        """Search local shards in parallel."""

        def search_shard(shard: ANNIndex) -> List[SearchResult]:
            if shard.size == 0:
                return []
            return shard.search(query, k)

        all_results: List[SearchResult] = []

        # Use thread pool for parallel search
        with ThreadPoolExecutor(max_workers=self.num_shards) as executor:
            futures = [executor.submit(search_shard, s) for s in self.shards]
            for future in futures:
                all_results.extend(future.result())

        # Sort by distance and take top-k
        all_results.sort(key=lambda r: r.distance)
        return all_results[:k]

    def _search_remote(self, query: np.ndarray, k: int) -> List[SearchResult]:
        """Search remote shards via HTTP."""
        try:
            import requests
        except ImportError:
            raise DependencyError("requests required for remote mode: pip install requests")

        def search_remote_shard(endpoint: str) -> List[SearchResult]:
            try:
                response = requests.post(
                    f"{endpoint}/search",
                    json={"query": query.tolist(), "k": k},
                    timeout=10,
                )
                if response.ok:
                    data = response.json()
                    return [
                        SearchResult(
                            index=r["index"],
                            distance=r["distance"],
                            metadata=r["metadata"],
                        )
                        for r in data.get("results", [])
                    ]
            except Exception:
                pass
            return []

        all_results: List[SearchResult] = []

        with ThreadPoolExecutor(max_workers=len(self.remote_endpoints)) as executor:
            futures = [
                executor.submit(search_remote_shard, ep) for ep in self.remote_endpoints
            ]
            for future in futures:
                all_results.extend(future.result())

        all_results.sort(key=lambda r: r.distance)
        return all_results[:k]

    def _remote_add(
        self, shard_id: int, descriptors: np.ndarray, metadata_list: List
    ) -> None:
        """Add to remote shard via HTTP."""
        try:
            import requests
        except ImportError:
            raise DependencyError("requests required for remote mode: pip install requests")

        if shard_id >= len(self.remote_endpoints):
            raise ConfigurationError(f"Shard {shard_id} not configured")

        endpoint = self.remote_endpoints[shard_id]
        requests.post(
            f"{endpoint}/add",
            json={"descriptors": descriptors.tolist(), "metadata": metadata_list},
            timeout=30,
        )

    def save(self, base_path: str) -> None:
        """
        Save all local shards to disk.

        Creates files: {base_path}.shard{i}.index, {base_path}.shard{i}.meta.json
        Plus manifest: {base_path}.shards.json
        """
        if self.mode != "local":
            raise SPADEIndexError("Cannot save remote shards locally")

        for i, shard in enumerate(self.shards):
            shard.save(f"{base_path}.shard{i}")

        # Save manifest
        manifest = {
            "num_shards": self.num_shards,
            "dim": self.dim,
            "shard_sizes": [s.size for s in self.shards],
        }
        with open(f"{base_path}.shards.json", "w") as f:
            json.dump(manifest, f)

    def load(self, base_path: str) -> None:
        """
        Load all local shards from disk.

        Reads manifest and all shard files.
        """
        with open(f"{base_path}.shards.json") as f:
            manifest = json.load(f)

        self.num_shards = manifest["num_shards"]
        self.dim = manifest["dim"]
        self.shards = []

        for i in range(self.num_shards):
            shard = ANNIndex(dim=self.dim, use_gpu=self.use_gpu)
            shard.load(f"{base_path}.shard{i}")
            self.shards.append(shard)

    def configure_remote(self, endpoints: List[str]) -> None:
        """
        Configure remote shard endpoints for remote mode.

        Args:
            endpoints: List of base URLs for each shard (e.g., ["http://host1:8000", ...])
        """
        if len(endpoints) != self.num_shards:
            raise ConfigurationError(
                f"Expected {self.num_shards} endpoints, got {len(endpoints)}"
            )
        self.remote_endpoints = endpoints
        self.mode = "remote"

    @property
    def size(self) -> int:
        """Total vectors across all shards."""
        if self.mode == "local":
            return sum(s.size for s in self.shards)
        return 0  # Unknown for remote

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """Combined metadata from all shards (local mode only)."""
        if self.mode != "local":
            return []
        result: List[Dict[str, Any]] = []
        for shard in self.shards:
            result.extend(shard.metadata)
        return result

    def get_shard_info(self) -> List[ShardInfo]:
        """Get information about all shards."""
        if self.mode == "local":
            return [
                ShardInfo(shard_id=i, size=s.size) for i, s in enumerate(self.shards)
            ]
        return [ShardInfo(shard_id=i, size=0) for i in range(self.num_shards)]
