"""Exact nearest-neighbour index over patch descriptors (FAISS)."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from spade.exceptions import (
    DependencyError,
    IndexCorruptedError,
    IndexStoreError,
    TargetExistsError,
)

try:
    import faiss
except ImportError:  # pragma: no cover - faiss is a hard dependency
    faiss = None

_FORMAT_VERSION = 2
_SEARCH_CHUNK = 8192


@dataclass
class SearchHits:
    """k-NN results for a batch of queries.

    ``owner[i, j]`` indexes into ``image_ids`` (-1 = no hit); ``patch_idx[i, j]``
    is the patch's position within that image's indexed patches. Distances are
    squared L2.
    """
    distances: np.ndarray   # (N, k) float32
    owner: np.ndarray       # (N, k) int64
    patch_idx: np.ndarray   # (N, k) int64
    image_ids: List[str]


class ANNIndex:
    """Flat L2 index where every image owns a contiguous block of vector ids.

    Ids are never reused, so removing an image is an exact ``remove_ids`` on its
    range and nothing stale is ever returned by a search.
    """

    def __init__(self, dim: int):
        if faiss is None:
            raise DependencyError("faiss-cpu required: pip install faiss-cpu")
        self.dim = dim
        self._lock = threading.RLock()
        self._index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        self._next_id = 0
        self._ranges: "OrderedDict[str, Tuple[int, int]]" = OrderedDict()  # image_id -> (start, count)

    # ------------------------------------------------------------------

    def add(self, image_id: str, descriptors: np.ndarray) -> None:
        with self._lock:
            if image_id in self._ranges:
                raise TargetExistsError(f"'{image_id}' is already indexed")
            descriptors = np.ascontiguousarray(descriptors, dtype=np.float32)
            if descriptors.ndim != 2 or descriptors.shape[1] != self.dim:
                raise IndexStoreError(f"expected (N, {self.dim}) descriptors, got {descriptors.shape}")
            n = len(descriptors)
            ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
            if n:
                self._index.add_with_ids(descriptors, ids)
            self._ranges[image_id] = (self._next_id, n)
            self._next_id += n

    def remove(self, image_id: str) -> bool:
        with self._lock:
            entry = self._ranges.pop(image_id, None)
            if entry is None:
                return False
            start, count = entry
            if count:
                self._index.remove_ids(faiss.IDSelectorRange(start, start + count))
            return True

    def search(self, queries: np.ndarray, k: int) -> SearchHits:
        with self._lock:
            queries = np.ascontiguousarray(queries, dtype=np.float32).reshape(-1, self.dim)
            n = len(queries)
            image_ids = list(self._ranges)
            k = min(k, self._index.ntotal)
            if n == 0 or k == 0:
                empty_f = np.zeros((n, 0), dtype=np.float32)
                empty_i = np.zeros((n, 0), dtype=np.int64)
                return SearchHits(empty_f, empty_i, empty_i.copy(), image_ids)

            distances = np.empty((n, k), dtype=np.float32)
            ids = np.empty((n, k), dtype=np.int64)
            for start in range(0, n, _SEARCH_CHUNK):
                stop = min(start + _SEARCH_CHUNK, n)
                distances[start:stop], ids[start:stop] = self._index.search(queries[start:stop], k)

            owner, patch_idx = self._resolve(ids, image_ids)
            return SearchHits(distances, owner, patch_idx, image_ids)

    def _resolve(self, ids: np.ndarray, image_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Map vector ids to (owner index, patch index); -1 where there's no hit."""
        # Ranges are handed out in insertion order, so starts are already sorted.
        # Empty ranges share their start with the next image and are left out,
        # otherwise they could be credited with that image's hits.
        owners = [i for i, image_id in enumerate(image_ids) if self._ranges[image_id][1] > 0]
        if not owners:
            missing = np.full(ids.shape, -1, dtype=np.int64)
            return missing, missing.copy()
        owners = np.array(owners, dtype=np.int64)
        starts = np.array([self._ranges[image_ids[i]][0] for i in owners], dtype=np.int64)
        pos = np.searchsorted(starts, ids, side="right") - 1
        valid = (ids >= 0) & (pos >= 0)
        pos = np.clip(pos, 0, None)
        owner = np.where(valid, owners[pos], -1)
        patch_idx = np.where(valid, ids - starts[pos], -1)
        return owner, patch_idx

    # ------------------------------------------------------------------

    def count(self, image_id: str) -> int:
        entry = self._ranges.get(image_id)
        return entry[1] if entry else 0

    @property
    def image_ids(self) -> List[str]:
        return list(self._ranges)

    @property
    def size(self) -> int:
        with self._lock:
            return int(self._index.ntotal)

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write ``{path}.index`` and ``{path}.meta.json``, each atomically."""
        with self._lock:
            meta = {
                "version": _FORMAT_VERSION,
                "dim": self.dim,
                "next_id": self._next_id,
                "targets": [[image_id, start, count] for image_id, (start, count) in self._ranges.items()],
            }
            _atomic_write(f"{path}.index", lambda tmp: faiss.write_index(self._index, tmp))
            _atomic_write(f"{path}.meta.json", lambda tmp: _write_json(tmp, meta))

    @classmethod
    def load(cls, path: str) -> "ANNIndex":
        try:
            with open(f"{path}.meta.json") as f:
                meta = json.load(f)
        except FileNotFoundError as e:
            raise IndexStoreError(f"no index metadata at {path}.meta.json") from e
        if meta.get("version") != _FORMAT_VERSION:
            raise IndexStoreError(
                f"unsupported index format {meta.get('version')!r}; re-index with this SPADE version"
            )
        index = cls(int(meta["dim"]))
        index._index = faiss.read_index(f"{path}.index")
        index._next_id = int(meta["next_id"])
        index._ranges = OrderedDict((t[0], (int(t[1]), int(t[2]))) for t in meta["targets"])
        expected = sum(count for _, count in index._ranges.values())
        if index._index.ntotal != expected or index._index.d != index.dim:
            raise IndexCorruptedError(
                f"index holds {index._index.ntotal} vectors of dim {index._index.d}, "
                f"metadata expects {expected} of dim {index.dim}"
            )
        return index


def _write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)


def _atomic_write(path: str, writer) -> None:
    """Write via a temp file in the same directory, then os.replace (atomic on POSIX and Windows)."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".spade-", suffix=".tmp")
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
