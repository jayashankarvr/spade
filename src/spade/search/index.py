"""Nearest-neighbour index over patch descriptors (FAISS).

Two storage kinds, same interface:

* ``flat``  - exact search, 4 bytes per descriptor value. The default.
* ``ivfpq`` - product quantization (about one byte per 5 values) plus an
  inverted file, so a search only visits ``nprobe`` of the cells. Needs
  training, so vectors sit in an exact buffer until ``train_size`` of them
  have arrived; small collections never leave that exact mode.

Only candidate generation is approximate; every candidate is still verified
against real pixels, so a lossy index can miss matches but not invent them.
"""

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


KINDS = ("flat", "ivfpq")


class ANNIndex:
    """L2 index where every image owns a contiguous block of vector ids.

    Ids are never reused, so removing an image is an exact ``remove_ids`` on its
    range and nothing stale is ever returned by a search.
    """

    def __init__(self, dim: int, kind: str = "flat", train_size: int = 50_000, nprobe: int = 16):
        if faiss is None:
            raise DependencyError("faiss-cpu required: pip install faiss-cpu")
        if kind not in KINDS:
            raise IndexStoreError(f"index kind must be one of {KINDS}, got {kind!r}")
        self.dim = dim
        self.kind = kind
        self.train_size = train_size
        self.nprobe = nprobe
        self._lock = threading.RLock()
        self._trained = kind != "ivfpq"
        self._index = faiss.IndexIDMap2(faiss.IndexFlatL2(dim))
        self._next_id = 0
        self._ranges: "OrderedDict[str, Tuple[int, int]]" = OrderedDict()  # image_id -> (start, count)

    def _train_ivfpq(self) -> None:
        """Swap the exact buffer for a trained IVF-PQ index holding the same vectors."""
        ids = np.concatenate([np.arange(s, s + c, dtype=np.int64) for s, c in self._ranges.values() if c])
        vectors = self._index.reconstruct_batch(ids)
        nlist = int(np.clip(4 * np.sqrt(len(vectors)), 16, 4096))
        # About five descriptor values per sub-quantizer, one byte each.
        m = max(d for d in range(1, max(2, self.dim // 4 + 1)) if self.dim % d == 0)
        ivf = faiss.IndexIVFPQ(faiss.IndexFlatL2(self.dim), self.dim, nlist, m, 8)
        ivf.train(vectors)
        ivf.nprobe = self.nprobe
        # IVF keeps our ids itself. Wrapping it in IndexIDMap2 would break
        # deletes: the wrapper assumes the inner index renumbers after a
        # removal, which IVF doesn't. The hash-table direct map keeps
        # remove_ids and reconstruct working with arbitrary ids.
        ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
        ivf.add_with_ids(vectors, ids)
        self._index, self._trained = ivf, True

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
            if not self._trained and self._index.ntotal >= self.train_size:
                self._train_ivfpq()

    def remove(self, image_id: str) -> bool:
        with self._lock:
            entry = self._ranges.pop(image_id, None)
            if entry is None:
                return False
            start, count = entry
            if count:
                if self.kind == "ivfpq" and self._trained:
                    # The IVF hash-table direct map only removes by explicit id list.
                    self._index.remove_ids(np.arange(start, start + count, dtype=np.int64))
                else:
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
                "kind": self.kind,
                "trained": self._trained,
                "train_size": self.train_size,
                "nprobe": self.nprobe,
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
        index = cls(int(meta["dim"]), kind=meta.get("kind", "flat"),
                    train_size=int(meta.get("train_size", 50_000)), nprobe=int(meta.get("nprobe", 16)))
        index._index = faiss.read_index(f"{path}.index")
        index._trained = bool(meta.get("trained", True))
        if index.kind == "ivfpq" and index._trained:
            faiss.extract_index_ivf(index._index).nprobe = index.nprobe
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
