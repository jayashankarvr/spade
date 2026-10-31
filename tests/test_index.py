"""Tests for the FAISS index wrapper."""

import json

import numpy as np
import pytest

from spade.exceptions import IndexCorruptedError, IndexStoreError, TargetExistsError
from spade.search.index import ANNIndex


def _vectors(n, dim=9, seed=0):
    v = np.random.RandomState(seed).rand(n, dim).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestANNIndex:
    def test_search_resolves_owner_and_patch(self):
        index = ANNIndex(9)
        a, b = _vectors(10, seed=1), _vectors(5, seed=2)
        index.add("a", a)
        index.add("b", b)
        hits = index.search(b[[3]], k=1)
        assert hits.image_ids[hits.owner[0, 0]] == "b"
        assert hits.patch_idx[0, 0] == 3
        assert hits.distances[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_remove_really_deletes(self):
        index = ANNIndex(9)
        index.add("a", _vectors(10, seed=1))
        index.add("b", _vectors(5, seed=2))
        assert index.remove("a")
        assert index.size == 5
        hits = index.search(_vectors(10, seed=1), k=5)
        owners = {hits.image_ids[o] for o in hits.owner.ravel() if o >= 0}
        assert owners == {"b"}
        assert not index.remove("a")

    def test_ids_not_reused_after_remove(self):
        index = ANNIndex(9)
        index.add("a", _vectors(4, seed=1))
        index.remove("a")
        index.add("c", _vectors(3, seed=3))
        hits = index.search(_vectors(3, seed=3)[[2]], k=1)
        assert hits.image_ids[hits.owner[0, 0]] == "c" and hits.patch_idx[0, 0] == 2

    def test_duplicate_id_rejected(self):
        index = ANNIndex(9)
        index.add("a", _vectors(2))
        with pytest.raises(TargetExistsError):
            index.add("a", _vectors(2))

    def test_save_load_roundtrip(self, tmp_path):
        index = ANNIndex(9)
        index.add("a", _vectors(6, seed=1))
        index.add("b", _vectors(4, seed=2))
        index.remove("a")
        path = str(tmp_path / "idx")
        index.save(path)
        loaded = ANNIndex.load(path)
        assert loaded.size == 4 and loaded.image_ids == ["b"]
        hits = loaded.search(_vectors(4, seed=2)[[1]], k=1)
        assert hits.patch_idx[0, 0] == 1
        assert not list(tmp_path.glob(".spade-*"))  # no temp files left behind

    def test_load_rejects_mismatched_metadata(self, tmp_path):
        index = ANNIndex(9)
        index.add("a", _vectors(3))
        path = str(tmp_path / "idx")
        index.save(path)
        meta = json.loads((tmp_path / "idx.meta.json").read_text())
        meta["targets"][0][2] = 99
        (tmp_path / "idx.meta.json").write_text(json.dumps(meta))
        with pytest.raises(IndexCorruptedError):
            ANNIndex.load(path)

    def test_load_rejects_old_format(self, tmp_path):
        (tmp_path / "idx.meta.json").write_text(json.dumps({"version": 1}))
        with pytest.raises(IndexStoreError):
            ANNIndex.load(str(tmp_path / "idx"))

    def test_empty_search(self):
        hits = ANNIndex(9).search(_vectors(3), k=5)
        assert hits.distances.shape == (3, 0)

    def test_empty_target_never_owns_hits(self):
        # An empty range starts at the same id as the next image's range.
        index = ANNIndex(9)
        index.add("empty", np.zeros((0, 9), np.float32))
        index.add("full", _vectors(5, seed=4))
        hits = index.search(_vectors(5, seed=4), k=3)
        owners = {hits.image_ids[o] for o in hits.owner.ravel() if o >= 0}
        assert owners == {"full"}
        assert hits.patch_idx[2, 0] == 2


class TestIVFPQ:
    def test_exact_until_trained_then_compressed(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3000, 25)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        index = ANNIndex(25, kind="ivfpq", train_size=2000)
        index.add("a", x[:1000])
        assert not index._trained                      # still the exact buffer
        index.add("b", x[1000:3000])
        assert index._trained and index.size == 3000   # trained, nothing lost

        # Approximate: the vector's own entry must be among the k candidates
        # the engine goes on to verify, not necessarily first.
        hits = index.search(x[[1500]], k=8)
        found = {(hits.image_ids[o], int(p)) for o, p in zip(hits.owner[0], hits.patch_idx[0])}
        assert ("b", 500) in found

    def test_remove_and_roundtrip(self, tmp_path):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2500, 25)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        index = ANNIndex(25, kind="ivfpq", train_size=2000, nprobe=8)
        index.add("a", x[:1500])
        index.add("b", x[1500:])
        assert index.remove("a") and index.size == 1000
        index.save(str(tmp_path / "idx"))
        loaded = ANNIndex.load(str(tmp_path / "idx"))
        assert loaded.kind == "ivfpq" and loaded._trained and loaded.size == 1000
        hits = loaded.search(x[[1600]], k=8)
        found = {(hits.image_ids[o], int(p)) for o, p in zip(hits.owner[0], hits.patch_idx[0])}
        assert ("b", 100) in found and all(name == "b" for name, _ in found)

    def test_unknown_kind(self):
        with pytest.raises(IndexStoreError):
            ANNIndex(25, kind="hnsw")
