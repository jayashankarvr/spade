"""Tests for the sharded distributed index."""

import hashlib

from spade.search.distributed import ShardedIndex


class TestShardAssignment:
    def test_shard_id_is_stable_across_instances(self):
        """Regression: shard assignment must be deterministic. The built-in
        hash() is salted per process (PYTHONHASHSEED), so the same image_id
        could land in different shards across runs. We use a stable md5 hash."""
        a = ShardedIndex(dim=64, num_shards=8)._get_shard_id("image_42")
        b = ShardedIndex(dim=64, num_shards=8)._get_shard_id("image_42")
        assert a == b

    def test_shard_id_matches_stable_formula(self):
        idx = ShardedIndex(dim=64, num_shards=8)
        expected = int(hashlib.md5(b"image_42").hexdigest(), 16) % 8
        assert idx._get_shard_id("image_42") == expected

    def test_shard_id_in_range(self):
        idx = ShardedIndex(dim=64, num_shards=4)
        for name in ["a", "b", "c", "some/long/path.jpg", "ünïcode"]:
            assert 0 <= idx._get_shard_id(name) < 4
