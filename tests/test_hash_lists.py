"""Tests for spade.hashing hash lists: parsing the file formats and matching."""

import time

import numpy as np
import pytest

from spade.hashing import HashList, ImageHash, load_hash_list, match_hashes

PDQ_A = "f8f8f0cee0f4a24f0b5b6a6ba0bd6d9d6d8d6d8d6d8d6d8d6d8d6d8d6d8d6d8d"[:64]
PDQ_B = "0" * 64
PHASH_A = "d1d1d1d1d1d1d1d1"


def flip_bits(hex_str, n):
    """The hash with its first n bits inverted."""
    raw = bytearray(bytes.fromhex(hex_str))
    for i in range(n):
        raw[i // 8] ^= 0x80 >> (i % 8)
    return bytes(raw).hex()


def write(tmp_path, text, name="list.csv", encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return str(path)


def pdq_list(*entries):
    """A HashList of PDQ entries, ids "e0", "e1", ..."""
    return HashList(
        algorithms=np.array(["pdq"] * len(entries), dtype=object),
        hexes=np.array(entries, dtype=object),
        ids=np.array([f"e{i}" for i in range(len(entries))], dtype=object),
    )


def entries(hl):
    return list(zip(hl.algorithms, hl.hexes, hl.ids))


def test_formats(tmp_path):
    expected = [("pdq", PDQ_A, "a"), ("pdq", PDQ_B, "b")]
    full = write(tmp_path, f"pdq,{PDQ_A},a\npdq,{PDQ_B},b\n", "full.csv")
    two = write(tmp_path, f"{PDQ_A},a\n{PDQ_B},b\n", "two.csv")
    header3 = write(tmp_path, f"algorithm,hash,id\npdq,{PDQ_A},a\npdq,{PDQ_B},b\n", "h3.csv")
    header2 = write(tmp_path, f"hash,id\n{PDQ_A},a\n{PDQ_B},b\n", "h2.csv")

    assert entries(load_hash_list(full)) == expected
    assert entries(load_hash_list(two, algorithm="pdq")) == expected
    assert entries(load_hash_list(header3)) == expected
    assert entries(load_hash_list(header2, algorithm="pdq")) == expected
    assert len(load_hash_list(full)) == 2

    bare = write(tmp_path, f"{PDQ_A}\n{PDQ_B}\n", "bare.csv")
    assert entries(load_hash_list(bare, algorithm="pdq")) == [("pdq", PDQ_A, "1"), ("pdq", PDQ_B, "2")]
    bare_header = write(tmp_path, f"hash\n{PDQ_A}\n{PDQ_B}\n", "bareh.csv")
    assert entries(load_hash_list(bare_header, algorithm="pdq")) == [("pdq", PDQ_A, "1"), ("pdq", PDQ_B, "2")]


def test_messy_files_parse(tmp_path):
    text = f"algorithm,hash,id\r\n\r\nPDQ,{PDQ_A.upper()},a\r\n\r\n   \r\npdq,{PDQ_B},b\r\n"
    path = write(tmp_path, "﻿" + text)
    assert entries(load_hash_list(path)) == [("pdq", PDQ_A, "a"), ("pdq", PDQ_B, "b")]
    # The same without a header, and with a BOM on the first hash itself.
    path = write(tmp_path, f"﻿{PDQ_A.upper()}\r\n\r\n{PDQ_B}\r\n", "bare.csv")
    got = entries(load_hash_list(path, algorithm="pdq"))
    assert [h for _, h, _ in got] == [PDQ_A, PDQ_B]


def test_bad_lines(tmp_path):
    short = write(tmp_path, f"pdq,{PDQ_A},a\npdq,{PDQ_A[:63]},b\n", "short.csv")
    with pytest.raises(ValueError, match=r"line 2"):
        load_hash_list(short)

    nonhex = write(tmp_path, f"pdq,{PDQ_A},a\n\npdq,{'g' + PDQ_A[1:]},b\n", "nonhex.csv")
    with pytest.raises(ValueError, match=r"line 3"):
        load_hash_list(nonhex)

    no_algo = write(tmp_path, f"{PDQ_A}\n{PDQ_B}\n", "noalgo.csv")
    with pytest.raises(ValueError, match=r"line 1"):
        load_hash_list(no_algo)

    unknown = write(tmp_path, f"md5,{PDQ_A},a\n", "unknown.csv")
    with pytest.raises(ValueError, match=r"line 1"):
        load_hash_list(unknown)

    wrong_len = write(tmp_path, f"phash,{PDQ_A},a\n", "wronglen.csv")
    with pytest.raises(ValueError, match=r"line 1"):
        load_hash_list(wrong_len)


def test_photodna_entries_load_but_never_match(tmp_path):
    path = write(tmp_path, f"photodna,AAAA,x\npdq,{PDQ_A},a\n")
    hl = load_hash_list(path)
    assert len(hl) == 2
    report = match_hashes([ImageHash("pdq", PDQ_A, 90)], hl)
    assert [m.id for m in report.matches] == ["a"]


def test_nearest_first_within_distance():
    hl = pdq_list(*(flip_bits(PDQ_A, n) for n in (32, 31, 5, 0)))
    query = [ImageHash("pdq", PDQ_A, 90)]
    report = match_hashes(query, hl)
    assert [m.distance for m in report.matches] == [0, 5, 31]
    assert [m.id for m in report.matches] == ["e3", "e2", "e1"]
    assert report.matches[0].algorithm == "pdq"
    assert report.matches[0].hex == PDQ_A
    assert [m.distance for m in match_hashes(query, hl, max_distance=5).matches] == [0, 5]


def test_other_algorithms_ignored():
    hl = HashList(
        algorithms=np.array(["phash", "pdq", "phash-lib"], dtype=object),
        hexes=np.array([PHASH_A, PDQ_A, PHASH_A], dtype=object),
        ids=np.array(["p", "q", "r"], dtype=object),
    )
    report = match_hashes([ImageHash("pdq", PDQ_A, 90)], hl)
    assert [m.id for m in report.matches] == ["q"]
    report = match_hashes([ImageHash("phash", PHASH_A, None)], hl)
    assert [m.id for m in report.matches] == ["p"]


def test_quality_gate():
    hl = pdq_list(PDQ_A)
    low = ImageHash("pdq", PDQ_A, 40)
    report = match_hashes([low], hl)
    assert report.matches == []
    assert report.skipped_low_quality == [low]
    assert [m.id for m in match_hashes([low], hl, min_quality=0).matches] == ["e0"]

    ph = ImageHash("phash", PHASH_A, None)
    ph_list = HashList(
        algorithms=np.array(["phash"], dtype=object),
        hexes=np.array([PHASH_A], dtype=object),
        ids=np.array(["p"], dtype=object),
    )
    report = match_hashes([ph], ph_list, min_quality=100)
    assert report.skipped_low_quality == []
    assert len(report.matches) == 1


def test_variant_reported():
    hl = pdq_list(flip_bits(PDQ_B, 3))
    queries = [
        ImageHash("pdq", PDQ_A, 90, "original"),
        ImageHash("pdq", PDQ_B, 90, "rot90"),
    ]
    report = match_hashes(queries, hl)
    assert len(report.matches) == 1
    assert report.matches[0].variant == "rot90"
    assert report.matches[0].distance == 3


def test_million_entries_fast():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(1_000_000, 32), dtype=np.uint8)
    flat = raw.tobytes().hex()
    hexes = np.array([flat[i : i + 64] for i in range(0, len(flat), 64)], dtype=object)
    hl = HashList(
        algorithms=np.full(len(hexes), "pdq", dtype=object),
        hexes=hexes,
        ids=np.array([str(i) for i in range(len(hexes))], dtype=object),
    )
    query = [ImageHash("pdq", hexes[123], 90)]
    start = time.perf_counter()
    report = match_hashes(query, hl)
    elapsed = time.perf_counter() - start
    print(f"million-entry match_hashes: {elapsed:.3f}s")
    assert [m.id for m in report.matches] == ["123"]
    assert elapsed < 1.0
