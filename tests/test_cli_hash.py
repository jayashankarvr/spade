"""Tests for the `spade hash` command."""

import json
from pathlib import Path

import pytest

pytest.importorskip("click")
pytest.importorskip("pdqhash")
pytest.importorskip("imagehash")

from click.testing import CliRunner

from spade.cli import cli
from spade.hashing import available_algorithms

DATA = Path(__file__).parent / "data" / "hashes"
ORIGINAL = str(DATA / "bridge-1-original.jpg")
ROTATED = str(DATA / "bridge-2-rotate-90.jpg")


def expected_pdq(name):
    for line in (DATA / "expected.csv").read_text().splitlines()[1:]:
        filename, value, _ = line.split(",")
        if filename == name:
            return value
    raise KeyError(name)


def write_list(tmp_path, *rows):
    path = tmp_path / "list.csv"
    path.write_text("algorithm,hash,id\n" + "".join(r + "\n" for r in rows))
    return str(path)


def run(*args):
    return CliRunner().invoke(cli, ["hash", *args])


def test_prints_hashes():
    result = run(ORIGINAL, ROTATED, "--algorithm", "pdq", "--algorithm", "phash")
    assert result.exit_code == 0, result.output
    lines = [l for l in result.output.splitlines() if l.strip()]
    assert len(lines) == 4
    pdq_line = next(l for l in lines if "bridge-1-original.jpg" in l and "pdq" in l)
    assert expected_pdq("bridge-1-original.jpg") in pdq_line
    assert "quality=100" in pdq_line
    phash_line = next(l for l in lines if "bridge-1-original.jpg" in l and "phash" in l)
    assert "quality" not in phash_line


def test_against_list(tmp_path):
    hit = write_list(tmp_path, "pdq," + expected_pdq("bridge-1-original.jpg") + ",known-1")
    result = run(ORIGINAL, "--algorithm", "pdq", "--against", hit)
    assert result.exit_code == 0, result.output
    assert "known-1" in result.output
    assert "distance=0" in result.output

    miss = write_list(tmp_path, "pdq," + "0" * 64 + ",other")
    result = run(ORIGINAL, "--algorithm", "pdq", "--against", miss)
    assert result.exit_code == 0, result.output
    assert "no match" in result.output
    assert "other" not in result.output


def test_thorough_matches_rotation(tmp_path):
    only = write_list(tmp_path, "pdq," + expected_pdq("bridge-2-rotate-90.jpg") + ",rotated")
    plain = run(ORIGINAL, "--algorithm", "pdq", "--against", only)
    assert plain.exit_code == 0, plain.output
    assert "no match" in plain.output
    thorough = run(ORIGINAL, "--algorithm", "pdq", "--against", only, "--thorough")
    assert thorough.exit_code == 0, thorough.output
    assert "rotated" in thorough.output
    assert "rot90" in thorough.output


def test_json_output(tmp_path):
    lst = write_list(tmp_path, "pdq," + expected_pdq("bridge-1-original.jpg") + ",known-1")
    out = tmp_path / "out.json"
    result = run(ORIGINAL, "--algorithm", "pdq", "--against", lst, "--json", str(out))
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    (image,) = data["images"]
    assert image["path"] == ORIGINAL
    assert image["hashes"][0]["algorithm"] == "pdq"
    assert image["hashes"][0]["hex"] == expected_pdq("bridge-1-original.jpg")
    assert image["hashes"][0]["quality"] == 100
    assert image["matches"][0]["id"] == "known-1"
    assert image["matches"][0]["distance"] == 0
    assert image["skipped_low_quality"] == []


def test_errors(tmp_path):
    bad_image = tmp_path / "bad.jpg"
    bad_image.write_text("not an image")
    result = run(str(bad_image), "--algorithm", "pdq")
    assert result.exit_code != 0
    assert "Traceback" not in result.output and "bad.jpg" in result.output

    bad_list = tmp_path / "bad.csv"
    bad_list.write_text("pdq,xyz,id1\n")
    result = run(ORIGINAL, "--algorithm", "pdq", "--against", str(bad_list))
    assert result.exit_code != 0
    assert "line 1" in result.output
    assert not isinstance(result.exception, ValueError)

    if "phash-lib" not in available_algorithms():
        result = run(ORIGINAL, "--algorithm", "phash-lib")
        assert result.exit_code != 0
        assert "libpHash" in result.output
        assert isinstance(result.exception, SystemExit)

    result = run(ORIGINAL, "--algorithm", "bogus")
    assert result.exit_code != 0
    assert "bogus" in result.output


def test_list_without_algorithm_column(tmp_path):
    pdq = expected_pdq("bridge-1-original.jpg")
    two_columns = tmp_path / "two.csv"
    two_columns.write_text(f"{pdq},bridge\n")
    bare = tmp_path / "bare.txt"
    bare.write_text(f"{pdq}\n")

    result = run(ORIGINAL, "--algorithm", "pdq", "--against", str(two_columns))
    assert result.exit_code == 0, result.output
    assert "match bridge (pdq) distance=0" in result.output

    result = run(ORIGINAL, "-a", "PDQ", "--against", str(bare))
    assert result.exit_code == 0, result.output
    assert "match 1 (pdq) distance=0" in result.output

    # With no single --algorithm the list's algorithm is unknown.
    for args in ([], ["-a", "pdq", "-a", "phash"], ["-a", "all"]):
        result = run(ORIGINAL, *args, "--against", str(two_columns))
        assert result.exit_code != 0
        assert "add an algorithm column" in result.output
        assert "--algorithm" in result.output
        assert "algorithm=" not in result.output
        assert "line 1" in result.output


def test_all_excludes_photodna():
    result = run(ORIGINAL, "--algorithm", "all")
    assert result.exit_code == 0, result.output
    assert "photodna" not in result.output.lower()
    algos = {line.split()[1] for line in result.output.splitlines() if line.strip()}
    assert algos == set(available_algorithms())
    default = run(ORIGINAL)
    assert default.exit_code == 0 and "photodna" not in default.output.lower()


def _nothing_installed(monkeypatch):
    import sys

    from spade import hashing
    monkeypatch.setitem(sys.modules, "pdqhash", None)
    monkeypatch.setitem(sys.modules, "imagehash", None)
    monkeypatch.setattr(hashing, "_load_phash_lib", lambda: None)


def test_named_algorithm_not_installed(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pdqhash", None)
    result = run(ORIGINAL, "--algorithm", "pdq")
    assert result.exit_code != 0
    assert "pip install spade-forensics[hash]" in result.output
    assert "Unknown algorithm" not in result.output


def test_nothing_installed(monkeypatch):
    _nothing_installed(monkeypatch)
    result = run(ORIGINAL)
    assert result.exit_code != 0
    assert "pip install spade-forensics[hash]" in result.output


def test_not_compared_when_quality_too_low(tmp_path):
    from PIL import Image

    flat = tmp_path / "flat.png"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(flat)
    listing = write_list(tmp_path, "pdq," + "0" * 64 + ",other")
    result = run(str(flat), "--algorithm", "pdq", "--against", listing)
    assert result.exit_code == 0, result.output
    assert "no match" not in result.output
    assert "not compared (all hashes below quality 50)" in result.output

    result = run(ORIGINAL, "--algorithm", "pdq", "--against", listing)
    assert "  no match" in result.output


def test_list_id_is_escaped(tmp_path):
    path = tmp_path / "list.csv"
    path.write_text("pdq," + expected_pdq("bridge-1-original.jpg") + ",evil\x1b[2Jid\n")
    result = run(ORIGINAL, "--algorithm", "pdq", "--against", str(path))
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output.replace("\\x1b", "")
    assert "evil\\x1b[2Jid" in result.output


def test_json_path_checked_before_hashing(tmp_path):
    result = run(ORIGINAL, "--algorithm", "pdq", "--json", str(tmp_path / "missing-dir" / "out.json"))
    assert result.exit_code != 0
    assert "Can't write" in result.output
    assert ORIGINAL not in result.output


def test_failed_run_leaves_json_output_alone(tmp_path):
    out = tmp_path / "out.json"
    out.write_text("earlier result")
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    result = run(str(broken), "--algorithm", "pdq", "--json", str(out))
    assert result.exit_code != 0
    assert out.read_text() == "earlier result"
