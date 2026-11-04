"""Tests for the REST API."""

import csv
import os
import subprocess
import io
import sys
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from spade.api import create_app  # noqa: E402
from spade.bench.synthetic import make_recolored_splice  # noqa: E402


def _png(array):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return ("image.png", buf.getvalue(), "image/png")


@pytest.fixture(scope="module")
def sample():
    return make_recolored_splice(seed=0, image_size=96, fragment_size=24)


def test_index_match_delete(sample):
    client = TestClient(create_app())

    r = client.post("/index", params={"image_id": "donor"}, files={"file": _png(sample.donor)})
    assert r.status_code == 200 and r.json()["patches_indexed"] > 0
    assert client.post("/index", params={"image_id": "donor"}, files={"file": _png(sample.donor)}).status_code == 409

    r = client.post("/match", files={"file": _png(sample.tampered)})
    body = r.json()
    assert r.status_code == 200 and body["detected"]
    best = body["coherent_regions"][0]
    assert best["significant"] and best["log10_nfa"] < 0 and best["image_id"] == "donor"

    assert client.delete("/targets/donor").status_code == 200
    assert client.get("/health/ready").json()["index_size"] == 0
    assert client.delete("/targets/donor").status_code == 404


def test_generated_ids_do_not_collide_after_delete(sample):
    client = TestClient(create_app())
    first = client.post("/index", files={"file": _png(sample.donor)}).json()["image_id"]
    second = client.post("/index", files={"file": _png(sample.tampered)}).json()["image_id"]
    client.delete(f"/targets/{first}")
    third = client.post("/index", files={"file": _png(sample.donor)}).json()["image_id"]
    assert len({first, second, third}) == 3


# --- /hash -----------------------------------------------------------------

HASH_DATA = Path(__file__).parent / "data" / "hashes"


def _reference_pdq():
    with open(HASH_DATA / "expected.csv", newline="") as f:
        return next(r for r in csv.DictReader(f) if r["filename"] == "bridge-1-original.jpg")


def _bridge():
    return ("bridge.jpg", (HASH_DATA / "bridge-1-original.jpg").read_bytes(), "image/jpeg")


def test_hash_endpoint():
    pytest.importorskip("pdqhash")
    client = TestClient(create_app())
    r = client.post("/hash", params={"algorithm": "pdq"}, files={"file": _bridge()})
    assert r.status_code == 200
    body = r.json()
    assert "matches" not in body
    (h,) = body["hashes"]
    ref = _reference_pdq()
    assert h["algorithm"] == "pdq" and h["hash"] == ref["hash"]
    assert h["quality"] == int(ref["quality"]) and h["variant"] == "original"


def test_hash_endpoint_with_list(tmp_path):
    pytest.importorskip("pdqhash")
    ref = _reference_pdq()
    listing = tmp_path / "known.csv"
    listing.write_text(f"pdq,{ref['hash']},known-1\n")
    client = TestClient(create_app(hash_list_path=str(listing)))
    r = client.post("/hash", params={"algorithm": "pdq"}, files={"file": _bridge()})
    assert r.status_code == 200
    (m,) = r.json()["matches"]
    assert m["id"] == "known-1" and m["distance"] == 0


def test_hash_list_env_var(tmp_path, monkeypatch):
    pytest.importorskip("pdqhash")
    ref = _reference_pdq()
    listing = tmp_path / "known.csv"
    listing.write_text(f"pdq,{ref['hash']},from-env\n")
    monkeypatch.setenv("SPADE_HASH_LIST", str(listing))
    client = TestClient(create_app())
    r = client.post("/hash", params={"algorithm": "pdq"}, files={"file": _bridge()})
    assert r.json()["matches"][0]["id"] == "from-env"


def test_bad_hash_list_fails_at_startup(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("pdq,not-hex,x\n")
    with pytest.raises(ValueError, match="hash list"):
        create_app(hash_list_path=str(bad))
    with pytest.raises(ValueError, match="hash list"):
        create_app(hash_list_path=str(tmp_path / "missing.csv"))


def test_hash_endpoint_missing_algorithm():
    client = TestClient(create_app())
    with mock.patch("spade.hashing._load_phash_lib", return_value=None):
        r = client.post("/hash", params={"algorithm": "phash-lib"}, files={"file": _bridge()})
    assert r.status_code == 501
    assert "apt install libphash0" in r.json()["detail"]


def test_hash_endpoint_missing_package():
    client = TestClient(create_app())
    with mock.patch.dict(sys.modules, {"pdqhash": None}):
        r = client.post("/hash", params={"algorithm": "pdq"}, files={"file": _bridge()})
    assert r.status_code == 501
    assert "pip install spade-forensics[hash]" in r.json()["detail"]


def test_hash_endpoint_unknown_algorithm():
    client = TestClient(create_app())
    r = client.post("/hash", params={"algorithm": "nope"}, files={"file": _bridge()})
    assert r.status_code == 400


def test_hash_endpoint_libphash_failure():
    client = TestClient(create_app())
    from spade import hashing
    with mock.patch.object(hashing._ALGORITHMS["pdq"], "compute", side_effect=RuntimeError("boom")):
        r = client.post("/hash", params={"algorithm": "pdq"}, files={"file": _bridge()})
    assert r.status_code == 502
    assert "pdq" in r.json()["detail"]


def test_import_ignores_bad_hash_list(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("pdq,zz,x\n")
    code = "import spade.api"
    env = dict(os.environ, SPADE_HASH_LIST=str(bad))
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    # the lazily built default app is where the bad list surfaces
    code = "import spade.api as a\ntry:\n    a.app\nexcept ValueError as e:\n    print('bad list')"
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert "bad list" in done.stdout


def test_photodna_needs_server_key(monkeypatch):
    monkeypatch.delenv("SPADE_PHOTODNA_KEY", raising=False)
    client = TestClient(create_app())
    with mock.patch("spade.photodna.PhotoDnaClient") as client_cls:
        r = client.post("/hash", params={"algorithm": "photodna"}, files={"file": _bridge()})
    assert r.status_code == 403
    assert "SPADE_PHOTODNA_KEY" in r.json()["detail"]
    client_cls.assert_not_called()


def test_photodna_with_server_key(monkeypatch):
    monkeypatch.setenv("SPADE_PHOTODNA_KEY", "secret-key")
    client = TestClient(create_app())
    with mock.patch("spade.photodna.PhotoDnaClient") as client_cls:
        client_cls.return_value.match.return_value = {"IsMatch": False}
        r = client.post("/hash", params={"algorithm": "photodna"}, files={"file": _bridge()})
    assert r.status_code == 200
    body = r.json()
    assert body["photodna"] == {"IsMatch": False} and body["hashes"] == []
    assert "secret-key" not in r.text


def test_hash_endpoint_rejects_bad_upload():
    client = TestClient(create_app())
    r = client.post("/hash", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_hash_endpoint_nothing_installed():
    client = TestClient(create_app())
    from spade import hashing
    with mock.patch.dict(sys.modules, {"pdqhash": None, "imagehash": None}), \
            mock.patch.object(hashing, "_load_phash_lib", return_value=None):
        r = client.post("/hash", files={"file": _bridge()})
    assert r.status_code == 501
    assert "pip install spade-forensics[hash]" in r.json()["detail"]
