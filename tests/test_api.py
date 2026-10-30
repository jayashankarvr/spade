"""Tests for the REST API."""

import io

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
