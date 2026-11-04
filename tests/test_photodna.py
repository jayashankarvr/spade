"""Tests for the PhotoDNA Cloud Service adapter.

Nothing here touches the network: requests.post is always mocked, and the
key is a made-up string.
"""

import base64
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from click.testing import CliRunner

from spade import photodna
from spade.cli import cli
from spade.photodna import PhotoDnaClient, PhotoDnaError

KEY = "sekret-key-0123456789abcdef"
ENDPOINT = "https://photodna.example.test/v1.0/Match"
IMAGE = Path(__file__).parent / "data" / "hashes" / "bridge-1-original.jpg"


def response(status=200, json_body=None, text="", json_error=None):
    r = MagicMock()
    r.status_code = status
    r.text = text
    if json_error:
        r.json.side_effect = json_error
    else:
        r.json.return_value = json_body
    return r


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("SPADE_PHOTODNA_KEY", raising=False)
    monkeypatch.delenv("SPADE_PHOTODNA_ENDPOINT", raising=False)


def test_no_key():
    with pytest.raises(PhotoDnaError) as exc:
        PhotoDnaClient()
    msg = str(exc.value)
    assert "licence" in msg and "Cloud Service key" in msg
    assert "https://www.microsoft.com/en-us/photodna" in msg


def test_key_from_environment(monkeypatch):
    monkeypatch.setenv("SPADE_PHOTODNA_KEY", KEY)
    monkeypatch.setenv("SPADE_PHOTODNA_ENDPOINT", ENDPOINT)
    verdict = {"Status": {"Code": 3000}, "IsMatch": False}
    with patch("requests.post", return_value=response(json_body=verdict)) as post:
        assert PhotoDnaClient().match(b"img") == verdict
    assert post.call_args.args[0] == ENDPOINT
    assert post.call_args.kwargs["headers"]["Ocp-Apim-Subscription-Key"] == KEY


def test_sends_image_and_key_header():
    verdict = {"Status": {"Code": 3000}, "IsMatch": True, "MatchDetails": {"MatchFlags": []}}
    client = PhotoDnaClient(key=KEY, endpoint=ENDPOINT, timeout=7.5)
    with patch("requests.post", return_value=response(json_body=verdict)) as post:
        assert client.match(b"\x00\x01image") == verdict
    assert post.call_args.args[0] == ENDPOINT
    kw = post.call_args.kwargs
    assert kw["headers"]["Ocp-Apim-Subscription-Key"] == KEY
    assert kw["json"] == {"DataRepresentation": "inline",
                          "Value": base64.b64encode(b"\x00\x01image").decode()}
    assert kw["timeout"] == 7.5


def test_default_endpoint():
    client = PhotoDnaClient(key=KEY)
    with patch("requests.post", return_value=response(json_body={})) as post:
        client.match(b"x")
    assert post.call_args.args[0] == "https://api.microsoftmoderator.com/photodna/v1.0/Match"


def test_repr_hides_key():
    client = PhotoDnaClient(key=KEY, endpoint=ENDPOINT)
    assert KEY not in repr(client) and KEY not in str(client)


@pytest.mark.parametrize("outcome", ["timeout", "http500", "html", "connection"])
def test_service_error_hides_key(outcome, caplog):
    client = PhotoDnaClient(key=KEY, endpoint=ENDPOINT)
    if outcome == "timeout":
        kwargs = {"side_effect": requests.Timeout(f"timed out, header {KEY}")}
    elif outcome == "connection":
        kwargs = {"side_effect": requests.ConnectionError(f"refused for {KEY}")}
    elif outcome == "http500":
        kwargs = {"return_value": response(500, text=f"boom {KEY}", json_error=ValueError("no json"))}
    else:
        kwargs = {"return_value": response(200, text=f"<html>{KEY}</html>",
                                           json_error=ValueError("not json"))}
    with caplog.at_level(logging.DEBUG), patch("requests.post", **kwargs):
        with pytest.raises(PhotoDnaError) as exc:
            client.match(b"img")
    assert KEY not in str(exc.value)
    assert KEY not in repr(exc.value)
    assert KEY not in caplog.text
    assert exc.value.__cause__ is None or KEY not in str(exc.value.__cause__)


def test_http_500_mentions_status():
    client = PhotoDnaClient(key=KEY, endpoint=ENDPOINT)
    with patch("requests.post", return_value=response(500, json_error=ValueError())):
        with pytest.raises(PhotoDnaError, match="500"):
            client.match(b"img")


def run(*args):
    return CliRunner().invoke(cli, ["hash", *args])


def test_cli_needs_explicit_name():
    pytest.importorskip("pdqhash")
    fake = MagicMock()
    fake.match.return_value = {"IsMatch": False, "Status": {"Code": 3000}}
    with patch.object(photodna, "PhotoDnaClient", return_value=fake) as ctor:
        plain = run(str(IMAGE))
        every = run(str(IMAGE), "--algorithm", "all")
        assert plain.exit_code == 0 and every.exit_code == 0
        ctor.assert_not_called()

        result = run(str(IMAGE), "--algorithm", "photodna")
    assert result.exit_code == 0, result.output
    assert "photodna" in result.output and "IsMatch" in result.output
    ctor.assert_called_once_with()
    fake.match.assert_called_once_with(IMAGE.read_bytes())


def test_cli_json_and_local_algorithms(tmp_path):
    pytest.importorskip("pdqhash")
    fake = MagicMock()
    fake.match.return_value = {"IsMatch": True}
    out = tmp_path / "out.json"
    with patch.object(photodna, "PhotoDnaClient", return_value=fake):
        result = run(str(IMAGE), "-a", "pdq", "-a", "photodna", "--json", str(out))
    assert result.exit_code == 0, result.output
    import json
    image = json.loads(out.read_text())["images"][0]
    assert image["photodna"] == {"IsMatch": True}
    assert [h["algorithm"] for h in image["hashes"]] == ["pdq"]


def test_cli_without_key_exits_cleanly():
    result = run(str(IMAGE), "--algorithm", "photodna")
    assert result.exit_code != 0
    assert "licence" in result.output
    assert "Traceback" not in result.output


def test_cli_service_error_exits_cleanly():
    fake = MagicMock()
    fake.match.side_effect = PhotoDnaError("PhotoDNA service returned HTTP 500")
    with patch.object(photodna, "PhotoDnaClient", return_value=fake):
        result = run(str(IMAGE), "--algorithm", "photodna")
    assert result.exit_code != 0
    assert "HTTP 500" in result.output
