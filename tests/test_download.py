"""Where the dataset downloader finds a Kaggle token."""

import pytest

from spade.bench import download


@pytest.fixture
def no_token_env(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(download.Path, "home", lambda: tmp_path)
    return tmp_path


def test_reads_access_token_file(no_token_env):
    (no_token_env / ".kaggle").mkdir()
    (no_token_env / ".kaggle" / "access_token").write_text("abc123\n")
    assert download._token() == "abc123"


def test_environment_wins_over_file(no_token_env, monkeypatch):
    (no_token_env / ".kaggle").mkdir()
    (no_token_env / ".kaggle" / "access_token").write_text("from-file")
    monkeypatch.setenv("KAGGLE_ACCESS_TOKEN", "from-env")
    assert download._token() == "from-env"


def test_no_token_anywhere(no_token_env):
    assert download._token() is None
