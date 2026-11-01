"""Fetch benchmark datasets into $SPADE_DATA_DIR (default ~/.cache/spade).

Usage:
    python -m spade.bench.download casia2      # or: coverage, comofod

Datasets come from Kaggle and need credentials: an API token in
KAGGLE_API_TOKEN (or KAGGLE_ACCESS_TOKEN), or KAGGLE_USERNAME and KAGGLE_KEY,
or ~/.kaggle/kaggle.json
(kaggle.com -> Settings -> API). They're research datasets under their own
terms; nothing here redistributes them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from spade.bench.datasets import CASIA2_KAGGLE_REF, COMOFOD_KAGGLE_REF, COVERAGE_KAGGLE_REF, data_dir

# name -> (kaggle ref, folder, a path that must exist afterwards, minimum files under it)
DATASETS = {
    "casia2": (CASIA2_KAGGLE_REF, "casia2", "CASIA2/Tp", 5000),
    "coverage": (COVERAGE_KAGGLE_REF, "coverage", ".", 200),
    "comofod": (COMOFOD_KAGGLE_REF, "comofod", ".", 480),
}


def _token():
    """An API token from the environment; both variable names are in use."""
    return os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_ACCESS_TOKEN")


def _kaggle_api():
    has_token = bool(_token())
    has_key = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    if not (has_token or has_key or (Path.home() / ".kaggle" / "kaggle.json").exists()):
        raise SystemExit(
            "Kaggle credentials missing. Set KAGGLE_API_TOKEN (or KAGGLE_ACCESS_TOKEN), or KAGGLE_USERNAME "
            "and KAGGLE_KEY "
            "(kaggle.com -> Settings -> API)."
        )
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit("the kaggle package is needed: pip install spade-forensics[bench]")
    api = KaggleApi()
    api.authenticate()
    return api


def _download_with_token(ref: str, target: Path, token: str) -> None:
    """Fetch a dataset archive through www.kaggle.com with an API token.

    The kaggle package checks new-style tokens against api.kaggle.com first,
    which restricted networks often block; the download endpoint itself
    accepts the token and redirects to a signed storage URL.
    """
    import zipfile

    import requests

    archive = target / "archive.zip"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}"
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(archive, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)
    archive.unlink()


def download(name: str) -> Path:
    ref, folder, check, min_files = DATASETS[name]
    target = data_dir() / folder
    done = target / check
    if done.exists() and sum(1 for _ in done.iterdir()) >= min_files:
        print(f"{name} already present in {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ref} into {target} ...")
    token = _token()
    if token:
        _download_with_token(ref, target, token)
    else:
        _kaggle_api().dataset_download_files(ref, path=str(target), unzip=True, quiet=False)
    count = sum(1 for _ in done.iterdir()) if done.exists() else 0
    if count < min_files:
        raise SystemExit(f"download looks incomplete: {count} files in {done}, expected >= {min_files}")
    print(f"{name}: {count} files in {done}")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    args = parser.parse_args(argv)
    download(args.dataset)


if __name__ == "__main__":
    sys.exit(main())
