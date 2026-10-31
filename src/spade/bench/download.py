"""Fetch benchmark datasets into $SPADE_DATA_DIR (default ~/.cache/spade).

Usage:
    python -m spade.bench.download casia2

Datasets come from Kaggle and need an API token: set KAGGLE_USERNAME and
KAGGLE_KEY (kaggle.com -> Settings -> API -> Create New Token). They're
research datasets under their own terms; nothing here redistributes them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from spade.bench.datasets import CASIA2_KAGGLE_REF, data_dir

# name -> (kaggle ref, folder, a path that must exist afterwards, minimum files under it)
DATASETS = {
    "casia2": (CASIA2_KAGGLE_REF, "casia2", "CASIA2/Tp", 5000),
}


def _kaggle_api():
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) and not (
        Path.home() / ".kaggle" / "kaggle.json"
    ).exists():
        raise SystemExit(
            "Kaggle credentials missing. Set KAGGLE_USERNAME and KAGGLE_KEY "
            "(kaggle.com -> Settings -> API -> Create New Token)."
        )
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise SystemExit("the kaggle package is needed: pip install spade-forensics[bench]")
    api = KaggleApi()
    api.authenticate()
    return api


def download(name: str) -> Path:
    ref, folder, check, min_files = DATASETS[name]
    target = data_dir() / folder
    done = target / check
    if done.exists() and sum(1 for _ in done.iterdir()) >= min_files:
        print(f"{name} already present in {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ref} into {target} ...")
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
