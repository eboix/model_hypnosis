#!/usr/bin/env python3
"""Fetch the experiment-data archive from the paper's data release.

One archive holds every per-cell result -- the per-trial exact log-odds
(``raw.jsonl``), the fitted models, the measured extremizers, and the transfer
results -- i.e. everything needed to regenerate the figures exactly with
``make figures`` (no GPU). It extracts into ``data/`` at the repo root.

Set the release URL via the ``MHYP_DATA_URL`` environment variable or ``--url``
until it is baked into ``ARCHIVE_URL`` below.

    MHYP_DATA_URL=https://zenodo.org/records/.../data.tar.gz \
        python scripts/download_data.py
"""
import argparse
import hashlib
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Zenodo record 21981022 (DOI 10.5281/zenodo.21981022); MHYP_DATA_URL overrides.
ARCHIVE_URL = os.environ.get(
    "MHYP_DATA_URL",
    "https://zenodo.org/records/21981022/files/mhyp-data.tar.gz?download=1")
ARCHIVE_SHA256 = "e545fc93966556aaef41501000b7d1480c27491bdc4298b4233158453f83464e"


def _download(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk); done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {done/1e6:8.1f} / {total/1e6:.1f} MB")
                sys.stdout.flush()
    print()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="", help="override the archive URL")
    args = ap.parse_args()

    url = args.url or ARCHIVE_URL
    if not url:
        sys.exit("No archive URL. Set $MHYP_DATA_URL, pass --url, or fill "
                 "ARCHIVE_URL in this script.")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "data.tar.gz"
        _download(url, tmp)
        if ARCHIVE_SHA256:
            got = _sha256(tmp)
            if got != ARCHIVE_SHA256:
                sys.exit(f"checksum mismatch: got {got}, expected {ARCHIVE_SHA256}")
            print("checksum ok")
        print(f"extracting into {ROOT}")
        with tarfile.open(tmp) as t:
            t.extractall(ROOT)          # archive paths are repo-relative (data/...)
    print("done.")


if __name__ == "__main__":
    main()
