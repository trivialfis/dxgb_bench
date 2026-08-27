"""Atomic filesystem and download helpers for public dataset sources."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
USER_AGENT = "dxgb-bench-public-datasets/1.0"


def write_json(path: Path, value: Any) -> None:
    """Atomically write JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_array(path: Path, array: np.ndarray) -> None:
    """Atomically write an ``.npy`` array without enabling pickles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    with temporary.open("wb") as sink:
        np.save(sink, array, allow_pickle=False)
    temporary.replace(path)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local source file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> Path:
    """Download one source atomically, resuming a partial file when supported."""
    if destination.is_file():
        print(f"Using cached download {destination}", flush=True)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(url, headers=headers)
    print(f"Fetching {url}", flush=True)
    with urllib.request.urlopen(request, timeout=180) as response:
        resumed = offset > 0 and response.status == 206
        if not resumed:
            offset = 0
        mode = "ab" if resumed else "wb"
        response_length = response.headers.get("Content-Length")
        expected = offset + int(response_length) if response_length else None
        downloaded = offset
        last_report = time.monotonic()

        with partial.open(mode) as sink:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                sink.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5.0:
                    suffix = (
                        f"/{expected / 2**20:.1f} MiB" if expected is not None else ""
                    )
                    print(
                        f"  downloaded {downloaded / 2**20:.1f}{suffix}",
                        flush=True,
                    )
                    last_report = now

        if expected is not None and downloaded != expected:
            raise OSError(
                f"Incomplete download for {url}: expected {expected} bytes, "
                f"received {downloaded}"
            )

    partial.replace(destination)
    print(f"Cached {destination} ({destination.stat().st_size / 2**20:.1f} MiB)")
    return destination
