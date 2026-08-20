"""Read-only access to the public Vesuvius Challenge bucket.

A single entry point for network access, so upstream modules do not write files on their own
and the cache stays in a predictable location. No credentials: the bucket is public and
requests are anonymous.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "vesuvius-challenge-open-data"
HOST = f"https://{BUCKET}.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _default_cache_root() -> Path:
    """Where to keep downloaded predictions when nobody says otherwise.

    This used to default to an absolute path on the author's external SSD, which was the
    right choice for one machine and a broken one for every other: a corpus run is tens of
    gigabytes, so the size concern is real, but hard-coding /Volumes/... shipped a tool that
    writes to a disk only its author has. Found by installing the published package into a
    clean virtualenv and reading the first line it printed.

    The environment variable still wins, and is the answer when the system disk is too small:

        INKFLOOR_CACHE=/Volumes/BigDisk/inkfloor inkfloor corpus --kind model
    """
    env = os.environ.get("INKFLOOR_CACHE")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "inkfloor"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "inkfloor"


CACHE_ROOT = _default_cache_root()

TIMEOUT = 300


class FetchError(RuntimeError):
    """The bucket responded in a way that we cannot use."""


class CacheIntegrityError(FetchError):
    """A cached or downloaded object does not match its declared identity."""


def _local_path(key: str) -> Path:
    return CACHE_ROOT / key


def _verify_file(
    path: Path,
    key: str,
    *,
    expected_size: int | None = None,
    sha256: str | None = None,
) -> None:
    """Raise when a local object disagrees with supplied immutable metadata."""
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise CacheIntegrityError(
            f"cached size mismatch for {key}: expected {expected_size} bytes, found {size}; "
            f"remove {path} and retry"
        )
    if sha256 is None:
        return
    expected = sha256.lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("sha256 must be a 64-character hexadecimal digest")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise CacheIntegrityError(
            f"cached SHA-256 mismatch for {key}: expected {expected}, found {actual}; "
            f"remove {path} and retry"
        )


def fetch(
    key: str,
    *,
    expected_size: int | None = None,
    sha256: str | None = None,
) -> Path:
    """Download `key` into the cache and return the local path.

    Existing files are reused only after checking any supplied object size and SHA-256.
    Downloads are always checked against their HTTP Content-Length, when present, as well as
    the supplied metadata. A mismatch is reported without overwriting the suspect cache file.

    `expected_size` normally comes from the same S3 listing that discovered a prediction.
    A published result manifest can additionally pass `sha256` to pin content even if an
    object is later replaced by another object of the same size.
    """
    if sha256 is not None:
        expected = sha256.lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
    dst = _local_path(key)
    if dst.exists() and dst.stat().st_size > 0:
        _verify_file(dst, key, expected_size=expected_size, sha256=sha256)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(f"{HOST}/{key}", timeout=TIMEOUT) as r, open(tmp, "wb") as f:
            response_size = r.headers.get("Content-Length")
            while chunk := r.read(1 << 20):
                f.write(chunk)
        declared_size = expected_size
        if declared_size is None and response_size is not None:
            declared_size = int(response_size)
        _verify_file(tmp, key, expected_size=declared_size, sha256=sha256)
    except (urllib.error.HTTPError, urllib.error.URLError, CacheIntegrityError) as e:
        tmp.unlink(missing_ok=True)
        if isinstance(e, CacheIntegrityError):
            raise
        if isinstance(e, urllib.error.HTTPError):
            raise FetchError(f"HTTP {e.code} on {key}") from e
        raise FetchError(f"network error on {key}: {e.reason}") from e
    tmp.replace(dst)
    return dst


def get_bytes(key: str, start: int | None = None, end: int | None = None) -> bytes:
    """GET, optionally with Range. Partial requests are NOT cached."""
    req = urllib.request.Request(f"{HOST}/{key}")
    if start is not None:
        req.add_header("Range", f"bytes={start}-{'' if end is None else end}")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise FetchError(f"HTTP {e.code} on {key}") from e
    except urllib.error.URLError as e:
        raise FetchError(f"network error on {key}: {e.reason}") from e


def list_keys(prefix: str, suffix: str | None = None) -> list[tuple[str, int]]:
    """List (key, size) under `prefix`, following pagination all the way through.

    Does not use delimiter: returns keys recursively. Use `list_prefixes` to list
    "directories".
    """
    out: list[tuple[str, int]] = []
    token: str | None = None
    while True:
        url = f"{HOST}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        root = ET.fromstring(_raw(url))
        for c in root.findall(NS + "Contents"):
            key = c.find(NS + "Key").text
            size = int(c.find(NS + "Size").text)
            if suffix is None or key.endswith(suffix):
                out.append((key, size))
        nxt = root.findtext(NS + "NextContinuationToken")
        truncated = root.findtext(NS + "IsTruncated", "false").lower() == "true"
        if not truncated:
            return out
        if not nxt or nxt == token:
            raise FetchError(f"truncated listing for {prefix!r} did not advance")
        token = nxt


def list_prefixes(prefix: str) -> list[str]:
    """The immediate "subdirectories" under `prefix`, following all pages."""
    out: list[str] = []
    token: str | None = None
    while True:
        url = f"{HOST}/?list-type=2&prefix={urllib.parse.quote(prefix)}&delimiter=/&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        root = ET.fromstring(_raw(url))
        for item in root.findall(NS + "CommonPrefixes"):
            value = item.findtext(NS + "Prefix")
            if value is not None:
                out.append(value)
        nxt = root.findtext(NS + "NextContinuationToken")
        truncated = root.findtext(NS + "IsTruncated", "false").lower() == "true"
        if not truncated:
            return out
        if not nxt or nxt == token:
            raise FetchError(f"truncated prefix listing for {prefix!r} did not advance")
        token = nxt


def get_json(key: str) -> dict:
    return json.loads(fetch(key).read_bytes())


def _raw(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise FetchError(f"HTTP {e.code} on {url}") from e
    except urllib.error.URLError as e:
        raise FetchError(f"network error on {url}: {e.reason}") from e


def cache_size_bytes() -> int:
    if not CACHE_ROOT.exists():
        return 0
    return sum(p.stat().st_size for p in CACHE_ROOT.rglob("*") if p.is_file())
