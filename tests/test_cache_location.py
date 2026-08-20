"""The cache must land somewhere the user actually has.

The published package defaulted `CACHE_ROOT` to an absolute path on the author's external
SSD. It worked on one machine and was broken on every other, and it was found by installing
the published package into a clean virtualenv and reading the first line it printed. A tool
whose pitch is "runs on a laptop, no credentials" must not write to a volume only its author
has mounted.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest


def _root(monkeypatch, **env) -> Path:
    import importlib

    from inkfloor import cache as cache_mod

    for k in ("INKFLOOR_CACHE", "XDG_CACHE_HOME"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(cache_mod).CACHE_ROOT


def test_default_is_under_the_user_home(monkeypatch):
    root = _root(monkeypatch)
    assert Path.home() in root.parents, f"default cache {root} is outside the user's home"


def test_default_is_not_an_author_specific_volume(monkeypatch):
    root = _root(monkeypatch)
    assert "AppsAndFiles" not in str(root)
    assert not str(root).startswith("/Volumes/"), f"{root} is a mount only its author has"


def test_environment_wins(monkeypatch, tmp_path):
    assert _root(monkeypatch, INKFLOOR_CACHE=str(tmp_path / "elsewhere")) == tmp_path / "elsewhere"


@pytest.mark.skipif(os.sys.platform == "darwin", reason="XDG is the Linux convention")
def test_xdg_is_honoured_off_macos(monkeypatch, tmp_path):
    assert _root(monkeypatch, XDG_CACHE_HOME=str(tmp_path)) == tmp_path / "inkfloor"


def test_cached_object_is_checked_against_listing_size(monkeypatch, tmp_path):
    from inkfloor import cache

    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    key = "sample/prediction.tif"
    path = tmp_path / key
    path.parent.mkdir(parents=True)
    path.write_bytes(b"complete")

    assert cache.fetch(key, expected_size=8) == path
    with pytest.raises(cache.CacheIntegrityError, match="expected 7 bytes, found 8"):
        cache.fetch(key, expected_size=7)


def test_cached_object_accepts_and_rejects_explicit_sha256(monkeypatch, tmp_path):
    from inkfloor import cache

    monkeypatch.setattr(cache, "CACHE_ROOT", tmp_path)
    key = "sample/prediction.tif"
    path = tmp_path / key
    path.parent.mkdir(parents=True)
    payload = b"published artifact"
    path.write_bytes(payload)

    digest = hashlib.sha256(payload).hexdigest()
    assert cache.fetch(key, sha256=digest) == path
    with pytest.raises(cache.CacheIntegrityError, match="SHA-256 mismatch"):
        cache.fetch(key, sha256="0" * 64)


def test_list_prefixes_follows_continuation_tokens(monkeypatch):
    from inkfloor import cache

    page_1 = b"""<?xml version='1.0' encoding='UTF-8'?>
    <ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>
      <IsTruncated>true</IsTruncated>
      <CommonPrefixes><Prefix>PHerc0172/</Prefix></CommonPrefixes>
      <NextContinuationToken>next page</NextContinuationToken>
    </ListBucketResult>"""
    page_2 = b"""<?xml version='1.0' encoding='UTF-8'?>
    <ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>
      <IsTruncated>false</IsTruncated>
      <CommonPrefixes><Prefix>PHerc0500P2/</Prefix></CommonPrefixes>
    </ListBucketResult>"""
    urls: list[str] = []

    def fake_raw(url: str) -> bytes:
        urls.append(url)
        return page_2 if "continuation-token=" in url else page_1

    monkeypatch.setattr(cache, "_raw", fake_raw)
    assert cache.list_prefixes("") == ["PHerc0172/", "PHerc0500P2/"]
    assert len(urls) == 2
    assert "continuation-token=next%20page" in urls[1]
