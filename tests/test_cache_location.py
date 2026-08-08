"""The cache must land somewhere the user actually has.

The published package defaulted `CACHE_ROOT` to an absolute path on the author's external
SSD. It worked on one machine and was broken on every other, and it was found by installing
the published package into a clean virtualenv and reading the first line it printed. A tool
whose pitch is "runs on a laptop, no credentials" must not write to a volume only its author
has mounted.
"""

from __future__ import annotations

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
