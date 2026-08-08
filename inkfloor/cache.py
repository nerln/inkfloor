"""Accesso in sola lettura al bucket pubblico della Vesuvius Challenge.

Un solo punto di ingresso per la rete, così i moduli a monte non scrivono file da soli e
la cache resta in un posto prevedibile. Nessuna credenziale: il bucket è pubblico e le
richieste sono anonime.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "vesuvius-challenge-open-data"
HOST = f"https://{BUCKET}.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# Sull'SSD esterno per default: il corpus è grande e non deve riempire il disco di sistema.
CACHE_ROOT = Path(os.environ.get("INKFLOOR_CACHE", "/Volumes/AppsAndFiles/dev/inkfloor/cache"))

TIMEOUT = 300


class FetchError(RuntimeError):
    """Il bucket ha risposto in modo che non possiamo usare."""


def _local_path(key: str) -> Path:
    return CACHE_ROOT / key


def fetch(key: str) -> Path:
    """Scarica `key` in cache e restituisce il path locale. Non riscarica se già presente.

    Non valida il contenuto: un file troncato da una corsa precedente resta troncato. Usa
    `fetch(key, verify=True)` quando servirà, per ora la cache è append-only e monoprocesso.
    """
    dst = _local_path(key)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(f"{HOST}/{key}", timeout=TIMEOUT) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"HTTP {e.code} su {key}") from e
    tmp.replace(dst)
    return dst


def get_bytes(key: str, start: int | None = None, end: int | None = None) -> bytes:
    """GET, opzionalmente con Range. Le richieste parziali NON vengono messe in cache."""
    req = urllib.request.Request(f"{HOST}/{key}")
    if start is not None:
        req.add_header("Range", f"bytes={start}-{'' if end is None else end}")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise FetchError(f"HTTP {e.code} su {key}") from e


def list_keys(prefix: str, suffix: str | None = None) -> list[tuple[str, int]]:
    """Elenca (key, size) sotto `prefix`, seguendo la paginazione fino in fondo.

    Non usa delimiter: restituisce le key ricorsivamente. Per l'elenco delle "cartelle"
    usa `list_prefixes`.
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
        nxt = root.find(NS + "NextContinuationToken")
        if nxt is None:
            return out
        token = nxt.text


def list_prefixes(prefix: str) -> list[str]:
    """I "sottodirectory" immediati sotto `prefix`."""
    url = f"{HOST}/?list-type=2&prefix={urllib.parse.quote(prefix)}&delimiter=/&max-keys=1000"
    root = ET.fromstring(_raw(url))
    return [p.find(NS + "Prefix").text for p in root.findall(NS + "CommonPrefixes")]


def get_json(key: str) -> dict:
    return json.loads(fetch(key).read_bytes())


def _raw(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise FetchError(f"HTTP {e.code} su {url}") from e


def cache_size_bytes() -> int:
    if not CACHE_ROOT.exists():
        return 0
    return sum(p.stat().st_size for p in CACHE_ROOT.rglob("*") if p.is_file())
