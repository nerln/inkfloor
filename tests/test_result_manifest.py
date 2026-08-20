"""The committed historical report must remain byte-for-byte identifiable."""

import hashlib
from pathlib import Path


def test_anchor_result_checksums():
    root = Path(__file__).resolve().parent.parent
    manifest = root / "results" / "anchors-2026-08-08.sha256"
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
