"""stdout belongs to the machine when --json is asked for.

The first corpus run of this tool wrote 341 KB that would not parse, because the download
plan and the progress lines sat on stdout above the JSON document. Narration is still worth
having during a run that moves gigabytes, so it goes to stderr rather than away.
"""

from __future__ import annotations

import json

from inkfloor import cli


def test_json_keeps_the_plan_off_stdout(capsys):
    rc = cli.main(["corpus", "--dry-run", "--json"])
    out, err = capsys.readouterr()
    assert rc == cli.EXIT_OK
    assert out == "", f"stdout must carry only the JSON document, got {out[:120]!r}"
    assert "plan:" in err, "the plan should still be shown, on stderr"


def test_without_json_the_plan_stays_on_stdout(capsys):
    rc = cli.main(["corpus", "--dry-run"])
    out, _ = capsys.readouterr()
    assert rc == cli.EXIT_OK
    assert "plan:" in out, "a human run should keep its narration on stdout"


def test_whatever_lands_on_stdout_under_json_parses(capsys):
    """The property that matters: `inkfloor ... --json > f` yields a parseable f, or nothing."""
    cli.main(["corpus", "--dry-run", "--json"])
    out, _ = capsys.readouterr()
    if out.strip():
        json.loads(out)
