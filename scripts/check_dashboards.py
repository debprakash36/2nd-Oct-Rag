"""Check that every dashboard SQL query is executable against the real schema.

The dashboards target PostgreSQL, but the schema in `app/db/models.py` is shared, so
the queries can be run against the SQLite load database with the Grafana macros and a
few dialect-specific functions substituted. This catches the failure mode that
matters for a dashboard: a column name that does not exist, or a typo in an
aggregate, which otherwise shows up as an empty panel in production.

Grafana macros (`$__timeGroupAlias`, `$__timeFilter`) and `percentile_cont` are
rewritten; everything else - table names, column names, joins, filters - is executed
verbatim, which is the part worth checking.

Run: .venv\\Scripts\\python.exe scripts/check_dashboards.py
"""

from __future__ import annotations

import glob
import json
import re
import sqlite3
import sys
from pathlib import Path

# The Windows console here is cp1252 and raises on the arrows used in panel titles.
# Reconfigure rather than strip the characters, so the output still identifies the
# panel that failed.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent


def adapt(sql: str) -> str:
    """Rewrite Grafana/Postgres-isms into something SQLite will execute."""
    # Column may be qualified (`c.created_at`) for a joined table.
    out = re.sub(
        r"\$__timeGroupAlias\(([\w.]+), '([^']+)'\)",
        r"strftime('%Y-%m-%dT%H:%M', \1)",
        sql,
    )
    out = re.sub(r"\$__timeFilter\(([\w.]+)\)", r"\1 IS NOT NULL", out)
    # SQLite has no ordered-set aggregate; a real p95 is not needed to prove the
    # query references real columns, so the whole ordered set collapses to MAX.
    out = re.sub(
        r"percentile_cont\([\d.]+\) WITHIN GROUP \(ORDER BY (.*?)\)(?=\s*(?:AS|,|FROM|$))",
        r"MAX(\1)",
        out,
        flags=re.IGNORECASE,
    )
    out = out.replace(
        "EXTRACT(EPOCH FROM (indexed_at - uploaded_at))",
        "((julianday(indexed_at) - julianday(uploaded_at)) * 86400.0)",
    )
    return out


def main() -> int:
    db_files = glob.glob(str(REPO_ROOT / ".load_cache" / "load-*.db"))
    if not db_files:
        print("no load cache; run `make load-test` once to populate it", file=sys.stderr)
        return 2

    connection = sqlite3.connect(db_files[0])
    total = failed = 0

    for path in sorted(glob.glob(str(REPO_ROOT / "ops" / "dashboards" / "*.json"))):
        dashboard = json.loads(Path(path).read_text(encoding="utf-8"))
        name = Path(path).name
        print(f"{name} - {dashboard['title']}")
        for panel in dashboard["panels"]:
            for target in panel.get("targets", []):
                if "rawSql" not in target:
                    continue
                total += 1
                label = f"  {panel.get('title', panel['type'])[:44]}"
                try:
                    connection.execute(adapt(target["rawSql"])).fetchall()
                except sqlite3.Error as exc:
                    failed += 1
                    # ASCII only: Windows consoles here are cp1252 and will raise on
                    # the arrows used in some panel descriptions.
                    print(f"{label}\n      FAIL: {exc}")
                else:
                    print(f"{label}\n      ok")

    print(f"\n{total - failed}/{total} queries executable")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
