"""Report whether this process is a local demo or a public-ready deploy.

Usage:
    python scripts/check_launch.py
    python scripts/check_launch.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.launch import assess, overall_ok  # noqa: E402
from app.db.session import get_session_factory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    session = get_session_factory(settings)()
    try:
        checks = assess(settings, session)
    finally:
        session.close()

    local = settings.environment in {"local", "test"}
    ok = overall_ok(checks, allow_local=local)

    if args.json:
        print(
            json.dumps(
                {
                    "environment": settings.environment,
                    "ok": ok,
                    "checks": [c.as_dict() for c in checks],
                },
                indent=2,
            )
        )
    else:
        print(f"launch check ({settings.environment})")
        print("=" * 60)
        for c in checks:
            print(f"  {c.state.upper():<8} {c.name}: {c.detail}")
        print()
        if local and ok:
            print("  Local demo: OK. Do not bind this SQLite process to a public URL.")
        elif ok:
            print("  Production config: OK. Phase 6 traffic is still a separate gate.")
        else:
            print("  Not ready. Fix FAIL rows before exposing the service.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
