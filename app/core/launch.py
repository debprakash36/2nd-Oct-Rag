"""Launch readiness. Distinguishes a working local demo from a public deploy.

A local `/health` of 200 does not mean production is safe: sqlite, fake
providers, and localhost CORS are all fine on a laptop and fatal on a public
URL. This module grades those separately so an operator cannot confuse them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings


@dataclass
class LaunchCheck:
    name: str
    state: str  # "pass" | "fail" | "warn" | "blocked"
    detail: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def assess(settings: Settings, session: Session | None = None) -> list[LaunchCheck]:
    """Grade the current process for the environment it claims to be in."""
    deployed = settings.environment in {"production", "staging"}
    checks: list[LaunchCheck] = []

    try:
        settings.validate_production()
        if deployed:
            checks.append(
                LaunchCheck(
                    "production guards",
                    "pass",
                    "validate_production() accepted this config.",
                )
            )
        else:
            checks.append(
                LaunchCheck(
                    "environment",
                    "pass",
                    f"{settings.environment}: sqlite and fake providers are allowed. "
                    "This is a local demo, not a public deploy.",
                )
            )
    except RuntimeError as exc:
        checks.append(LaunchCheck("production guards", "fail", str(exc)))

    if deployed and (
        "localhost" in settings.cors_allow_origins
        or "127.0.0.1" in settings.cors_allow_origins
    ) and "https://" not in settings.cors_allow_origins:
        checks.append(
            LaunchCheck(
                "CORS",
                "fail",
                "CORS_ALLOW_ORIGINS is still localhost-only. The browser UI on "
                "the public host will be blocked. Set it to the web service origin.",
            )
        )
    elif not settings.cors_origin_list:
        checks.append(
            LaunchCheck("CORS", "fail", "CORS_ALLOW_ORIGINS is empty; the UI cannot call the API.")
        )
    else:
        checks.append(
            LaunchCheck(
                "CORS",
                "pass",
                f"{len(settings.cors_origin_list)} origin(s) allowed.",
            )
        )

    if not settings.api_token:
        checks.append(
            LaunchCheck(
                "API_TOKEN",
                "fail" if deployed else "warn",
                "Empty token leaves every endpoint open, including document delete.",
            )
        )
    else:
        checks.append(LaunchCheck("API_TOKEN", "pass", "Set (value not printed)."))

    if session is not None:
        try:
            from app.retrieval.vector_store import (
                build_vector_store,
                measure_divergence,
            )

            divergence = measure_divergence(
                session, build_vector_store(session, settings), dim=settings.embedding_dim
            )
            if divergence.is_malformed or divergence.is_empty:
                checks.append(
                    LaunchCheck(
                        "corpus",
                        "fail",
                        divergence.summary()
                        + ("; re-embed before serving" if divergence.is_malformed else ""),
                    )
                )
            else:
                checks.append(LaunchCheck("corpus", "pass", divergence.summary()))
        except (RuntimeError, ValueError, SQLAlchemyError) as exc:
            checks.append(
                LaunchCheck("corpus", "fail", f"{type(exc).__name__}: could not inspect the index")
            )

    checks.append(
        LaunchCheck(
            "Phase 6 traffic gate",
            "blocked",
            "UNMEASURED until real users produce enough distinct queries. "
            "Do not treat a local demo as a cleared PRD section 8 gate.",
        )
    )
    return checks


def overall_ok(checks: list[LaunchCheck], *, allow_local: bool = True) -> bool:
    """FAIL blocks launch. WARN and BLOCKED do not: they are known gaps."""
    del allow_local
    return all(c.state != "fail" for c in checks)
