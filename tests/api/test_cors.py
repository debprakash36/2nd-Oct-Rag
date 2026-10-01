"""CORS configuration for the browser client (FR-23, FR-29).

The web UI is served from a different origin than the API, so without these headers
every request from the chat page is blocked by the browser. These tests pin the
behaviour that matters: the local dev origin works, an unrelated origin does not.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings

UI_ORIGIN = "http://localhost:3000"


def _get(client: TestClient, origin: str) -> str | None:
    # `/health` rather than `/conversations`: it needs no documents, so the test does
    # not depend on a populated corpus just to check a CORS header.
    response = client.get("/health", headers={"Origin": origin})
    return response.headers.get("access-control-allow-origin")


def test_preflight_from_the_web_ui_is_allowed(client: TestClient) -> None:
    response = client.options(
        "/chat/stream",
        headers={
            "Origin": UI_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == UI_ORIGIN
    # A wildcard would allow every site to call the API on a user's behalf.
    assert response.headers["access-control-allow-origin"] != "*"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "PUT" in response.headers["access-control-allow-methods"], "feedback votes use PUT"


def test_request_from_the_web_ui_carries_the_allow_origin_header(
    client: TestClient,
) -> None:
    assert _get(client, UI_ORIGIN) == UI_ORIGIN


def test_unknown_origin_is_not_allowed(client: TestClient) -> None:
    """Fail closed: an unconfigured deployment must not serve every origin."""
    assert _get(client, "https://evil.example") is None


def test_credentials_are_not_allowed(client: TestClient) -> None:
    """Nothing here is cookie-authenticated, and a wildcard plus credentials is a
    vulnerability the moment auth lands."""
    response = client.get("/health", headers={"Origin": UI_ORIGIN})

    assert response.headers.get("access-control-allow-credentials") != "true"


def test_origin_list_ignores_whitespace_and_blanks() -> None:
    settings = Settings(cors_allow_origins="http://a.test, ,http://b.test ,")

    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_origin_list_defaults_to_local_dev_only() -> None:
    settings = Settings()

    assert settings.cors_origin_list == [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]