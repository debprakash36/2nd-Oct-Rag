"""Shared-secret gate. Open when API_TOKEN is empty; closed when it is set."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings


def test_open_when_no_token_is_configured(client: TestClient) -> None:
    assert client.get("/auth/status").json() == {"required": False}
    assert client.get("/conversations").status_code == 200


def test_rejects_a_missing_or_wrong_token(client: TestClient, settings_env: Settings) -> None:
    settings_env.api_token = "correct-token"

    assert client.get("/auth/status").json() == {"required": True}
    assert client.get("/health").status_code == 200

    blocked = client.get("/conversations")
    assert blocked.status_code == 401
    assert blocked.json()["detail"] == "Sign in required."

    wrong = client.get("/conversations", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401

    ok = client.get("/conversations", headers={"Authorization": "Bearer correct-token"})
    assert ok.status_code == 200


def test_login_accepts_only_the_configured_token(
    client: TestClient, settings_env: Settings
) -> None:
    settings_env.api_token = "correct-token"

    rejected = client.post("/auth/login", json={"token": "nope"})
    assert rejected.status_code == 401
    accepted = client.post("/auth/login", json={"token": "correct-token"})
    assert accepted.status_code == 200
    assert accepted.json() == {"ok": True}
