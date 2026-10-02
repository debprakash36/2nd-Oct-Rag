"""Login probe. The token itself is checked by middleware on every other route."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.auth import token_matches
from app.core.config import Settings, get_settings
from app.core.errors import AppError

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    token: str = Field(min_length=1)


@router.get("/status")
def auth_status(settings: Settings = Depends(get_settings)) -> dict[str, bool]:
    """Whether the UI should ask for a token. Public: it reveals no secret."""
    return {"required": bool(settings.api_token)}


@router.post("/login")
def login(body: LoginRequest, settings: Settings = Depends(get_settings)) -> dict[str, bool]:
    """Confirm a token before the browser stores it.

    Public so the login call itself is not rejected by the gate. A wrong token
    is a 401 with the same message as a missing one, so the two are not
    distinguishable to a caller fishing for which check failed.
    """
    if settings.api_token and not token_matches(body.token, settings.api_token):
        raise AppError(
            "rejected login",
            user_message="That sign-in token was not accepted.",
            status_code=401,
        )
    return {"ok": True}
