from __future__ import annotations

from fastapi import APIRouter

from app.session_registry import snapshot

router = APIRouter(tags=["sessions"])


@router.get("/sessions")
async def list_sessions() -> dict:
    return {"sessions": snapshot()}
