from __future__ import annotations

from threading import Lock

from core.state import SessionState

_lock = Lock()
_sessions: dict[str, SessionState] = {}


def register(session: SessionState) -> None:
    with _lock:
        _sessions[session.session_id] = session


def unregister(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


def snapshot() -> list[dict]:
    with _lock:
        sessions = list(_sessions.values())
    return [_entry(s) for s in sessions]


def _entry(s: SessionState) -> dict:
    return {
        "session_id": s.session_id,
        "state": s.state.name.lower(),
        "topic": s.topic,
        "intent": s.intent,
        "engagement_score": round(s.engagement_score, 2),
        "total_user_turns": s.total_user_turns,
        "total_ai_turns": s.total_ai_turns,
        "last_activity": round(s.last_activity, 1),
    }
