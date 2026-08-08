from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.state import SessionState, DialogueState


class TestSessionState:
    def test_initial_state(self) -> None:
        s = SessionState(session_id="test-1")
        assert s.session_id == "test-1"
        assert s.state == DialogueState.IDLE
        assert s.transcript == []
        assert s.engagement_score == 0.5

    def test_add_user_turn(self) -> None:
        s = SessionState(session_id="test-1")
        s.add_user_turn("hello")
        assert len(s.transcript) == 1
        assert s.transcript[0].role == "user"
        assert s.transcript[0].content == "hello"
        assert s.total_user_turns == 1

    def test_add_ai_turn(self) -> None:
        s = SessionState(session_id="test-1")
        s.add_ai_turn("hi there")
        assert len(s.transcript) == 1
        assert s.transcript[0].role == "assistant"
        assert s.total_ai_turns == 1

    def test_context_messages(self) -> None:
        s = SessionState(session_id="test-1")
        s.add_user_turn("hello")
        s.add_ai_turn("hi")
        msgs = s.context_messages(max_turns=10)
        assert len(msgs) == 3  # system + user + assistant
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_set_state(self) -> None:
        s = SessionState(session_id="test-1")
        s.set_state(DialogueState.LISTENING)
        assert s.state == DialogueState.LISTENING
        s.set_state(DialogueState.PROCESSING)
        assert s.state == DialogueState.PROCESSING

    def test_engagement_updates(self) -> None:
        s = SessionState(session_id="test-1")
        assert s.engagement_score == 0.5
        for _ in range(5):
            s.add_user_turn("hello")
        assert s.engagement_score > 0.5

    def test_topic_intent_propagation(self) -> None:
        s = SessionState(session_id="test-1")
        assert s.topic == ""
        assert s.intent == "statement"

        ctx = SimpleNamespace(topic="latency optimization", intent="question")
        s.update(ctx)

        assert s.topic == "latency optimization"
        assert s.intent == "question"

    def test_update_handles_blank_ctx(self) -> None:
        s = SessionState(session_id="test-1")
        s.update(SimpleNamespace(topic="", intent=""))
        assert s.topic == ""
        assert s.intent == ""