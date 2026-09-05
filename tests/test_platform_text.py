from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.pipeline import PipelineEvent, StreamingPipeline
from modules.memory.session import SessionMemory


class FakeSTT:
    async def transcribe(self, audio_blob: bytes) -> str:
        return ""


class FakeVAD:
    sample_rate = 16000

    def is_speech(self, chunk: bytes) -> bool:
        return False

    def reset(self) -> None:
        pass


class FakeLLMText:
    async def generate_stream(self, messages):
        for token in ["Hello ", "there."]:
            yield token


class FakeTTS:
    sample_rate = 16000

    def __init__(self) -> None:
        self.synthesized: list[str] = []

    async def synthesize_stream(self, text_chunks, prosody=None):
        buf = ""
        async for chunk in text_chunks:
            buf += chunk
        if buf:
            self.synthesized.append(buf)
            yield b"\x00\x00"

    async def synthesize(self, text: str, prosody=None) -> bytes:
        return b""


class BlockingLLM:
    def __init__(self) -> None:
        self.started = 0
        self.cancelled = 0

    async def generate_stream(self, messages):
        self.started += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if False:
            yield ""


class FakeRetrieval:
    def warm_up(self) -> None:
        pass

    def add_to_long_term(self, text: str, topic: str | None = None) -> None:
        pass

    def retrieve_context(
        self,
        query: str,
        session_memory: SessionMemory,
        top_k: int = 3,
        topic: str | None = None,
    ) -> list[str]:
        return []

    def retrieve_context_with_topics(
        self,
        query: str,
        session_memory: SessionMemory,
        top_k: int = 3,
        topic: str | None = None,
    ) -> list[tuple[str, str | None]]:
        return []


def _pipeline() -> StreamingPipeline:
    return StreamingPipeline(FakeSTT(), FakeLLMText(), FakeTTS(), FakeVAD())


class TestPushText:
    def test_empty_text_raises(self) -> None:
        async def run() -> None:
            p = _pipeline()
            with pytest.raises(ValueError):
                await p.push_text("   ", "sess")

        asyncio.run(run())

    def test_push_text_streams_full_response(self) -> None:
        async def run() -> None:
            p = _pipeline()
            memory = SessionMemory()
            p.register_session("sess", memory, FakeRetrieval())
            await p.push_text("hello world", "sess")

            msgs: list = []
            while not p._output_queue.empty():
                msgs.append(p._output_queue.get_nowait())
            events = [m.event for m in msgs]

            assert PipelineEvent.FINAL_TRANSCRIPT in events
            assert PipelineEvent.LLM_TOKEN in events
            assert PipelineEvent.LLM_DONE in events
            assert PipelineEvent.TTS_CHUNK in events
            assert PipelineEvent.TTS_DONE in events

            tokens = "".join(
                str(m.data) for m in msgs if m.event == PipelineEvent.LLM_TOKEN
            )
            assert tokens == "Hello there."
            assert p.tts.synthesized == ["Hello there."]

            roles = [e.role for e in memory.get_history(10)]
            assert roles == ["user", "assistant"]

        asyncio.run(run())

    def test_push_text_barges_in_running_turn(self) -> None:
        async def run() -> None:
            llm = BlockingLLM()
            p = StreamingPipeline(FakeSTT(), llm, FakeTTS(), FakeVAD())
            p.turn_timing.compute_delay = lambda *args, **kwargs: 0.0

            first = asyncio.create_task(p.push_text("hi", "sess"))
            await asyncio.sleep(0.05)
            assert llm.started == 1
            assert p._current_tasks.get("sess") is first

            second = asyncio.create_task(p.push_text("oh", "sess"))
            with pytest.raises(asyncio.CancelledError):
                await first
            for _ in range(5):
                await asyncio.sleep(0.01)
            assert llm.cancelled == 1
            assert llm.started == 2

            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            assert "sess" not in p._current_tasks

        asyncio.run(run())


class _FakePlatformPipeline:
    def __init__(self) -> None:
        self.text_calls: list[tuple[str, str]] = []

    def register_session(self, session_id: str, memory, retrieval) -> None:
        pass

    def unregister_session(self, session_id: str) -> None:
        pass

    def context(self, session_id: str):
        return object()

    async def signal_interrupt(self, session_id: str) -> None:
        pass

    async def output_stream(self):
        if False:
            yield None

    async def push_text(self, text: str, session_id: str = "default") -> None:
        self.text_calls.append((session_id, text))


def _client(pipeline: object) -> TestClient:
    from app.routes.webrtc import router

    app = FastAPI()
    app.state.pipeline = pipeline
    app.include_router(router)
    return TestClient(app)


class TestWebRTCText:
    def test_text_message_calls_push_text(self) -> None:
        fp = _FakePlatformPipeline()
        with _client(fp).websocket_connect("/ws/signal") as ws:
            ws.send_json({"type": "text", "text": "Hi there"})
            deadline = time.monotonic() + 3
            while not fp.text_calls and time.monotonic() < deadline:
                time.sleep(0.01)
        assert fp.text_calls
        assert fp.text_calls[0][1] == "Hi there"

    def test_empty_text_returns_error(self) -> None:
        fp = _FakePlatformPipeline()
        with _client(fp).websocket_connect("/ws/signal") as ws:
            ws.send_json({"type": "text", "text": "   "})
            msg = ws.receive_json()
            while msg.get("type") != "error":
                msg = ws.receive_json()
        assert msg["text"] == "empty text message"
        assert fp.text_calls == []

    def test_oversized_text_returns_error(self) -> None:
        fp = _FakePlatformPipeline()
        with _client(fp).websocket_connect("/ws/signal") as ws:
            ws.send_json({"type": "text", "text": "x" * 2001})
            msg = ws.receive_json()
            while msg.get("type") != "error":
                msg = ws.receive_json()
        assert "exceeds" in msg["text"]


class TestWebRTCConfig:
    def test_config_returns_default_stun(self, monkeypatch) -> None:
        monkeypatch.delenv("RESUMEVOICE_TURN_URL", raising=False)
        monkeypatch.setenv("RESUMEVOICE_STUN_URL", "stun:stun.example.com:19302")
        from app.routes.webrtc import build_ice_servers

        servers = build_ice_servers()
        assert servers and servers[0]["urls"] == "stun:stun.example.com:19302"
        assert len(servers) == 1

    def test_config_adds_turn_when_configured(self, monkeypatch) -> None:
        monkeypatch.setenv("RESUMEVOICE_TURN_URL", "turn:turn.example.com:3478")
        monkeypatch.setenv("RESUMEVOICE_TURN_USERNAME", "resumevoice")
        monkeypatch.setenv("RESUMEVOICE_TURN_CREDENTIAL", "secret")
        from app.routes.webrtc import build_ice_servers

        servers = build_ice_servers()
        assert len(servers) == 2
        assert servers[1]["urls"] == "turn:turn.example.com:3478"
        assert servers[1]["username"] == "resumevoice"
        assert servers[1]["credential"] == "secret"

    def test_candidate_without_mline_gets_defaults(self) -> None:
        from app.routes.webrtc import _candidate_from_payload

        c = _candidate_from_payload({"ip": "1.2.3.4", "port": 5000})
        assert c.sdpMid == "0"
        assert c.sdpMLineIndex == 0

    def test_candidate_preserves_client_mline(self) -> None:
        from app.routes.webrtc import _candidate_from_payload

        c = _candidate_from_payload(
            {
                "ip": "1.2.3.4",
                "port": 5000,
                "sdpMid": "audio-mid",
                "sdpMLineIndex": 1,
            }
        )
        assert c.sdpMid == "audio-mid"
        assert c.sdpMLineIndex == 1

    def test_candidate_to_payload_roundtrip(self) -> None:
        from app.routes.webrtc import _candidate_from_payload, _candidate_to_payload

        c = _candidate_from_payload(
            {
                "component": 1,
                "foundation": "abc",
                "ip": "192.168.1.10",
                "port": 12345,
                "priority": 2130706431,
                "protocol": "udp",
                "type": "host",
                "sdpMid": "0",
                "sdpMLineIndex": 0,
            }
        )
        payload = _candidate_to_payload(c)
        assert payload["candidate"].startswith("candidate:")
        assert "192.168.1.10 12345 typ host" in payload["candidate"]
        assert payload["sdpMid"] == "0"
        assert payload["sdpMLineIndex"] == 0
        reparse = _candidate_from_payload(payload)
        assert reparse.ip == "192.168.1.10"
        assert reparse.port == 12345

    def test_candidate_to_payload_defaults_missing_mline(self) -> None:
        from app.routes.webrtc import _candidate_from_payload, _candidate_to_payload

        c = _candidate_from_payload({"ip": "1.2.3.4", "port": 5000})
        payload = _candidate_to_payload(c)
        assert payload["sdpMid"] == "0"
        assert payload["sdpMLineIndex"] == 0

    def test_candidate_browser_format_parsed(self) -> None:
        """Browsers send the full SDP line under the "candidate" key."""
        from app.routes.webrtc import _candidate_from_payload

        c = _candidate_from_payload(
            {
                "candidate": (
                    "candidate:3853026675 1 udp 2122260223 192.168.1.97 48867 "
                    "typ host generation 0 ufrag kqGe network-id 4 network-cost 10"
                ),
                "sdpMid": "0",
                "sdpMLineIndex": 0,
                "usernameFragment": "kqGe",
            }
        )
        assert c.ip == "192.168.1.97"
        assert c.port == 48867
        assert c.foundation == "3853026675"
        assert c.type == "host"

class TestOutputBroadcast:
    def test_events_reach_every_subscriber(self) -> None:
        """Two concurrent sessions must each receive the full event stream.

        Regression: consumers raced on one shared queue, so an event for
        session A could be consumed-and-dropped by session B's iterator —
        clients then missed llm_done/tts_done whenever another client was
        connected.
        """
        async def run() -> None:
            p = StreamingPipeline(FakeSTT(), FakeLLMText(), FakeTTS(), FakeVAD())
            await p.start()
            p._session_outputs.setdefault("A", asyncio.Queue(8))
            p._session_outputs.setdefault("B", asyncio.Queue(8))

            got_a: list[str] = []
            got_b: list[str] = []

            async def consume(sid: str, into: list[str]) -> None:
                async for msg in p.output_stream(sid):
                    into.append(msg.event.name)

            task_a = asyncio.create_task(consume("A", got_a))
            task_b = asyncio.create_task(consume("B", got_b))
            await asyncio.sleep(0.05)

            for event, data in [
                (PipelineEvent.FINAL_TRANSCRIPT, "hi"),
                (PipelineEvent.LLM_TOKEN, "Hi"),
                (PipelineEvent.LLM_DONE, "Hi"),
                (PipelineEvent.TTS_DONE, None),
            ]:
                await p._emit(event, data, "A")
            await asyncio.sleep(0.1)

            for event, data in [
                (PipelineEvent.INTERRUPT, None),
            ]:
                await p._emit(event, data, "B")
            await asyncio.sleep(0.1)

            for name in ("FINAL_TRANSCRIPT", "LLM_TOKEN", "LLM_DONE", "TTS_DONE"):
                assert name in got_a, f"A missed {name}: {got_a}"
                assert name in got_b, f"B missed {name}: {got_b}"
            assert "INTERRUPT" in got_b
            assert "INTERRUPT" in got_a

            task_a.cancel()
            task_b.cancel()
            await asyncio.gather(task_a, task_b, return_exceptions=True)
            await p.stop()

        asyncio.run(run())

    def test_late_subscriber_backfills_buffered_events(self) -> None:
        async def run() -> None:
            p = StreamingPipeline(FakeSTT(), FakeLLMText(), FakeTTS(), FakeVAD())
            p._running = True
            await p._emit(PipelineEvent.LLM_DONE, "buffered", "sess")

            seen: list[str] = []

            async def consume() -> None:
                async for msg in p.output_stream("sess"):
                    seen.append(msg.event.name)

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.1)
            assert "LLM_DONE" in seen, f"missing buffered event: {seen}"
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run())
