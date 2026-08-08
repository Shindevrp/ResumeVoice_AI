from __future__ import annotations

import asyncio

from core.pipeline import ConversationContext, StreamingPipeline
from tests.test_streaming import (
    FakeSTTText,
    FakeTTSStream,
    FakeVAD,
    _is_label_call,
)


class TrackingLLM:
    def __init__(self) -> None:
        self.messages_log: list[list[dict[str, str]]] = []

    async def generate_stream(self, messages):
        if _is_label_call(messages):
            for token in ["Latency ", "tuning"]:
                yield token
            return
        self.messages_log.append(messages)
        for token in ["Hello ", "world."]:
            yield token


class _Transcribe:
    def __init__(self, text: str) -> None:
        self._text = text

    async def transcribe(self, audio_blob: bytes) -> str:
        return self._text


def _pipeline(text: str) -> tuple[StreamingPipeline, TrackingLLM]:
    llm = TrackingLLM()
    p = StreamingPipeline(_Transcribe(text), llm, FakeTTSStream(), FakeVAD())
    return p, llm


class TestPipelineTopic:
    def test_topic_injected_into_system_prompt(self) -> None:
        async def run() -> None:
            p, _ = _pipeline("planning a birthday party")
            ctx = ConversationContext()
            ctx.turn_count = 2
            ctx.topic = "planning a birthday party"
            msgs = await p._build_messages("more about it", ctx, None, None)
            assert "Current topic:" in msgs[0]["content"]
            assert "birthday" in msgs[0]["content"]

        asyncio.run(run())

    def test_topic_updated_after_segment(self) -> None:
        async def run() -> None:
            p, _ = _pipeline("let us plan a birthday party")
            ctx = ConversationContext()
            await p._process_speech_segment(b"\x00" * 1600, "sess", ctx)
            assert ctx.topic_shift is False
            assert "birthday" in ctx.topic

        asyncio.run(run())


class TestPipelineIntent:
    def test_correction_blocks_speculative_reuse(self) -> None:
        async def run() -> None:
            p, llm = _pipeline("no, I meant the red car")
            ctx = ConversationContext()
            ctx.last_partial_transcript = "no, I meant"
            await p._process_speech_segment(b"\x00" * 1600, "sess", ctx)
            assert ctx.intent == "correction"
            assert len(llm.messages_log) >= 2
            assert "corrected you" in str(llm.messages_log[-1])

        asyncio.run(run())

    def test_non_correction_reuses_speculation(self) -> None:
        async def run() -> None:
            p, llm = _pipeline("tell me about space")
            ctx = ConversationContext()
            ctx.last_partial_transcript = "tell me about"
            await p._process_speech_segment(b"\x00" * 1600, "sess", ctx)
            assert ctx.intent == "command"
            assert len(llm.messages_log) == 1

        asyncio.run(run())

    def test_intent_set_on_segment(self) -> None:
        async def run() -> None:
            p, _ = _pipeline("what time is it?")
            ctx = ConversationContext()
            await p._process_speech_segment(b"\x00" * 1600, "sess", ctx)
            assert ctx.intent == "question"

        asyncio.run(run())
