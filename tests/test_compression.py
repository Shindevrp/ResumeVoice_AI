from __future__ import annotations

import asyncio

from core.pipeline import ConversationContext
from modules.memory.compression import ContextCompressor
from modules.memory.session import SessionMemory
from tests.test_streaming import FakeLLMText, _make_pipeline


class TestFoldSummary:
    def test_evicts_only_batch_entries(self) -> None:
        sm = SessionMemory(max_turns=10)
        for i in range(3):
            sm.add("user", f"u{i}")
            sm.add("assistant", f"a{i}")
        batch = sm.entries_snapshot_oldest(4)
        sm.fold_summary("early turns covered", batch)
        assert "early turns covered" in sm.summary
        assert len(sm.entries) == 2
        assert all(e.content not in {"u0", "a0", "u1", "a1"} for e in sm.entries)
        assert any(e.content == "u2" for e in sm.entries)

    def test_summary_is_capped(self) -> None:
        sm = SessionMemory(max_summary_chars=20)
        sm.add("user", "x")
        sm.fold_summary("a" * 50, list(sm.entries))
        assert len(sm.summary) <= 20

    def test_summary_accumulates_across_folds(self) -> None:
        sm = SessionMemory(max_summary_chars=200)
        for i in range(2):
            sm.add("user", f"q{i}")
            sm.add("assistant", f"a{i}")
        sm.fold_summary("first part.", sm.entries_snapshot_oldest(2))
        sm.fold_summary("second part.", sm.entries_snapshot_oldest(2))
        assert "first part." in sm.summary
        assert "second part." in sm.summary

    def test_token_estimate_counts_summary(self) -> None:
        sm = SessionMemory()
        sm.add("user", "hello")
        before = sm.token_estimate()
        sm.summary = "a b c d e"
        assert sm.token_estimate() == before + 5


class TestContextCompressor:
    def test_build_prompt_embeds_summary_and_turns(self) -> None:
        sm = SessionMemory()
        sm.add("user", "I want a boat")
        c = ContextCompressor()
        prompt = c.build_prompt("old summary", list(sm.entries))
        assert prompt[0]["role"] == "system"
        assert "old summary" in prompt[1]["content"]
        assert "I want a boat" in prompt[1]["content"]

    def test_compress_streams_llm_output(self) -> None:
        class FakeSummLLM:
            def __init__(self) -> None:
                self.prompt: list[dict[str, str]] | None = None

            async def generate_stream(self, messages):
                self.prompt = messages
                for tok in ["User ", "prefers ", "sailing."]:
                    yield tok

        async def run() -> None:
            sm = SessionMemory()
            sm.add("user", "I like sailing")
            c = ContextCompressor()
            llm = FakeSummLLM()
            out = await c.compress(llm, "existing", list(sm.entries))
            assert out == "User prefers sailing."
            assert "existing" in llm.prompt[1]["content"]

        asyncio.run(run())

    def test_compress_caps_output(self) -> None:
        class FakeSummLLM:
            async def generate_stream(self, messages):
                for tok in ["aaaa", "bbbb", "cccc", "dddd", "eeee"]:
                    yield tok

        async def run() -> None:
            sm = SessionMemory()
            sm.add("user", "x")
            c = ContextCompressor(max_chars=10)
            out = await c.compress(FakeSummLLM(), "", list(sm.entries))
            assert len(out) <= 10

        asyncio.run(run())


class TestPipelineCompression:
    def test_trigger_folds_summary_in_background(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.llm = FakeLLMText()
            sm = SessionMemory(max_turns=10)
            p._memories["sess"] = sm
            for i in range(6):
                sm.add("user", f"question {i} about the weather")
                sm.add(
                    "assistant",
                    f"here is the answer number {i} about the weather "
                    f"in quite some detail " * 4,
                )
            p.compress_at_tokens = 100
            p.compress_batch = 6
            assert sm.token_estimate() > 100
            p._maybe_compress("sess")
            await asyncio.sleep(0.05)
            assert sm.summary
            assert len(sm.entries) == 4

        asyncio.run(run())

    def test_no_repeat_compression_while_pending(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.llm = FakeLLMText()
            sm = SessionMemory(max_turns=10)
            p._memories["sess"] = sm
            for i in range(6):
                sm.add("user", f"question {i} about the weather")
                sm.add(
                    "assistant",
                    f"answer {i} about the weather with extra words " * 4,
                )
            p.compress_at_tokens = 100
            p.compress_batch = 6
            p._maybe_compress("sess")
            assert "sess" in p._compression_tasks
            p._maybe_compress("sess")
            await asyncio.sleep(0.05)
            assert "sess" not in p._compression_tasks
            assert sm.summary

        asyncio.run(run())

    def test_build_messages_injects_summary(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            sm = SessionMemory()
            sm.summary = "User likes sailing and wants a dinghy."
            ctx = ConversationContext()
            ctx.turn_count = 5
            msgs = await p._build_messages("what about hulls?", ctx, sm, None)
            joined = "\n".join(m["content"] for m in msgs)
            assert "Conversation summary so far" in joined
            assert "sailing" in joined

        asyncio.run(run())
