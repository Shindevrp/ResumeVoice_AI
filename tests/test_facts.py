from __future__ import annotations

import asyncio

from core.pipeline import ConversationContext, StreamingPipeline
from modules.memory.facts import Fact, FactMemory
from modules.memory.session import SessionMemory
from tests.test_streaming import (
    FakeLLMText,
    FakeSTTText,
    FakeTTSStream,
    FakeVAD,
)


class FakeLLMWithGenerate:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def generate_stream(self, messages):
        if False:
            yield ""

    async def generate(self, messages):
        return self._reply


def _make_pipeline() -> StreamingPipeline:
    return StreamingPipeline(FakeSTTText(), FakeLLMText(), FakeTTSStream(), FakeVAD())


class TestFactExtraction:
    def test_name_and_age(self) -> None:
        fm = FactMemory()
        facts = fm.extract("My name is Alex and I am 30 years old")
        assert {f.key: f.value for f in facts} == {"name": "Alex", "age": "30"}

    def test_pet_with_and_without_name(self) -> None:
        fm = FactMemory()
        facts = fm.extract("I have a dog named Rex")
        assert ("pet:dog", "Rex") in {(f.key, f.value) for f in facts}

        fm2 = FactMemory()
        facts2 = fm2.extract("we have a cat")
        assert ("pet:cat", "cat") in {(f.key, f.value) for f in facts2}

    def test_diet_and_avoids(self) -> None:
        fm = FactMemory()
        facts = fm.extract("I'm a vegetarian and I don't eat meat")
        pairs = {(f.key, f.value) for f in facts}
        assert ("diet", "vegetarian") in pairs
        assert ("avoids:meat", "meat") in pairs

    def test_location_and_job(self) -> None:
        fm = FactMemory()
        facts = fm.extract("I live in Austin and I work as a teacher")
        pairs = {(f.key, f.value) for f in facts}
        assert ("location", "Austin") in pairs
        assert ("job", "a teacher") in pairs

    def test_family(self) -> None:
        fm = FactMemory()
        facts = fm.extract("my daughter is Emma and my son is 5")
        pairs = {(f.key, f.value) for f in facts}
        assert ("family:daughter", "Emma") in pairs
        assert ("family:son:age", "5") in pairs

    def test_no_false_positive_on_question(self) -> None:
        fm = FactMemory()
        assert fm.extract("What is my name again?") == []
        assert fm.extract("I have a question about the test") == []


class TestFactMemoryMerge:
    def test_latest_statement_wins(self) -> None:
        fm = FactMemory()
        fm.advance_turn()
        fm.add_all(fm.extract("I live in Austin"))
        fm.advance_turn()
        fm.add_all(fm.extract("I moved to Seattle"))
        assert fm.get("location").value == "Seattle"
        assert fm.get("location").source_turn == 2

    def test_to_block_and_empty(self) -> None:
        assert FactMemory().to_block() == ""
        fm = FactMemory()
        fm.add(Fact("name", "Alex", "identity", 1))
        fm.add(Fact("diet", "vegan", "preference", 1))
        block = fm.to_block()
        assert block.startswith("Facts about the user:")
        assert "- name: Alex" in block
        assert "- diet: vegan" in block

    def test_max_facts_cap(self) -> None:
        fm = FactMemory(max_facts=2)
        for i in range(5):
            fm.add(Fact(f"k{i}", str(i), "personal", i))
        block = fm.to_block()
        assert block.count("\n- ") == 2


class TestPipelineFacts:
    def test_facts_block_injected_into_messages(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            fm = FactMemory()
            fm.advance_turn()
            fm.add_all(fm.extract("My name is Alex"))
            ctx = ConversationContext()
            msgs = await p._build_messages(
                "hello", ctx, None, None, facts=fm
            )
            system = msgs[0]["content"]
            assert "Facts about the user:" in system
            assert "- name: Alex" in system

        asyncio.run(run())

    def test_regex_extraction_runs_inline_on_segment(self) -> None:
        async def run() -> None:
            class SttAlex:
                async def transcribe(self, audio_blob: bytes) -> str:
                    return "My name is Alex"

            p = StreamingPipeline(
                SttAlex(), FakeLLMText(), FakeTTSStream(), FakeVAD()
            )
            fm = FactMemory()
            p._facts["sess"] = fm
            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )
            assert fm.get("name") is not None
            assert fm.get("name").value == "Alex"

        asyncio.run(run())


class TestLLMFactExtraction:
    def test_llm_facts_added(self) -> None:
        async def run() -> None:
            p = StreamingPipeline(
                FakeSTTText(),
                FakeLLMWithGenerate('[{"key": "hobby", "value": "hiking", "confidence": 0.9}]'),
                FakeTTSStream(),
                FakeVAD(),
            )
            fm = FactMemory()
            fm.advance_turn()
            await p._extract_facts_llm("sess", "I love hiking", fm)
            f = fm.get("hobby")
            assert f is not None
            assert f.value == "hiking"
            assert f.source == "llm"

        asyncio.run(run())

    def test_llm_empty_and_fenced(self) -> None:
        async def run() -> None:
            p = StreamingPipeline(
                FakeSTTText(),
                FakeLLMWithGenerate("```json\n[]\n```"),
                FakeTTSStream(),
                FakeVAD(),
            )
            fm = FactMemory()
            fm.advance_turn()
            await p._extract_facts_llm("sess", "hello there", fm)
            assert len(fm) == 0

        asyncio.run(run())

    def test_schedule_skips_questions(self) -> None:
        async def run() -> None:
            p = StreamingPipeline(
                FakeSTTText(),
                FakeLLMWithGenerate("[]"),
                FakeTTSStream(),
                FakeVAD(),
            )
            fm = FactMemory()
            p._schedule_llm_facts("sess", "What time is it?", fm)
            assert "sess" not in p._last_fact_extract

        asyncio.run(run())

    def test_schedule_rate_limited(self) -> None:
        import time

        async def run() -> None:
            p = StreamingPipeline(
                FakeSTTText(),
                FakeLLMWithGenerate("[]"),
                FakeTTSStream(),
                FakeVAD(),
            )
            fm = FactMemory()
            p._last_fact_extract["sess"] = time.monotonic()
            p._schedule_llm_facts("sess", "I love hiking", fm)
            assert "sess" in p._last_fact_extract

        asyncio.run(run())


class TestMessageDedup:
    def test_context_duplicate_of_history_is_dropped(self) -> None:
        class FakeRetrieval:
            def retrieve_context(self, query, sm, top_k=3):
                return ["the weather is nice"]

        async def run() -> None:
            p = _make_pipeline()
            sm = SessionMemory()
            sm.add("user", "the weather is nice")
            msgs = await p._build_messages(
                "tell me more", ConversationContext(), sm, FakeRetrieval()
            )
            context_msgs = [
                m for m in msgs if m["content"].startswith("Relevant context")
            ]
            assert context_msgs == []

        asyncio.run(run())

    def test_context_novel_hit_is_injected(self) -> None:
        class FakeRetrieval:
            def retrieve_context(self, query, sm, top_k=3):
                return ["you once mentioned a boat trip"]

        async def run() -> None:
            p = _make_pipeline()
            sm = SessionMemory()
            sm.add("user", "the weather is nice")
            msgs = await p._build_messages(
                "tell me more", ConversationContext(), sm, FakeRetrieval()
            )
            context_msgs = [
                m for m in msgs if m["content"].startswith("Relevant context")
            ]
            assert len(context_msgs) == 1
            assert "boat trip" in context_msgs[0]["content"]

        asyncio.run(run())
