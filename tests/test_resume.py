from __future__ import annotations

import asyncio

from core.pipeline import ConversationContext, StreamingPipeline
from modules.dialogue.resume import (
    ResumeData,
    _build,
    _split_sections,
    load_resume_data,
)
from modules.memory.session import SessionMemory
from tests.test_streaming import FakeLLM, FakeSTT, FakeTTS, FakeVAD


class TestResumeParsing:
    def test_load_default_resume(self) -> None:
        resume = load_resume_data()
        assert resume is not None
        assert "VINAYAK" in resume.name
        assert "AI / ML Engineer" in resume.headline

    def test_sections_are_split(self) -> None:
        resume = load_resume_data()
        assert resume is not None
        titles = [title for title, _ in resume.sections]
        assert "PROFESSIONAL EXPERIENCE" in titles
        assert "TECHNICAL SKILLS" in titles
        assert "EDUCATION" in titles
        assert "PROFESSIONAL SUMMARY" in titles

    def test_get_section(self) -> None:
        resume = load_resume_data()
        assert resume is not None
        summary = resume.get_section("PROFESSIONAL SUMMARY")
        assert "Nabhe" in summary
        assert "IIIT Hyderabad" in summary

    def test_split_sections_header_ordering(self) -> None:
        text = (
            "NAME HERE\nHeadline line\n\n"
            "PROFESSIONAL SUMMARY\nFirst section body.\n\n"
            "TECHNICAL SKILLS\nPython, PyTorch\n\n"
            "EDUCATION\nMS Data Science\n"
        )
        header_block, sections = _split_sections(text)
        assert header_block == "NAME HERE\nHeadline line"
        assert [t for t, _ in sections] == [
            "PROFESSIONAL SUMMARY",
            "TECHNICAL SKILLS",
            "EDUCATION",
        ]

    def test_build_extracts_header(self) -> None:
        data = _build(
            "VINAYAK SHINDE\nEngineer\ncontact\n\nPROFESSIONAL SUMMARY\nHello"
        )
        assert data is not None
        assert data.name == "VINAYAK SHINDE"
        assert data.headline == "Engineer"
        assert data.summary == "Hello"


class TestResumePrompt:
    def test_prompt_block_first_person(self) -> None:
        resume = load_resume_data()
        assert resume is not None
        block = resume.to_prompt_block()
        assert "FIRST PERSON" in block
        assert "VINAYAK" in block
        assert "Candidate profile" in block
        assert "Nabhe" in block

    def test_retrieval_sections_skip_summary(self) -> None:
        resume = load_resume_data()
        assert resume is not None
        chunks = resume.retrieval_sections()
        assert chunks
        assert not any(c.startswith("PROFESSIONAL SUMMARY") for c in chunks)
        assert any(c.startswith("TECHNICAL SKILLS") for c in chunks)


class TestResumePipeline:
    def _pipeline(self, resume: ResumeData | None) -> StreamingPipeline:
        return StreamingPipeline(
            FakeSTT(), FakeLLM(), FakeTTS(), FakeVAD(), resume=resume
        )

    def test_resume_block_injected_into_messages(self) -> None:
        resume = load_resume_data()
        assert resume is not None

        async def run() -> None:
            p = self._pipeline(resume)
            msgs = await p._build_messages(
                "Tell me about yourself", ConversationContext(), None, None
            )
            assert "Candidate profile" in msgs[0]["content"]

        asyncio.run(run())

    def test_no_resume_block_when_unset(self) -> None:
        async def run() -> None:
            p = self._pipeline(None)
            msgs = await p._build_messages("hi", ConversationContext(), None, None)
            assert "Candidate profile" not in msgs[0]["content"]

        asyncio.run(run())

    def test_register_session_seeds_retrieval(self) -> None:
        resume = load_resume_data()
        assert resume is not None

        class FakeRetrieval:
            def __init__(self) -> None:
                self.added: list[tuple[str, str | None]] = []

            def warm_up(self) -> None:
                pass

            def add_to_long_term(self, text: str, topic: str | None = None) -> None:
                self.added.append((text, topic))

        async def run() -> None:
            p = self._pipeline(resume)
            fr = FakeRetrieval()
            p.register_session("sess", SessionMemory(), fr)
            for _ in range(50):
                if len(fr.added) >= len(resume.retrieval_sections()):
                    break
                await asyncio.sleep(0.01)
            assert len(fr.added) == len(resume.retrieval_sections())
            assert all(topic == "resume" for _, topic in fr.added)

        asyncio.run(run())

    def test_seed_resume_sync_is_deterministic(self) -> None:
        resume = load_resume_data()
        assert resume is not None

        class FakeRetrieval:
            def __init__(self) -> None:
                self.added = []

            def add_to_long_term(self, text, topic=None) -> None:
                self.added.append((text, topic))

        p = self._pipeline(resume)
        fr = FakeRetrieval()
        p._seed_resume_sync(fr, resume.retrieval_sections())
        assert len(fr.added) == len(resume.retrieval_sections())
