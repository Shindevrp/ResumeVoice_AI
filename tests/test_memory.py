from __future__ import annotations

import asyncio

import pytest

from modules.memory.session import SessionMemory
from modules.memory.vector_db import VectorDB
from modules.memory.retrieval import RetrievalModule
from core.pipeline import ConversationContext, StreamingPipeline
from tests.test_streaming import FakeSTT, FakeLLM, FakeTTS, FakeVAD


class TestSessionMemory:
    def test_add_and_retrieve(self) -> None:
        m = SessionMemory(max_turns=3)
        m.add("user", "hello")
        m.add("assistant", "hi")
        hist = m.get_history()
        assert len(hist) == 2
        assert hist[0].role == "user"
        assert hist[1].role == "assistant"

    def test_max_turns(self) -> None:
        m = SessionMemory(max_turns=2)
        m.add("user", "a")
        m.add("assistant", "b")
        m.add("user", "c")
        hist = m.get_history()
        assert len(hist) == 2
        assert hist[0].content == "b"
        assert hist[1].content == "c"

    def test_context_messages(self) -> None:
        m = SessionMemory()
        m.add("user", "hello")
        msgs = m.context_messages(system_prompt="You are ResumeVoice AI.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


class TestVectorDB:
    def test_search(self) -> None:
        db = VectorDB(embedding_dim=4)

        class FakeEncoder:
            def encode(self, text, **kw):
                import numpy as np
                if "weather" in text.lower():
                    return np.array([1, 0, 0, 0])
                return np.array([0, 1, 0, 0])

            def get_sentence_embedding_dimension(self):
                return 4

        db._encoder = FakeEncoder()
        db.add("the weather is nice")
        db.add("I like pizza")
        results = db.search("weather today", top_k=1)
        assert "weather" in results[0].lower()

    def test_search_scored_returns_scores(self) -> None:
        db = VectorDB(embedding_dim=4)

        class FakeEncoder:
            def encode(self, text, **kw):
                import numpy as np
                if "weather" in text.lower():
                    return np.array([1, 0, 0, 0])
                return np.array([0, 1, 0, 0])

            def get_sentence_embedding_dimension(self):
                return 4

        db._encoder = FakeEncoder()
        db.add("the weather is nice")
        db.add("I like pizza")
        scored = db.search_scored("weather today", top_k=2)
        by_doc = {d: s for d, s in scored}
        assert by_doc["the weather is nice"] == pytest.approx(1.0)
        assert by_doc["I like pizza"] == pytest.approx(0.0)

    def test_add_with_metadata(self) -> None:
        db = VectorDB(embedding_dim=4)
        db.add("hello", metadata={"topic": "weather"})
        assert db.metadata[-1] == {"topic": "weather"}

    def test_topic_boost_ranks_same_topic_docs(self) -> None:
        db = VectorDB(embedding_dim=4)

        class FakeEncoder:
            def encode(self, text, **kw):
                import numpy as np
                if "weather" in text.lower():
                    return np.array([1, 0, 0, 0])
                return np.array([0, 1, 0, 0])

            def get_sentence_embedding_dimension(self):
                return 4

        db._encoder = FakeEncoder()
        db.add("the weather is nice", metadata={"topic": "weather"})
        db.add("I like pizza", metadata={"topic": "food"})
        db.add("cloudy skies tonight", metadata={"topic": "weather"})

        docs = [d for d, _ in db.search_scored("weather today", top_k=3, topic="weather")]
        assert docs == [
            "the weather is nice",
            "cloudy skies tonight",
            "I like pizza",
        ]


def _fake_encoder_db(db: VectorDB) -> None:
    class FakeEncoder:
        def encode(self, text, **kw):
            import numpy as np
            if "weather" in text.lower():
                return np.array([1, 0, 0, 0])
            return np.array([0, 1, 0, 0])

        def get_sentence_embedding_dimension(self):
            return 4

    db._encoder = FakeEncoder()


class TestRetrievalModule:
    def test_retrieve_context_no_recent_echo(self) -> None:
        rm = RetrievalModule()
        sm = SessionMemory()
        sm.add("user", "hello")
        ctx = rm.retrieve_context("hello", sm, top_k=1)
        assert not any("[Recent]" in c for c in ctx)

    def test_retrieve_context_score_and_dedup(self) -> None:
        rm = RetrievalModule(min_score=0.5)
        sm = SessionMemory()

        class FakeEncoder:
            def encode(self, text, **kw):
                import numpy as np
                if "weather" in text.lower():
                    return np.array([1, 0, 0, 0])
                return np.array([0, 1, 0, 0])

            def get_sentence_embedding_dimension(self):
                return 4

        rm.vector_db._encoder = FakeEncoder()
        rm.vector_db.add("the weather is nice")
        rm.vector_db.add("I like pizza")
        sm.add("user", "I like pizza")  # also in recent -> deduped
        hits = rm.retrieve_context("weather today", sm, top_k=2)
        assert all("pizza" not in h for h in hits)
        assert len(hits) == 1
        assert "weather" in hits[0].lower()

    def test_retrieve_context_topic_boost(self) -> None:
        rm = RetrievalModule(min_score=0.0, topic_boost=0.08)
        _fake_encoder_db(rm.vector_db)
        rm.vector_db.add("the weather is nice", metadata={"topic": "weather"})
        rm.vector_db.add("I like pizza", metadata={"topic": "food"})
        rm.vector_db.add("cloudy skies tonight", metadata={"topic": "weather"})
        sm = SessionMemory()

        hits = rm.retrieve_context("weather today", sm, top_k=3, topic="weather")
        assert hits[0] == "the weather is nice"
        assert hits.index("cloudy skies tonight") < hits.index("I like pizza")

    def test_retrieve_context_with_topics(self) -> None:
        rm = RetrievalModule(min_score=0.0)
        _fake_encoder_db(rm.vector_db)
        rm.vector_db.add("the weather is nice", metadata={"topic": "weather"})
        sm = SessionMemory()

        hits = rm.retrieve_context_with_topics("weather today", sm, top_k=1, topic="weather")
        assert hits == [("the weather is nice", "weather")]

    def test_add_to_long_term_tags_topic(self) -> None:
        rm = RetrievalModule()
        _fake_encoder_db(rm.vector_db)
        rm.add_to_long_term("the forecast is sunny", topic="weather")
        assert rm.vector_db.metadata[-1] == {"topic": "weather"}

    def test_build_messages_tags_topic_context(self) -> None:
        async def run() -> None:
            rm = RetrievalModule(min_score=0.0)
            _fake_encoder_db(rm.vector_db)
            rm.vector_db.add("the weather is nice", metadata={"topic": "weather"})
            sm = SessionMemory()
            p = StreamingPipeline(FakeSTT(), FakeLLM(), FakeTTS(), FakeVAD())
            ctx = ConversationContext()
            ctx.topic = "weather"
            ctx.turn_count = 2
            msgs = await p._build_messages("what's the forecast", ctx, sm, rm)
            joined = "\n".join(m["content"] for m in msgs)
            assert "[weather] the weather is nice" in joined

        asyncio.run(run())