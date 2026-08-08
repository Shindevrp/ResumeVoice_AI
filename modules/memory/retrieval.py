from __future__ import annotations

from modules.memory.session import SessionMemory
from modules.memory.vector_db import VectorDB


class RetrievalModule:
    def __init__(
        self,
        vector_db: VectorDB | None = None,
        max_context_tokens: int = 2048,
        min_score: float = 0.3,
        topic_boost: float = 0.08,
    ) -> None:
        self.vector_db = vector_db or VectorDB()
        self.max_context_tokens = max_context_tokens
        self.min_score = min_score
        self.topic_boost = topic_boost

    def retrieve_context(
        self,
        query: str,
        session_memory: SessionMemory,
        top_k: int = 3,
        topic: str | None = None,
    ) -> list[str]:
        """Vector hits only, filtered by score and deduped against recent turns.

        Recent turns are already injected as full history messages, so echoing
        them here would only duplicate context and waste tokens. When a topic
        is given, same-topic memories are boosted.
        """
        return [
            doc
            for doc, _ in self.retrieve_context_with_topics(
                query, session_memory, top_k=top_k, topic=topic
            )
        ]

    def retrieve_context_with_topics(
        self,
        query: str,
        session_memory: SessionMemory,
        top_k: int = 3,
        topic: str | None = None,
    ) -> list[tuple[str, str | None]]:
        results = self.vector_db._search_scored_with_meta(
            query,
            top_k=top_k * 2,
            topic=topic,
            topic_boost=self.topic_boost,
        )
        recent = {e.content.strip().lower() for e in session_memory.get_history(4)}
        hits = [
            (doc, meta.get("topic"))
            for doc, score, meta in results
            if score >= self.min_score and doc.strip().lower() not in recent
        ]
        return hits[:top_k]

    def add_to_long_term(self, text: str, topic: str | None = None) -> None:
        self.vector_db.add(text, metadata={"topic": topic or ""})

    def warm_up(self) -> None:
        self.vector_db.warm_up()
