from __future__ import annotations

from modules.tts.chunker import TTSChunker


class TestTTSChunker:
    def test_sentence_boundary(self) -> None:
        c = TTSChunker()
        chunks = c.feed("Hello there. How are")
        assert chunks == ["Hello there."]
        assert c.flush() == "How are"

    def test_short_sentence_held_until_flush(self) -> None:
        c = TTSChunker()
        assert c.feed("Hi.") == []
        assert c.flush() == "Hi."

    def test_abbreviation_not_split(self) -> None:
        c = TTSChunker()
        chunks = c.feed("Dr. Smith is here.")
        assert chunks == ["Dr. Smith is here."]

    def test_decimal_not_split(self) -> None:
        c = TTSChunker()
        chunks = c.feed("It is 3.5 km away.")
        assert chunks == ["It is 3.5 km away."]

    def test_multi_sentence_token(self) -> None:
        c = TTSChunker()
        chunks = c.feed("One sentence. Two sentence. Three")
        assert chunks == ["One sentence. Two sentence."]
        assert c.flush() == "Three"

    def test_clause_split_for_long_span(self) -> None:
        c = TTSChunker()
        head = "This is a very long introductory clause that keeps going and going and going"
        tail = " it finally concludes with the remainder of the very long sentence that follows"
        assert len(head) + 1 + len(tail) > 150
        chunks = c.feed(head + "," + tail)
        assert chunks == [head + ","]
        assert c.flush() == tail.strip()

    def test_hard_character_cap(self) -> None:
        c = TTSChunker()
        chunks = c.feed("a" * 200)
        assert chunks == ["a" * 150]
        assert c.flush() == "a" * 50

    def test_reset_clears_buffer(self) -> None:
        c = TTSChunker()
        c.feed("Some partial")
        c.reset()
        assert c.flush() is None
