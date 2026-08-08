import asyncio

from core.pipeline import ConversationContext, StreamingPipeline
from modules.tts.prosody import (
    ProsodySelector,
    classify_sentiment,
    comma_fractions,
    comma_pause,
    pause_for,
    splice_audio,
    split_emphasis,
)
from tests.test_streaming import (
    FakeLLMText,
    FakeSTTText,
    FakeTTSStream,
    FakeVAD,
)


class _FakeSTTQuestion:
    async def transcribe(self, audio_blob: bytes) -> str:
        return "what time is it?"


class _FakeSTTEcho:
    async def transcribe(self, audio_blob: bytes) -> str:
        return "hello world and welcome back"


def _make_pipeline() -> StreamingPipeline:
    return StreamingPipeline(FakeSTTText(), FakeLLMText(), FakeTTSStream(), FakeVAD())


class TestProsodySelector:
    def test_question_is_slower_than_statement(self) -> None:
        sel = ProsodySelector()
        question = sel.select("What time is it?")
        statement = sel.select("It is noon.")
        assert question.length_scale > statement.length_scale
        assert question.label == "question"

    def test_emphatic_is_faster(self) -> None:
        sel = ProsodySelector()
        emphatic = sel.select("That is amazing!")
        conversational = sel.select("That is fine.")
        assert emphatic.length_scale < conversational.length_scale
        assert emphatic.label == "emphatic"

    def test_ellipsis_adds_pause(self) -> None:
        sel = ProsodySelector()
        thoughtful = sel.select("Well... let me think.")
        plain = sel.select("Let me think.")
        assert thoughtful.sentence_silence > plain.sentence_silence
        assert thoughtful.label == "thoughtful"

    def test_rising_engagement_is_eager_and_fast(self) -> None:
        sel = ProsodySelector()
        eager = sel.select(
            "Tell me more about it.", trajectory="rising", engagement=0.8
        )
        calm = sel.select(
            "Tell me more about it.", trajectory="falling", engagement=0.3
        )
        assert eager.label == "eager"
        assert eager.length_scale < calm.length_scale

    def test_falling_is_calm_and_slow(self) -> None:
        sel = ProsodySelector()
        calm = sel.select("I see.", trajectory="falling", engagement=0.3)
        assert calm.label == "calm"
        assert calm.length_scale > 1.0

    def test_topic_shift_intro_slower_only_on_first_chunk(self) -> None:
        sel = ProsodySelector()
        first = sel.select("Let us talk about space.", topic_shift=True, first=True)
        later = sel.select("Let us talk about space.", topic_shift=True, first=False)
        assert first.length_scale > later.length_scale
        assert first.label == "conversational-intro"

    def test_numbered_item_is_steady(self) -> None:
        sel = ProsodySelector()
        item = sel.select("1. First step")
        assert item.label == "list"
        assert item.length_scale == 1.0

    def test_negative_sentiment_is_supportive_and_slower(self) -> None:
        sel = ProsodySelector()
        supportive = sel.select("Let me help you.", user_sentiment="negative")
        neutral = sel.select("Let me help you.", user_sentiment="neutral")
        assert "supportive" in supportive.label
        assert supportive.length_scale > neutral.length_scale

    def test_positive_sentiment_is_warm(self) -> None:
        sel = ProsodySelector()
        warm = sel.select("Let me help you.", user_sentiment="positive")
        assert "warm" in warm.label
        assert (
            warm.noise_scale > ProsodySelector().select("Let me help you.").noise_scale
        )

    def test_repetition_is_patient_and_slower(self) -> None:
        sel = ProsodySelector()
        repeated = sel.select("Let me help you.", user_repeated=True)
        normal = sel.select("Let me help you.", user_repeated=False)
        assert "patient" in repeated.label
        assert repeated.length_scale > normal.length_scale

    def test_first_response_opens_slower(self) -> None:
        sel = ProsodySelector()
        opening = sel.select("Hello there.", first=True, first_response=True)
        normal = sel.select("Hello there.", first=False, first_response=False)
        assert "opening" in opening.label
        assert opening.length_scale > normal.length_scale

    def test_complex_query_is_structured_and_slower(self) -> None:
        sel = ProsodySelector()
        complex_p = sel.select("Here is the answer.", complexity="complex")
        standard_p = sel.select("Here is the answer.", complexity="standard")
        assert "structured" in complex_p.label
        assert complex_p.length_scale > standard_p.length_scale

    def test_ongoing_conversation_is_casual_and_faster(self) -> None:
        sel = ProsodySelector()
        casual = sel.select("Right, sure.", turn_count=6)
        fresh = sel.select("Right, sure.", turn_count=2)
        assert "casual" in casual.label
        assert casual.length_scale < fresh.length_scale


class TestClassifySentiment:
    def test_frustration_is_negative(self) -> None:
        assert classify_sentiment("this is frustrating and wrong") == "negative"

    def test_praise_is_positive(self) -> None:
        assert classify_sentiment("that is amazing great work") == "positive"

    def test_neutral_text(self) -> None:
        assert classify_sentiment("what is the time") == "neutral"


class TestPauseFor:
    def test_ellipsis_pause_is_largest(self) -> None:
        assert pause_for("...", 0.1) > pause_for(".", 0.1)
        assert pause_for("...", 0.1) >= 0.25

    def test_question_pause_longer_than_period(self) -> None:
        assert pause_for("?", 0.1) > pause_for(".", 0.1)

    def test_no_terminator_is_short(self) -> None:
        assert pause_for("", 0.1) < pause_for(".", 0.1)


class TestCommaPause:
    def test_comma_pause_scaled_and_clamped(self) -> None:
        assert comma_pause(0.12) == 0.12
        assert comma_pause(10.0) == 0.25
        assert comma_pause(0.01) == 0.05

    def test_comma_fractions_word_based(self) -> None:
        fractions = comma_fractions("Hello there, and welcome back.")
        assert len(fractions) == 1
        assert 0.0 < fractions[0] < 1.0

    def test_comma_fractions_skip_numeric_and_quoted(self) -> None:
        assert comma_fractions("A list of 1,000 and 2,500 items.") == []
        fractions = comma_fractions('He said, "no, thanks" to me.')
        assert len(fractions) == 1
        assert 0.0 < fractions[0] < 0.5

    def test_splice_audio_inserts_pad(self) -> None:
        audio = bytes([1] * 2 * 16)  # 16 samples
        pad = b"\x00\x00"
        pieces = splice_audio(audio, [0.5], 16000, pad)
        assert len(pieces) == 3
        assert pieces[1] == pad
        assert b"".join(pieces) == audio[:16] + pad + audio[16:]

    def test_splice_audio_no_commas_returns_single_piece(self) -> None:
        audio = b"\x01\x02"
        assert splice_audio(audio, [], 16000, b"\x00\x00") == [audio]
        assert splice_audio(audio, [0.5], 16000, b"") == [audio]


class TestSplitEmphasis:
    def test_all_caps_word_marked_emphasized(self) -> None:
        segments = split_emphasis("This is IMPORTANT to remember.")
        emph = [seg for seg, flag in segments if flag]
        assert emph == ["IMPORTANT"]
        assert any(not flag for _, flag in segments)

    def test_no_all_caps_returns_single_segment(self) -> None:
        segments = split_emphasis("This is normal speech.")
        assert segments == [("This is normal speech.", False)]


class TestProsodyPipelineIntegration:
    def test_statement_uses_conversational_profile(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )
            assert p.tts.synthesized
            labels = " ".join(p.tts.prosody_labels)
            assert "conversational" in labels
            assert "opening" in labels

        asyncio.run(run())

    def test_question_uses_answer_profile(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p.stt = _FakeSTTQuestion()
            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )
            assert p.tts.synthesized
            assert "answer" in p.tts.prosody_labels

        asyncio.run(run())


class TestEchoGating:
    def test_looks_like_echo_positive(self) -> None:
        p = _make_pipeline()
        p._last_spoken["sess"] = "let me explain the Hadamard matrix and its uses"
        assert p._looks_like_echo("sess", "explain the Hadamard matrix")

    def test_looks_like_echo_negative_on_distinct_topic(self) -> None:
        p = _make_pipeline()
        p._last_spoken["sess"] = "the weather is sunny and warm today"
        assert not p._looks_like_echo("sess", "what is the capital of France")

    def test_segment_dropped_when_transcript_matches_last_spoken(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p._last_spoken["sess"] = "hello world and welcome back everyone"
            p.stt = _FakeSTTEcho()

            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )
            assert p.tts.synthesized == []

        asyncio.run(run())

    def test_segment_kept_when_transcript_is_new(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p._last_spoken["sess"] = "the weather is sunny and warm"
            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )
            assert p.tts.synthesized

        asyncio.run(run())
