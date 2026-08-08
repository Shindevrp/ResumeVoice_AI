import asyncio

from core.pipeline import ConversationContext, StreamingPipeline
from modules.emotion.classifier import (
    EmotionClassifier,
    emotion_to_sentiment,
)
from modules.tts.prosody import classify_sentiment
from tests.test_streaming import (
    FakeLLMText,
    FakeSTTText,
    FakeTTSStream,
    FakeVAD,
)


class FakeEmotion:
    def __init__(self, sentiment: str) -> None:
        self.sentiment = sentiment
        self.timeout = 0.5
        self.last_label = None
        self.last_confidence = 0.0

    async def classify_async(self, text: str) -> str:
        return self.sentiment


def _make_pipeline() -> StreamingPipeline:
    return StreamingPipeline(FakeSTTText(), FakeLLMText(), FakeTTSStream(), FakeVAD())


class TestEmotionToSentiment:
    def test_positive_labels(self) -> None:
        assert emotion_to_sentiment("joy") == "positive"
        assert emotion_to_sentiment("LOVE") == "positive"

    def test_negative_labels(self) -> None:
        assert emotion_to_sentiment("anger") == "negative"
        assert emotion_to_sentiment("sadness") == "negative"

    def test_neutral_and_unknown(self) -> None:
        assert emotion_to_sentiment("neutral") == "neutral"
        assert emotion_to_sentiment("surprise") == "neutral"
        assert emotion_to_sentiment("not-a-label") == "neutral"


class TestEmotionClassifier:
    def test_disabled_uses_lexicon_fallback(self) -> None:
        c = EmotionClassifier(enabled=False)
        assert c.classify("that is amazing great work") == "positive"
        assert c.classify("this is frustrating and wrong") == "negative"

    def test_disabled_load_is_noop(self) -> None:
        c = EmotionClassifier(enabled=False)
        c.load()
        assert not c.loaded

    def test_disabled_async_fallback(self) -> None:
        c = EmotionClassifier(enabled=False)
        result = asyncio.run(c.classify_async("what is the time"))
        assert result == "neutral"


class TestEmotionPipelineIntegration:
    def test_angry_user_drives_supportive_profile(self) -> None:
        async def run() -> None:
            p = _make_pipeline()
            p._emotion = FakeEmotion("negative")
            await p._process_speech_segment(
                b"\x00" * 1600, "sess", ConversationContext()
            )
            labels = " ".join(p.tts.prosody_labels)
            assert "supportive" in labels

        asyncio.run(run())

    def test_fallback_keeps_lexicon_when_classifier_slow(self) -> None:
        async def run() -> None:
            p = _make_pipeline()

            class SlowEmotion:
                timeout = 0.0

                async def classify_async(self, text: str) -> str:
                    await asyncio.Event().wait()
                    return "negative"

            p._emotion = SlowEmotion()
            ctx = ConversationContext()
            await p._process_speech_segment(b"\x00" * 1600, "sess", ctx)
            assert ctx.user_sentiment == classify_sentiment(
                "hello world"
            )

        asyncio.run(run())
