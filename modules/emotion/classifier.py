from __future__ import annotations

import asyncio

from modules.tts.prosody import classify_sentiment
from utils.logger import get_logger

logger = get_logger("emotion")

DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Model emotion label -> conversation tone bucket used by the prosody selector.
_LABEL_MAP = {
    "joy": "positive",
    "love": "positive",
    "admiration": "positive",
    "gratitude": "positive",
    "neutral": "neutral",
    "surprise": "neutral",
    "curiosity": "neutral",
    "anger": "negative",
    "fear": "negative",
    "sadness": "negative",
    "disgust": "negative",
    "anxiety": "negative",
}


def emotion_to_sentiment(label: str) -> str:
    return _LABEL_MAP.get(label.strip().lower(), "neutral")


class EmotionClassifier:
    """Real emotion classification for the user's transcript.

    Runs a transformers text-classification model lazily on first use so
    startup is not blocked. Falls back to the lexicon `classify_sentiment`
    when disabled, when loading fails, or when a call times out.
    """

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        timeout: float = 0.5,
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.device = device
        self.timeout = timeout
        self._pipe = None
        self._load_error: str | None = None
        self.last_label: str | None = None
        self.last_confidence: float = 0.0

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        if not self.enabled:
            return
        if self._pipe is not None or self._load_error is not None:
            return
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=self.device,
            )
            logger.info(
                f"emotion classifier loaded model={self.model_name}"
                f" device={self.device}"
            )
        except Exception as e:
            self._load_error = str(e)
            logger.warning(
                f"emotion classifier unavailable, using lexicon fallback: {e}"
            )

    def classify(self, text: str) -> str:
        """Return 'positive' | 'negative' | 'neutral' for a user utterance."""
        if not self.enabled or self._pipe is None:
            return classify_sentiment(text)
        try:
            result = self._pipe(text)
            top = result[0]
            if isinstance(top, list):
                top = max(top, key=lambda item: item.get("score", 0.0))
            label = top.get("label", "neutral")
            self.last_label = label
            self.last_confidence = float(top.get("score", 0.0))
            return emotion_to_sentiment(label)
        except Exception as e:
            logger.warning(f"emotion classify failed, fallback: {e}")
            return classify_sentiment(text)

    async def classify_async(self, text: str) -> str:
        await asyncio.to_thread(self.load)
        if self._pipe is None:
            return classify_sentiment(text)
        return await asyncio.to_thread(self.classify, text)
