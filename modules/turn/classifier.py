from __future__ import annotations


END_TURN_FILLER_WORDS = {
    "um", "uh", "like", "well", "so", "actually", "basically",
    "you know", "i mean", "sort of", "kind of",
}

TRAILING_CONJUNCTIONS = {
    "and", "but", "or", "because", "so", "then", "if",
}

QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "who", "which",
    "do", "does", "did", "is", "are", "was", "were", "can", "could",
    "would", "should", "will", "shall", "have", "has", "had",
}


class TurnClassifier:
    def __init__(
        self,
        silence_threshold: float = 0.6,
        min_speech_duration: float = 0.5,
        prosody_weight: float = 0.3,
        silence_weight: float = 0.5,
        linguistic_weight: float = 0.2,
    ) -> None:
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.prosody_weight = prosody_weight
        self.silence_weight = silence_weight
        self.linguistic_weight = linguistic_weight

    def classify(
        self,
        features: dict[str, float],
        partial_transcript: str = "",
        prosody: dict[str, float | str] | None = None,
    ) -> str:
        silence = features.get("silence_duration", 0.0)
        speech_dur = features.get("speech_duration", 0.0)

        if speech_dur < self.min_speech_duration:
            return "continue"

        silence_score = min(silence / self.silence_threshold, 1.0)

        prosody_score = self._score_prosody(prosody)

        linguistic_score = self._score_linguistic(partial_transcript)

        combined = (
            self.silence_weight * silence_score
            + self.prosody_weight * prosody_score
            + self.linguistic_weight * linguistic_score
        )

        if combined >= 0.6:
            return "end_turn"
        if silence >= self.silence_threshold * 1.5:
            return "end_turn_force"
        return "continue"

    def _score_prosody(self, prosody: dict[str, float | str] | None) -> float:
        if not prosody:
            return 0.0
        trajectory = prosody.get("trajectory", "neutral")
        if trajectory == "falling":
            return 0.8
        if trajectory == "flat":
            return 0.4
        return 0.1

    def _score_linguistic(self, text: str) -> float:
        if not text:
            return 0.0

        text_lower = text.lower().strip()

        if any(text_lower.endswith(w) for w in TRAILING_CONJUNCTIONS):
            return 0.1

        if any(text_lower.startswith(w) for w in QUESTION_WORDS):
            return 0.9

        if text_lower.endswith("?"):
            return 0.9

        if text_lower.endswith(".") or text_lower.endswith("!"):
            return 0.7

        words = text_lower.split()
        if words and words[-1] in END_TURN_FILLER_WORDS:
            return 0.3

        return 0.4