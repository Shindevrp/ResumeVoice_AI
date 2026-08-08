from __future__ import annotations

from modules.turn.classifier import TurnClassifier
from modules.turn.features import TurnFeatures
from modules.prosody.analyzer import ProsodyAnalyzer


class TurnDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: float = 0.6,
    ) -> None:
        self.feature_extractor = TurnFeatures(sample_rate=sample_rate)
        self.prosody_analyzer = ProsodyAnalyzer()
        self.classifier = TurnClassifier(silence_threshold=silence_threshold)

    def process_chunk(
        self,
        audio_chunk: bytes,
        is_speech: bool,
        partial_transcript: str = "",
    ) -> str:
        features = self.feature_extractor.extract(audio_chunk, is_speech)
        prosody = self.prosody_analyzer.update(features)
        decision = self.classifier.classify(features, partial_transcript, prosody)
        return decision

    def reset(self) -> None:
        self.feature_extractor.reset()
        self.prosody_analyzer.reset()