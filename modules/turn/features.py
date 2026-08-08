from __future__ import annotations

from modules.prosody.extractor import ProsodyExtractor


class TurnFeatures:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.prosody = ProsodyExtractor(sample_rate=sample_rate)
        self._silence_duration = 0.0
        self._speech_duration = 0.0

    def extract(self, audio_chunk: bytes, is_speech: bool) -> dict[str, float]:
        features = self.prosody.extract(audio_chunk)
        frame_ms = len(audio_chunk) / (self.prosody.sample_rate * 2 / 1000)

        if is_speech:
            self._silence_duration = 0.0
            self._speech_duration += frame_ms
        else:
            self._silence_duration += frame_ms

        features["silence_duration"] = self._silence_duration
        features["speech_duration"] = self._speech_duration
        features["frame_duration_ms"] = frame_ms
        return features

    def reset(self) -> None:
        self._silence_duration = 0.0
        self._speech_duration = 0.0
        self.prosody.extractor = ProsodyExtractor()
