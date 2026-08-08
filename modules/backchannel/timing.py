from __future__ import annotations


class BackchannelTiming:
    def __init__(
        self,
        min_silence_ms: float = 400.0,
        max_silence_ms: float = 2000.0,
        min_speech_duration_ms: float = 2000.0,
    ) -> None:
        self.min_silence = min_silence_ms
        self.max_silence = max_silence_ms
        self.min_speech = min_speech_duration_ms

    def should_emit(
        self,
        silence_duration_ms: float,
        speech_duration_ms: float,
        engagement_score: float,
    ) -> bool:
        if silence_duration_ms < self.min_silence:
            return False
        if silence_duration_ms > self.max_silence:
            return False
        if speech_duration_ms < self.min_speech:
            return False
        if engagement_score < 0.3:
            return False
        return True