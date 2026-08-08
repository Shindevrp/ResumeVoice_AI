from __future__ import annotations

from modules.backchannel.generator import BackchannelGenerator
from modules.backchannel.timing import BackchannelTiming


class TurnBackchannel:
    """Turn-level backchannel integration.

    Delegates to the central BackchannelGenerator + BackchannelTiming
    but adds turn-context awareness (speech duration, engagement).
    """

    def __init__(
        self,
        generator: BackchannelGenerator | None = None,
        timing: BackchannelTiming | None = None,
    ) -> None:
        self.generator = generator or BackchannelGenerator()
        self.timing = timing or BackchannelTiming()

    def should_emit(
        self,
        silence_duration_ms: float,
        speech_duration_ms: float,
        engagement_score: float,
    ) -> bool:
        return self.timing.should_emit(
            silence_duration_ms, speech_duration_ms, engagement_score
        )

    def generate(
        self,
        transcript: str,
        last_backchannel_time: float,
        current_time: float,
    ) -> str | None:
        return self.generator.generate(transcript, last_backchannel_time, current_time)
