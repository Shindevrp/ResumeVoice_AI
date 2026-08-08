from __future__ import annotations

from modules.backchannel.generator import (
    BACKCHANNEL_CANDIDATES,
    BackchannelGenerator,
)
from modules.backchannel.timing import BackchannelTiming
from modules.turn.backchannel import TurnBackchannel


class TestBackchannelTiming:
    def test_too_little_silence(self) -> None:
        t = BackchannelTiming()
        assert not t.should_emit(100, 3000, 0.6)

    def test_too_little_speech(self) -> None:
        t = BackchannelTiming()
        assert not t.should_emit(500, 1000, 0.6)

    def test_disengaged(self) -> None:
        t = BackchannelTiming()
        assert not t.should_emit(500, 3000, 0.2)

    def test_too_much_silence(self) -> None:
        t = BackchannelTiming()
        assert not t.should_emit(5000, 3000, 0.6)

    def test_ok(self) -> None:
        t = BackchannelTiming()
        assert t.should_emit(500, 3000, 0.6)

    def test_boundaries(self) -> None:
        t = BackchannelTiming(min_silence_ms=400, max_silence_ms=2000)
        assert t.should_emit(400, 2000, 0.5)


class TestBackchannelGenerator:
    def test_cooldown_blocks(self) -> None:
        g = BackchannelGenerator(cooldown_seconds=10)
        assert g.generate("hi there", 0.0, 5.0) is None

    def test_generates_within_candidates(self) -> None:
        g = BackchannelGenerator(cooldown_seconds=0)
        bc = g.generate("hi there", 0.0, 1.0)
        assert bc in BACKCHANNEL_CANDIDATES["acknowledging"]

    def test_context_keywords(self) -> None:
        g = BackchannelGenerator(cooldown_seconds=0)
        assert (
            g.generate("that is really amazing", 0.0, 1.0)
            in (BACKCHANNEL_CANDIDATES["surprised"])
        )

    def test_empty_transcript_acknowledging(self) -> None:
        g = BackchannelGenerator(cooldown_seconds=0)
        assert g.generate("", 0.0, 1.0) in BACKCHANNEL_CANDIDATES["acknowledging"]

    def test_generate_thinking(self) -> None:
        g = BackchannelGenerator()
        assert g.generate_thinking() in BACKCHANNEL_CANDIDATES["thinking"]


class TestTurnBackchannel:
    def test_delegates_should_emit(self) -> None:
        tb = TurnBackchannel()
        assert tb.should_emit(500, 3000, 0.6)
        assert not tb.should_emit(500, 3000, 0.2)

    def test_delegates_generate(self) -> None:
        tb = TurnBackchannel(generator=BackchannelGenerator(cooldown_seconds=0))
        bc = tb.generate("hi there", 0.0, 1.0)
        assert bc in BACKCHANNEL_CANDIDATES["acknowledging"]
