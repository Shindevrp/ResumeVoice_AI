from __future__ import annotations

import time


class TurnTiming:
    """Conversation pacing engine — computes human-like response delays.

    Adjusts delay based on:
    - Engagement level (engaged users get faster responses)
    - Turn duration (longer user turns get slightly slower responses)
    - Conversation momentum (rapid exchanges stay fast)
    - Question detection (questions get faster answers)
    """

    def __init__(
        self,
        base_delay: float = 0.15,
        min_delay: float = 0.05,
        max_delay: float = 0.6,
        engagement_factor: float = 0.4,
        duration_factor: float = 0.1,
    ) -> None:
        self.base_delay = base_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.engagement_factor = engagement_factor
        self.duration_factor = duration_factor
        self._last_turn_time = 0.0
        self._rapid_fire_count = 0

    def compute_delay(
        self,
        pause_duration: float,
        engagement_score: float = 0.5,
        turn_duration_ms: float = 0.0,
        is_question: bool = False,
        is_backchannel: bool = False,
    ) -> float:
        if is_backchannel:
            return max(0.1, pause_duration * 0.6)

        delay = self.base_delay

        engaged_bonus = 1.0 - (engagement_score * self.engagement_factor)
        delay *= engaged_bonus

        if turn_duration_ms > 5000:
            delay *= 1.0 + self.duration_factor
        elif turn_duration_ms < 1000:
            delay *= 1.0 - self.duration_factor * 0.5

        if is_question:
            delay *= 0.7

        now = time.time()
        if self._last_turn_time > 0 and now - self._last_turn_time < 2.0:
            self._rapid_fire_count += 1
            if self._rapid_fire_count > 2:
                delay *= 0.8
        else:
            self._rapid_fire_count = 0
        self._last_turn_time = now

        return max(self.min_delay, min(self.max_delay, delay))

    def reset(self) -> None:
        self._last_turn_time = 0.0
        self._rapid_fire_count = 0
