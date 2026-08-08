from __future__ import annotations

from collections import deque


class ProsodyAnalyzer:
    def __init__(self, window_size: int = 30) -> None:
        self._pitch_buffer: deque[float] = deque(maxlen=window_size)
        self._energy_buffer: deque[float] = deque(maxlen=window_size)
        self._zcr_buffer: deque[float] = deque(maxlen=window_size)

    def update(self, features: dict[str, float]) -> dict[str, float | str]:
        self._pitch_buffer.append(features.get("pitch", 0.0))
        self._energy_buffer.append(features.get("energy", 0.0))
        self._zcr_buffer.append(features.get("zero_crossing_rate", 0.0))
        return self.analyze()

    def analyze(self) -> dict[str, float | str]:
        if len(self._pitch_buffer) < 3:
            return {"trajectory": "neutral", "pitch_trend": 0, "energy_trend": 0}

        pitch_vals = list(self._pitch_buffer)
        energy_vals = list(self._energy_buffer)
        pitch_trend = pitch_vals[-1] - pitch_vals[0]
        energy_trend = energy_vals[-1] - energy_vals[0]

        if pitch_trend > 20:
            trajectory = "rising"
        elif pitch_trend < -20:
            trajectory = "falling"
        else:
            trajectory = "flat"

        return {
            "trajectory": trajectory,
            "pitch_trend": pitch_trend,
            "energy_trend": energy_trend,
        }

    def reset(self) -> None:
        self._pitch_buffer.clear()
        self._energy_buffer.clear()
        self._zcr_buffer.clear()
