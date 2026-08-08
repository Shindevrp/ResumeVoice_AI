from __future__ import annotations

import time
from collections import defaultdict, deque
from statistics import median


class LatencyTracker:
    def __init__(self, window_size: int = 100) -> None:
        self._stages: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def mark(self, stage: str) -> None:
        self._stages[stage].append(time.perf_counter())

    def measure(self, stage: str, start: float) -> float:
        elapsed = time.perf_counter() - start
        self._stages[stage].append(elapsed)
        return elapsed

    def report(self) -> dict[str, dict[str, float]]:
        result = {}
        for stage, samples in self._stages.items():
            if not samples:
                continue
            sorted_s = sorted(samples)
            result[stage] = {
                "p50": median(samples),
                "p95": sorted_s[int(len(sorted_s) * 0.95)],
                "p99": sorted_s[int(len(sorted_s) * 0.99)],
                "count": len(samples),
            }
        return result

    def reset(self) -> None:
        self._stages.clear()
