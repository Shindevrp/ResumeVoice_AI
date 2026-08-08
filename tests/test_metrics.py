from __future__ import annotations

import time

import pytest

from modules.metrics.latency import LatencyTracker


class TestLatencyTracker:
    def test_report_empty(self) -> None:
        assert LatencyTracker().report() == {}

    def test_measure_records_elapsed(self) -> None:
        t = LatencyTracker()
        start = time.perf_counter() - 0.25
        elapsed = t.measure("stt", start)
        assert elapsed == pytest.approx(0.25, abs=0.05)

    def test_report_percentiles(self) -> None:
        t = LatencyTracker()
        for i in range(1, 101):
            t.measure("stage", time.perf_counter() - i)
        r = t.report()["stage"]
        assert r["count"] == 100
        assert r["p50"] == pytest.approx(50.5, abs=0.1)
        assert r["p95"] == pytest.approx(96.0, abs=0.1)
        assert r["p99"] == pytest.approx(100.0, abs=0.1)

    def test_mark_counts_samples(self) -> None:
        t = LatencyTracker()
        t.mark("x")
        t.mark("x")
        assert t.report()["x"]["count"] == 2

    def test_reset(self) -> None:
        t = LatencyTracker()
        t.measure("x", time.perf_counter() - 0.1)
        t.reset()
        assert t.report() == {}
