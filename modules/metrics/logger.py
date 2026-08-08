from __future__ import annotations

import json
import logging

logger = logging.getLogger("ResumeVoice.metrics")


class MetricsLogger:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def log(self, event: str, payload: dict | None = None) -> None:
        if not self.enabled:
            return
        data = {"event": event, **(payload or {})}
        logger.info(json.dumps(data))

    def latency_report(self, report: dict[str, dict[str, float]]) -> None:
        if not self.enabled:
            return
        logger.info(f"Latency report: {json.dumps(report, default=str)}")
