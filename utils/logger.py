from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

_KNOWN_ATTRS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
}


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in _KNOWN_ATTRS and not key.startswith("_"):
                log[key] = val
        if record.exc_info and record.exc_info[0]:
            log["exc"] = self.formatException(record.exc_info)
        return json.dumps(log, default=str)


def get_logger(name: str = "ResumeVoice", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(f"ResumeVoice.{name}" if name != "ResumeVoice" else "ResumeVoice")
    if logger.handlers:
        return logger
    env_level = os.getenv("RESUMEVOICE_LOG_LEVEL", "").strip().upper()
    if env_level == "DEBUG":
        level = logging.DEBUG
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = get_logger("ResumeVoice")
