from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: str = "1") -> bool:
    return os.getenv(name, default) != "0"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class CoreConfig:
    # STT
    stt_model: str = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_STT_MODEL", "base")
    )
    stt_device: str = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_STT_DEVICE", "cpu")
    )
    stt_compute: str = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_STT_COMPUTE", "int8")
    )

    # LLM
    llm_url: str = field(
        default_factory=lambda: os.getenv(
            "RESUMEVOICE_LLM_URL", "http://localhost:8000/v1"
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "RESUMEVOICE_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"
        )
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_LLM_API_KEY", "EMPTY")
    )
    llm_temperature: float = field(
        default_factory=lambda: _env_float("RESUMEVOICE_LLM_TEMPERATURE", 0.7)
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("RESUMEVOICE_LLM_MAX_TOKENS", "512"))
    )
    llm_top_p: float = field(
        default_factory=lambda: _env_float("RESUMEVOICE_LLM_TOP_P", 0.9)
    )

    # TTS
    tts_model: str = field(
        default_factory=lambda: os.getenv(
            "RESUMEVOICE_TTS_MODEL",
            "/usr/share/piper/voices/en_US-lessac-medium.onnx",
        )
    )

    # VAD
    vad_threshold: float = field(
        default_factory=lambda: _env_float("RESUMEVOICE_VAD_THRESHOLD", 0.5)
    )
    vad_device: str = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_VAD_DEVICE", "cpu")
    )

    # Emotion classification
    emotion_enabled: bool = field(
        default_factory=lambda: _env_bool("RESUMEVOICE_EMOTION_ENABLED")
    )
    emotion_model: str = field(
        default_factory=lambda: os.getenv(
            "RESUMEVOICE_EMOTION_MODEL",
            "j-hartmann/emotion-english-distilroberta-base",
        )
    )
    emotion_device: str | None = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_EMOTION_DEVICE", "cpu")
    )

    # Resume persona (personal voice agent)
    resume_enabled: bool = field(
        default_factory=lambda: _env_bool("RESUMEVOICE_RESUME_ENABLED")
    )
    resume_path: str = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_RESUME_PATH", "")
    )
