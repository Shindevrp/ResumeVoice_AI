from __future__ import annotations

import os
from dataclasses import dataclass, field


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
        default_factory=lambda: os.getenv("RESUMEVOICE_LLM_URL", "http://localhost:8000/v1")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "RESUMEVOICE_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"
        )
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_LLM_API_KEY", "EMPTY")
    )
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    llm_top_p: float = 0.9

    # TTS
    tts_model: str = field(
        default_factory=lambda: os.getenv(
            "RESUMEVOICE_TTS_MODEL",
            "/usr/share/piper/voices/en_US-lessac-medium.onnx",
        )
    )

    # VAD
    vad_threshold: float = field(
        default_factory=lambda: float(os.getenv("RESUMEVOICE_VAD_THRESHOLD", "0.5"))
    )
    vad_sample_rate: int = 16000
    silence_ms: float = 400.0

    # Emotion classification
    emotion_enabled: bool = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_EMOTION_ENABLED", "1") != "0"
    )
    emotion_model: str = field(
        default_factory=lambda: os.getenv(
            "RESUMEVOICE_EMOTION_MODEL",
            "j-hartmann/emotion-english-distilroberta-base",
        )
    )
    emotion_device: str | None = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_EMOTION_DEVICE", "cuda")
    )

    # Resume persona (personal voice agent)
    resume_enabled: bool = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_RESUME_ENABLED", "1") != "0"
    )
    resume_path: str = field(
        default_factory=lambda: os.getenv("RESUMEVOICE_RESUME_PATH", "")
    )

    # Pipeline
    audio_queue_size: int = 512
    output_queue_size: int = 512
    default_timeout: float = 30.0
    default_language: str = "en"
    session_timeout: float = 300.0
