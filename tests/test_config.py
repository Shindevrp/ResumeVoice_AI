from __future__ import annotations

from core.config import CoreConfig


class TestCoreConfigDefaults:
    def test_defaults(self, monkeypatch) -> None:
        for key in (
            "RESUMEVOICE_STT_MODEL",
            "RESUMEVOICE_STT_DEVICE",
            "RESUMEVOICE_STT_COMPUTE",
            "RESUMEVOICE_LLM_URL",
            "RESUMEVOICE_LLM_MODEL",
            "RESUMEVOICE_LLM_API_KEY",
            "RESUMEVOICE_LLM_TEMPERATURE",
            "RESUMEVOICE_LLM_MAX_TOKENS",
            "RESUMEVOICE_LLM_TOP_P",
            "RESUMEVOICE_TTS_MODEL",
            "RESUMEVOICE_VAD_THRESHOLD",
            "RESUMEVOICE_VAD_DEVICE",
            "RESUMEVOICE_EMOTION_ENABLED",
            "RESUMEVOICE_EMOTION_DEVICE",
            "RESUMEVOICE_RESUME_ENABLED",
            "RESUMEVOICE_RESUME_PATH",
        ):
            monkeypatch.delenv(key, raising=False)

        c = CoreConfig()
        assert c.stt_model == "base"
        assert c.stt_device == "cpu"
        assert c.stt_compute == "int8"
        assert c.llm_url == "http://localhost:8000/v1"
        assert c.llm_api_key == "EMPTY"
        assert c.llm_temperature == 0.7
        assert c.llm_max_tokens == 512
        assert c.llm_top_p == 0.9
        assert c.vad_threshold == 0.5
        assert c.vad_device == "cpu"
        assert c.emotion_enabled is True
        assert c.resume_enabled is True
        assert c.resume_path == ""

    def test_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv("RESUMEVOICE_STT_MODEL", "tiny")
        monkeypatch.setenv("RESUMEVOICE_STT_DEVICE", "cuda")
        monkeypatch.setenv("RESUMEVOICE_STT_COMPUTE", "float16")
        monkeypatch.setenv("RESUMEVOICE_LLM_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("RESUMEVOICE_LLM_MODEL", "qwen2.5:3b")
        monkeypatch.setenv("RESUMEVOICE_LLM_API_KEY", "sk-test")
        monkeypatch.setenv("RESUMEVOICE_LLM_TEMPERATURE", "0.3")
        monkeypatch.setenv("RESUMEVOICE_LLM_MAX_TOKENS", "128")
        monkeypatch.setenv("RESUMEVOICE_LLM_TOP_P", "0.5")
        monkeypatch.setenv("RESUMEVOICE_VAD_THRESHOLD", "0.3")
        monkeypatch.setenv("RESUMEVOICE_VAD_DEVICE", "cuda")
        monkeypatch.setenv("RESUMEVOICE_RESUME_ENABLED", "0")

        c = CoreConfig()
        assert c.stt_model == "tiny"
        assert c.stt_device == "cuda"
        assert c.stt_compute == "float16"
        assert c.llm_url == "http://localhost:11434/v1"
        assert c.llm_model == "qwen2.5:3b"
        assert c.llm_api_key == "sk-test"
        assert c.llm_temperature == 0.3
        assert c.llm_max_tokens == 128
        assert c.llm_top_p == 0.5
        assert c.vad_threshold == 0.3
        assert c.vad_device == "cuda"
        assert c.resume_enabled is False

    def test_invalid_float_env_falls_back(self, monkeypatch) -> None:
        monkeypatch.setenv("RESUMEVOICE_VAD_THRESHOLD", "not-a-number")
        monkeypatch.delenv("RESUMEVOICE_LLM_TEMPERATURE", raising=False)
        c = CoreConfig()
        assert c.vad_threshold == 0.5
