from __future__ import annotations

import struct

import pytest

from utils.audio import (
    audio_duration_seconds,
    float_to_pcm,
    is_silence,
    normalize_audio,
    pcm_to_float,
    resample_pcm,
    rms_energy,
)


def _pcm(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}h", *values)


class TestRmsEnergy:
    def test_empty(self) -> None:
        assert rms_energy(b"") == 0.0

    def test_digital_silence(self) -> None:
        assert rms_energy(b"\x00\x00" * 100) == 0.0

    def test_constant_amplitude(self) -> None:
        assert rms_energy(_pcm([1000] * 100)) == pytest.approx(1000.0, abs=0.01)


class TestIsSilence:
    def test_silence_true(self) -> None:
        assert is_silence(b"\x00\x00" * 100)

    def test_audio_false(self) -> None:
        assert not is_silence(_pcm([1000] * 100))


class TestFloatPcmRoundtrip:
    def test_roundtrip(self) -> None:
        data = _pcm([0, 1000, -1000, 32767, -32768])
        assert float_to_pcm(pcm_to_float(data)) == data


class TestNormalizeAudio:
    def test_silence_untouched(self) -> None:
        assert normalize_audio(b"\x00\x00" * 10) == b"\x00\x00" * 10

    def test_loud_audio_reduced(self) -> None:
        loud = _pcm([20000] * 100)
        norm = normalize_audio(loud, target_level=0.3)
        assert rms_energy(norm) < rms_energy(loud)


class TestResamplePcm:
    def test_same_rate_identity(self) -> None:
        data = b"\x00\x01\x02\x03"
        assert resample_pcm(data, 16000, 16000) == data

    def test_downsample_shortens(self) -> None:
        data = _pcm([500] * 160)
        out = resample_pcm(data, 16000, 8000)
        assert len(out) == 80 * 2

    def test_upsample_lengthens(self) -> None:
        data = _pcm([500] * 80)
        out = resample_pcm(data, 8000, 16000)
        assert len(out) == 160 * 2


class TestAudioDuration:
    def test_seconds(self) -> None:
        assert audio_duration_seconds(16000, 16000) == 1.0
        assert audio_duration_seconds(8000, 16000) == 0.5
