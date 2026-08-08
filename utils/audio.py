from __future__ import annotations

import numpy as np


def pcm_to_float(pcm: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to float32 array normalized to [-1, 1]."""
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm(samples: np.ndarray) -> bytes:
    """Convert float32 array [-1, 1] to 16-bit PCM bytes.

    Inverse of `pcm_to_float`: uses the same 32768 scale so a
    round-trip reproduces the original samples exactly.
    """
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32768).astype(np.int16).tobytes()


def rms_energy(pcm: bytes) -> float:
    """Compute RMS energy of a PCM audio chunk."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))


def is_silence(pcm: bytes, threshold: float = 500.0, sample_rate: int = 16000) -> bool:
    """Check if PCM chunk is silence based on RMS energy."""
    return rms_energy(pcm) < threshold


def normalize_audio(pcm: bytes, target_level: float = 0.3) -> bytes:
    """Normalize PCM audio to a target RMS level."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return pcm
    current_rms = float(np.sqrt(np.mean(samples**2)))
    if current_rms < 1.0:
        return pcm
    gain = target_level * 32767 / max(current_rms, 1.0)
    gain = min(gain, 5.0)
    samples = np.clip(samples * gain, -32767, 32767)
    return samples.astype(np.int16).tobytes()


def audio_duration_seconds(samples: int, sample_rate: int = 16000) -> float:
    """Return audio duration in seconds for a sample count."""
    return samples / sample_rate


def resample_pcm(pcm: bytes, orig_rate: int, target_rate: int) -> bytes:
    """Resample PCM audio from orig_rate to target_rate."""
    if orig_rate == target_rate:
        return pcm
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    duration = len(samples) / orig_rate
    target_len = int(duration * target_rate)
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, target_len),
        np.arange(len(samples)),
        samples,
    )
    return resampled.astype(np.int16).tobytes()
