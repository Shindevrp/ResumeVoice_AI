from __future__ import annotations

import numpy as np


class ProsodyExtractor:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30) -> None:
        self.sample_rate = sample_rate
        self.frame_size = sample_rate * frame_ms // 1000

    def extract(self, audio_chunk: bytes) -> dict[str, float]:
        samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return {"pitch": 0.0, "energy": 0.0, "zero_crossing_rate": 0.0}

        energy = float(np.sqrt(np.mean(samples**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2)
        pitch = self._estimate_pitch(samples)

        return {"pitch": pitch, "energy": energy, "zero_crossing_rate": zcr}

    def _estimate_pitch(self, samples: np.ndarray) -> float:
        autocorr = np.correlate(samples, samples, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        min_lag = int(self.sample_rate / 500)
        max_lag = int(self.sample_rate / 50)
        segment = autocorr[min_lag:max_lag]

        if len(segment) == 0 or np.max(segment) == 0:
            return 0.0

        peak_idx = int(np.argmax(segment)) + min_lag
        if peak_idx > 0:
            return float(self.sample_rate / peak_idx)
        return 0.0
