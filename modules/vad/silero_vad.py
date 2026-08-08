from __future__ import annotations

import numpy as np
import torch


class SileroVAD:
    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        device: str = "cpu",
    ) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_size = sample_rate * frame_ms // 1000
        self.device = device
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str | None) -> torch.nn.Module:
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        model.eval()
        model.to(self.device)
        return model

    def reset(self) -> None:
        self._state = None
        reset_fn = getattr(self.model, "reset_states", None)
        if callable(reset_fn):
            reset_fn()

    def is_speech(self, audio_chunk: bytes) -> bool:
        audio = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        num_samples = 512 if self.sample_rate == 16000 else 256

        if len(audio) < num_samples:
            return False

        with torch.no_grad():
            for start in range(0, len(audio), num_samples):
                frame = audio[start : start + num_samples]
                if len(frame) < num_samples:
                    break
                audio_tensor = torch.from_numpy(frame).unsqueeze(0).to(self.device)
                prob = self.model(audio_tensor, self.sample_rate)
                if isinstance(prob, tuple):
                    prob = prob[0]
                if prob.item() >= self.threshold:
                    return True
        return False
