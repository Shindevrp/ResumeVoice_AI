from __future__ import annotations

import asyncio
import numpy as np
from typing import AsyncGenerator

from faster_whisper import WhisperModel

from providers.stt.base import STTProvider


class FasterWhisperSTT(STTProvider):
    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
    ) -> None:
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type,
            num_workers=1,
        )
        self.language = language

    async def transcribe_stream(
        self, audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[str, None]:
        buffer = bytearray()
        async for chunk in audio_chunks:
            buffer.extend(chunk)
            if len(buffer) < 16000:
                continue
            chunk_to_process = bytes(buffer)
            buffer.clear()
            result = await asyncio.to_thread(
                self._transcribe_segment, chunk_to_process
            )
            if result:
                yield result

    def _transcribe_segment(self, audio_bytes: bytes) -> str | None:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            best_of=1,
            vad_filter=False,
        )
        text = " ".join(seg.text for seg in segments)
        return text.strip() or None

    async def transcribe(self, audio_bytes: bytes) -> str:
        return await asyncio.to_thread(self._transcribe_segment, audio_bytes) or ""