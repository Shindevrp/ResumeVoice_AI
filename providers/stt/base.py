from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


class STTProvider(ABC):
    @abstractmethod
    async def transcribe_stream(
        self, audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[str, None]:
        raise NotImplementedError

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> str:
        raise NotImplementedError
