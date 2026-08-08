from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from modules.tts.prosody import ProsodyProfile


class TTSProvider(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def synthesize_stream(
        self,
        text_chunks: AsyncGenerator[str, None],
        prosody: ProsodyProfile | None = None,
    ) -> AsyncGenerator[bytes, None]:
        raise NotImplementedError

    @abstractmethod
    async def synthesize(
        self, text: str, prosody: ProsodyProfile | None = None
    ) -> bytes:
        raise NotImplementedError
