from __future__ import annotations

from typing import AsyncGenerator

from openai import AsyncOpenAI

from providers.llm.base import LLMProvider


class VLLMProvider(LLMProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
        temperature: float = 0.5,
        max_tokens: int = 256,
        top_p: float = 0.9,
        repetition_penalty: float | None = 1.15,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def _extra_body(self) -> dict[str, object]:
        extra: dict[str, object] = {}
        if self.repetition_penalty is not None:
            extra["repetition_penalty"] = self.repetition_penalty
        if self.presence_penalty:
            extra["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty:
            extra["frequency_penalty"] = self.frequency_penalty
        return extra

    async def generate_stream(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stream=True,
            extra_body=self._extra_body(),
        )
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def generate(self, messages: list[dict[str, str]]) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stream=False,
            extra_body=self._extra_body(),
        )
        return response.choices[0].message.content or ""