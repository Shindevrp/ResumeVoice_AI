from __future__ import annotations

import asyncio

from providers.llm.vllm_llm import VLLMProvider


class FakeDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, delta=None, message=None) -> None:
        self.delta = delta
        self.message = message


class FakeResponse:
    def __init__(self, choices) -> None:
        self.choices = choices


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeAsyncIterator:
    def __init__(self, items) -> None:
        self.items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return FakeAsyncIterator(
                [FakeResponse([FakeChoice(delta=FakeDelta("hi"))])]
            )
        return FakeResponse([FakeChoice(message=FakeMessage("hello"))])


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


class TestVLLMProvider:
    def test_stream_passes_anti_stutter_extra_body(self) -> None:
        p = VLLMProvider()
        p.client = FakeClient()

        async def collect() -> None:
            async for t in p.generate_stream(
                [{"role": "user", "content": "hi"}]
            ):
                tokens.append(t)

        tokens: list[str] = []
        asyncio.run(collect())

        assert tokens == ["hi"]
        kw = p.client.chat.completions.calls[-1]
        assert kw["temperature"] == 0.5
        assert kw["extra_body"]["repetition_penalty"] == 1.15

    def test_generate_sends_extra_body(self) -> None:
        p = VLLMProvider()
        p.client = FakeClient()

        text = asyncio.run(p.generate([{"role": "user", "content": "hi"}]))

        assert text == "hello"
        kw = p.client.chat.completions.calls[-1]
        assert "repetition_penalty" in kw["extra_body"]
