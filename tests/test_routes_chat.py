from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes.chat import router


class FakeLLM:
    def __init__(
        self, tokens: list[str] | None = None, text: str = "hello from fake"
    ) -> None:
        self._tokens = tokens or ["hel", "lo ", "world"]
        self._text = text

    async def generate_stream(self, messages):
        for token in self._tokens:
            yield token

    async def generate(self, messages) -> str:
        return self._text


class FakeResume:
    def to_prompt_block(self) -> str:
        return "\n\nPERSONA-BLOCK"


_DEFAULT = object()


class FakePipeline:
    def __init__(
        self,
        llm: object = _DEFAULT,
        resume: FakeResume | None = None,
    ) -> None:
        self.llm = FakeLLM() if llm is _DEFAULT else llm
        self.resume = resume


class LLMNoGenerate:
    async def generate_stream(self, messages):
        yield "x"


def _client(pipeline: FakePipeline) -> TestClient:
    app = FastAPI()
    app.state.pipeline = pipeline
    app.include_router(router)
    return TestClient(app)


def test_chat_once_ok() -> None:
    client = _client(FakePipeline())
    resp = client.post("/chat/", json={"message": "hi"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "hello from fake"
    assert "elapsed" in body


def test_chat_stream_sse() -> None:
    client = _client(FakePipeline())
    with client.stream("POST", "/chat/stream", json={"message": "hi"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert '"type": "start"' in body
    assert '"type": "token"' in body
    assert '"type": "done"' in body
    assert "hello world" in body


def test_chat_missing_message() -> None:
    client = _client(FakePipeline())
    assert client.post("/chat/", json={}).status_code == 400
    assert client.post("/chat/", json={"message": ""}).status_code == 400
    assert client.post("/chat/", json={"message": "   "}).status_code == 400


def test_chat_pipeline_not_initialized() -> None:
    client = _client(FakePipeline(llm=None))
    assert client.post("/chat/", json={"message": "hi"}).status_code == 503
    assert client.post("/chat/stream", json={"message": "hi"}).status_code == 503


def test_chat_llm_without_generate() -> None:
    client = _client(FakePipeline(llm=LLMNoGenerate()))
    assert client.post("/chat/", json={"message": "hi"}).status_code == 503


def test_chat_oversized_body() -> None:
    client = _client(FakePipeline())
    resp = client.post("/chat/", json={"message": "x" * 20_000})
    assert resp.status_code == 413


def test_chat_invalid_json() -> None:
    client = _client(FakePipeline())
    resp = client.post(
        "/chat/",
        content="not json",
        headers={
            "content-type": "application/json",
        },
    )
    assert resp.status_code == 400


def test_system_prompt_includes_resume() -> None:
    from app.routes.chat import _system_prompt

    prompt = _system_prompt(FakePipeline(resume=FakeResume()))
    assert "PERSONA-BLOCK" in prompt
    assert "ResumeVoice" in prompt

    prompt = _system_prompt(FakePipeline(resume=None))
    assert "PERSONA-BLOCK" not in prompt
