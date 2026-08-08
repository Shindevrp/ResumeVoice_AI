from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from modules.dialogue.prompts import build_system_prompt

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_BODY_BYTES = 16_384
LLM_TIMEOUT = 60.0


async def _read_json(request: Request) -> dict:
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(
            status_code=413, detail=f"body exceeds {MAX_BODY_BYTES} bytes"
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return data


def _system_prompt(pipeline) -> str:
    content = build_system_prompt()
    resume = getattr(pipeline, "resume", None)
    if resume is not None:
        content += "\n\n" + resume.to_prompt_block()
    return content


@router.post("/stream")
async def chat_stream(request: Request):
    body = await _read_json(request)
    message = body.get("message", "")
    if isinstance(message, str):
        message = message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    pipeline = request.app.state.pipeline
    if pipeline is None or pipeline.llm is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    async def event_stream():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"

        messages = [
            {"role": "system", "content": _system_prompt(pipeline)},
            {"role": "user", "content": message},
        ]

        full = ""
        async for token in pipeline.llm.generate_stream(messages):
            full += token
            yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'text': full})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/")
async def chat_once(request: Request):
    import time

    t0 = time.perf_counter()
    body = await _read_json(request)
    message = body.get("message", "")
    if isinstance(message, str):
        message = message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    pipeline = request.app.state.pipeline
    if pipeline is None or pipeline.llm is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if not hasattr(pipeline.llm, "generate"):
        raise HTTPException(
            status_code=503, detail="LLM does not support one-shot chat"
        )

    messages = [
        {"role": "system", "content": _system_prompt(pipeline)},
        {"role": "user", "content": message},
    ]

    try:
        text = await asyncio.wait_for(
            pipeline.llm.generate(messages), timeout=LLM_TIMEOUT
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="LLM request timed out")
    elapsed = time.perf_counter() - t0
    return {"response": text, "elapsed": round(elapsed, 2)}
