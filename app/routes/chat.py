from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/chat", tags=["chat"])


def _system_prompt(pipeline) -> str:
    content = (
        "You are ResumeVoice AI, a real-time conversational assistant. "
        "Respond concisely and naturally. Keep responses short, "
        "conversational, and human-like."
    )
    resume = getattr(pipeline, "resume", None)
    if resume is not None:
        content += "\n\n" + resume.to_prompt_block()
    return content


@router.post("/stream")
async def chat_stream(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    pipeline = request.app.state.pipeline
    if pipeline is None:
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
    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if pipeline.llm is None:
        raise HTTPException(status_code=503, detail="LLM not configured")

    messages = [
        {"role": "system", "content": _system_prompt(pipeline)},
        {"role": "user", "content": message},
    ]

    text = await pipeline.llm.generate(messages)
    elapsed = time.perf_counter() - t0
    return {"response": text, "elapsed": round(elapsed, 2)}
