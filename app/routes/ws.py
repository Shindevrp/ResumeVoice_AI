from __future__ import annotations

import asyncio
import json
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.session_registry import register as register_session
from app.session_registry import unregister as unregister_session
from core.pipeline import PipelineEvent, StreamingPipeline
from core.state import DialogueState, SessionState
from modules.memory.retrieval import RetrievalModule
from modules.memory.session import SessionMemory
from utils.logger import get_logger

logger = get_logger("ws")

router = APIRouter(prefix="/ws", tags=["websocket"])

MAX_FRAME_BYTES = 65_536
MAX_SESSION_BYTES = 200 * 1024 * 1024
_session_bytes: dict[str, int] = {}


@router.websocket("/audio")
async def audio_websocket(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"session {session_id} connected")

    pipeline: StreamingPipeline | None = websocket.app.state.pipeline
    if pipeline is None:
        await websocket.send_json({"type": "error", "text": "Pipeline not initialized"})
        await websocket.close(1011)
        return

    session = SessionState(session_id=session_id)
    memory = SessionMemory()
    retrieval = RetrievalModule()

    pipeline.register_session(session_id, memory, retrieval)
    register_session(session)

    async def pump_output():
        interrupted = False
        async for msg in pipeline.output_stream():
            try:
                if msg.session_id != session_id:
                    continue

                if msg.event == PipelineEvent.SPEECH_START:
                    interrupted = False
                    session.set_state(DialogueState.LISTENING)
                    await websocket.send_json({"type": "speech_start"})

                elif msg.event == PipelineEvent.SPEECH_END:
                    session.set_state(DialogueState.PROCESSING)
                    await websocket.send_json({"type": "speech_end"})

                elif msg.event == PipelineEvent.PARTIAL_TRANSCRIPT:
                    await websocket.send_json(
                        {
                            "type": "partial_transcript",
                            "text": str(msg.data),
                        }
                    )

                elif msg.event == PipelineEvent.FINAL_TRANSCRIPT:
                    interrupted = False
                    session.add_user_turn(str(msg.data))
                    session.last_activity = time.time()
                    session.update(pipeline.context(session_id))
                    await websocket.send_json(
                        {
                            "type": "transcript",
                            "text": str(msg.data),
                        }
                    )

                elif msg.event == PipelineEvent.LLM_TOKEN:
                    session.set_state(DialogueState.INTERRUPTIBLE)
                    await websocket.send_json(
                        {
                            "type": "llm_token",
                            "token": str(msg.data),
                        }
                    )

                elif msg.event == PipelineEvent.LLM_DONE:
                    session.add_ai_turn(str(msg.data))
                    await websocket.send_json(
                        {
                            "type": "llm_done",
                            "text": str(msg.data),
                        }
                    )

                elif msg.event == PipelineEvent.TTS_CHUNK:
                    if isinstance(msg.data, bytes):
                        if interrupted:
                            continue
                        await websocket.send_bytes(msg.data)

                elif msg.event == PipelineEvent.RESPONSE_DELAY:
                    await websocket.send_json(
                        {
                            "type": "status",
                            "text": f"waiting {msg.data}s",
                        }
                    )

                elif msg.event == PipelineEvent.TTS_DONE:
                    session.set_state(DialogueState.IDLE)
                    await websocket.send_json({"type": "tts_done"})

                elif msg.event == PipelineEvent.BACKCHANNEL:
                    await websocket.send_json(
                        {
                            "type": "backchannel",
                            "text": str(msg.data),
                        }
                    )

                elif msg.event == PipelineEvent.INTERRUPT:
                    interrupted = True
                    session.set_state(DialogueState.LISTENING)
                    await websocket.send_json({"type": "interrupt"})

                elif msg.event == PipelineEvent.ERROR:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "text": str(msg.data),
                        }
                    )

            except Exception as ex:
                logger.warning(f"session {session_id} pump_output error: {ex}")
                break

    pump_task = None
    try:
        pump_task = asyncio.create_task(pump_output())

        while True:
            raw = await websocket.receive()

            if raw.get("type") == "websocket.disconnect":
                code = raw.get("code", 1005)
                logger.info(f"session {session_id} client disconnect code={code}")
                break

            if "bytes" in raw:
                chunk = raw["bytes"]
                if isinstance(chunk, bytes) and 0 < len(chunk) <= MAX_FRAME_BYTES:
                    total = _session_bytes.get(session_id, 0) + len(chunk)
                    if total > MAX_SESSION_BYTES:
                        logger.warning(
                            f"session {session_id} exceeded audio budget; "
                            f"dropping {len(chunk)} bytes"
                        )
                    else:
                        _session_bytes[session_id] = total
                        await pipeline.push_audio(chunk, session_id)

            elif "text" in raw:
                try:
                    data = json.loads(raw["text"])
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif data.get("type") == "interrupt":
                        await pipeline.signal_interrupt(session_id)
                except json.JSONDecodeError:
                    logger.debug(f"session {session_id} dropped invalid JSON frame")

    except WebSocketDisconnect:
        logger.info(f"session {session_id} disconnected")
    except Exception as e:
        logger.error(f"session {session_id} error: {e}")
    finally:
        if pump_task:
            pump_task.cancel()
        pipeline.unregister_session(session_id)
        _session_bytes.pop(session_id, None)
        unregister_session(session_id)
        logger.info(f"session {session_id} cleaned up")
