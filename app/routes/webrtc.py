from __future__ import annotations

import asyncio
import json
import struct
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.mediastreams import AudioFrame, MediaStreamTrack

from core.pipeline import StreamingPipeline, PipelineEvent
from core.state import SessionState, DialogueState
from app.session_registry import register as register_session
from app.session_registry import unregister as unregister_session
from modules.memory.session import SessionMemory
from modules.memory.retrieval import RetrievalModule
from utils.logger import get_logger

logger = get_logger("webrtc")

router = APIRouter(prefix="/ws", tags=["webrtc"])

SESSION_TIMEOUT = 300.0
_active_sessions: dict[str, float] = {}


class TTSTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=128)
        self._started = False

    def push_pcm(self, pcm: bytes, sample_rate: int) -> None:
        try:
            frame = AudioFrame(
                data=pcm,
                sample_rate=sample_rate,
                channels=1,
            )
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass

    def flush(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def recv(self) -> AudioFrame:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=0.3)
        except asyncio.TimeoutError:
            silence = AudioFrame(
                data=b"\x00" * 960 * 2,
                sample_rate=48000,
                channels=1,
            )
            return silence


def _strip_wav_header(data: bytes) -> bytes:
    if len(data) < 44:
        return data
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return data[44:]
    return data


def _extract_wav_info(data: bytes) -> tuple[int, int]:
    if len(data) < 44 or data[:4] != b"RIFF":
        return 16000, 16
    num_channels = struct.unpack_from("<H", data, 22)[0]
    sample_rate = struct.unpack_from("<I", data, 24)[0]
    bits_per_sample = struct.unpack_from("<H", data, 34)[0]
    return sample_rate, bits_per_sample


@router.websocket("/signal")
async def webrtc_signal(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    _active_sessions[session_id] = time.time()
    logger.info(f"webrtc session {session_id} connected")

    pipeline: StreamingPipeline | None = websocket.app.state.pipeline
    if pipeline is None:
        await websocket.send_json({"type": "error", "text": "Pipeline not initialized"})
        await websocket.close(1011)
        return

    pc = RTCPeerConnection()
    tts_track = TTSTrack()
    pc.addTrack(tts_track)

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
                    await websocket.send_json({
                        "type": "partial_transcript",
                        "text": str(msg.data),
                    })

                elif msg.event == PipelineEvent.FINAL_TRANSCRIPT:
                    interrupted = False
                    session.add_user_turn(str(msg.data))
                    session.last_activity = time.time()
                    session.update(pipeline.context(session_id))
                    await websocket.send_json({
                        "type": "transcript",
                        "text": str(msg.data),
                    })

                elif msg.event == PipelineEvent.LLM_TOKEN:
                    session.set_state(DialogueState.INTERRUPTIBLE)
                    await websocket.send_json({
                        "type": "llm_token",
                        "token": str(msg.data),
                    })

                elif msg.event == PipelineEvent.LLM_DONE:
                    session.add_ai_turn(str(msg.data))
                    await websocket.send_json({
                        "type": "llm_done",
                        "text": str(msg.data),
                    })

                elif msg.event == PipelineEvent.TTS_CHUNK:
                    if isinstance(msg.data, bytes):
                        if interrupted:
                            continue
                        pcm = _strip_wav_header(msg.data)
                        sr, _ = _extract_wav_info(msg.data)
                        if sr == 0:
                            sr = 16000
                        tts_track.push_pcm(pcm, sr)

                elif msg.event == PipelineEvent.RESPONSE_DELAY:
                    await websocket.send_json({
                        "type": "status",
                        "text": f"waiting {msg.data}s",
                    })

                elif msg.event == PipelineEvent.TTS_DONE:
                    session.set_state(DialogueState.IDLE)
                    await websocket.send_json({"type": "tts_done"})

                elif msg.event == PipelineEvent.BACKCHANNEL:
                    await websocket.send_json({
                        "type": "backchannel",
                        "text": str(msg.data),
                    })

                elif msg.event == PipelineEvent.INTERRUPT:
                    interrupted = True
                    tts_track.flush()
                    session.set_state(DialogueState.LISTENING)
                    await websocket.send_json({"type": "interrupt"})

                elif msg.event == PipelineEvent.ERROR:
                    await websocket.send_json({
                        "type": "error",
                        "text": str(msg.data),
                    })

            except Exception:
                break

    pump_task = asyncio.create_task(pump_output())

    @pc.on("track")
    async def on_track(track: MediaStreamTrack) -> None:
        if track.kind != "audio":
            return
        logger.info(f"session {session_id} received audio track")
        while True:
            try:
                frame = await track.recv()
                _active_sessions[session_id] = time.time()

                arr = frame.to_ndarray()
                pcm = arr.tobytes()

                if frame.sample_rate != 16000:
                    from utils.audio import resample_pcm
                    pcm = resample_pcm(pcm, frame.sample_rate, 16000)

                await pipeline.push_audio(pcm, session_id)
            except (asyncio.CancelledError, Exception) as e:
                logger.debug(f"session {session_id} audio track done: {e}")
                break

    @pc.on("iceconnectionstatechange")
    async def on_ice_state() -> None:
        logger.info(f"session {session_id} ice state: {pc.iceConnectionState}")
        if pc.iceConnectionState in ("failed", "closed", "disconnected"):
            await pc.close()

    try:
        while True:
            raw = await websocket.receive()

            if raw.get("type") == "websocket.disconnect":
                break

            _active_sessions[session_id] = time.time()

            if "text" in raw:
                try:
                    data = json.loads(raw["text"])
                    msg_type = data.get("type")

                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

                    elif msg_type == "interrupt":
                        tts_track.flush()
                        await pipeline.signal_interrupt(session_id)

                    elif msg_type == "offer":
                        offer = RTCSessionDescription(
                            sdp=data["sdp"], type="offer"
                        )
                        await pc.setRemoteDescription(offer)
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        await websocket.send_json({
                            "type": "answer",
                            "sdp": pc.localDescription.sdp,
                        })
                        logger.info(f"session {session_id} webrtc connected")

                    elif msg_type == "ice":
                        cand = data["candidate"]
                        candidate = RTCIceCandidate(
                            component=cand.get("component", 1),
                            foundation=cand.get("foundation", "0"),
                            ip=cand.get("ip", ""),
                            port=cand.get("port", 0),
                            priority=cand.get("priority", 0),
                            protocol=cand.get("protocol", "udp"),
                            type=cand.get("type", "host"),
                        )
                        await pc.addIceCandidate(candidate)

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"session {session_id} disconnected")
    except Exception as e:
        logger.error(f"session {session_id} error: {e}")
    finally:
        pump_task.cancel()
        pipeline.unregister_session(session_id)
        _active_sessions.pop(session_id, None)
        unregister_session(session_id)
        await pc.close()
        logger.info(f"session {session_id} cleaned up")
