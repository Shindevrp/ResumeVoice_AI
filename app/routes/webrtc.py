from __future__ import annotations

import asyncio
import json
import os
import struct
import uuid

from aiortc import RTCConfiguration, RTCIceCandidate, RTCIceServer, RTCPeerConnection
from aiortc import RTCSessionDescription
from aiortc.mediastreams import AudioFrame, MediaStreamTrack
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.session_registry import register as register_session
from app.session_registry import unregister as unregister_session
from core.pipeline import PipelineEvent, StreamingPipeline
from core.state import DialogueState, SessionState
from modules.memory.retrieval import RetrievalModule
from modules.memory.session import SessionMemory
from utils.audio import resample_pcm
from utils.logger import get_logger

logger = get_logger("webrtc")

router = APIRouter(prefix="/ws", tags=["webrtc"])

MAX_FRAME_BYTES = 65_536
MAX_SESSION_BYTES = 200 * 1024 * 1024
MAX_TEXT_CHARS = 2000
_session_bytes: dict[str, int] = {}


def build_ice_servers() -> list[dict[str, str]]:
    """ICE (STUN/TURN) servers for the platform deployment, from env.

    Set RESUMEVOICE_STUN_URL for a custom STUN and
    RESUMEVOICE_TURN_URL / RESUMEVOICE_TURN_USERNAME /
    RESUMEVOICE_TURN_CREDENTIAL to add a TURN relay. Used both by the
    WebRTC peer (server side) and by /webrtc/config for browser clients.
    """
    servers = [
        {
            "urls": os.getenv(
                "RESUMEVOICE_STUN_URL", "stun:stun.l.google.com:19302"
            ).strip()
        }
    ]
    turn_url = os.getenv("RESUMEVOICE_TURN_URL", "").strip()
    if turn_url:
        servers.append(
            {
                "urls": turn_url,
                "username": os.getenv("RESUMEVOICE_TURN_USERNAME", ""),
                "credential": os.getenv("RESUMEVOICE_TURN_CREDENTIAL", ""),
            }
        )
    return servers


def _make_ice_configuration() -> RTCConfiguration:
    return RTCConfiguration(
        iceServers=[RTCIceServer(**s) for s in build_ice_servers()]
    )


class TTSTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=128)
        self._started = False

    def push_pcm(self, pcm: bytes, sample_rate: int) -> None:
        try:
            self._queue.put_nowait(_new_audio_frame(pcm, sample_rate))
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
        except TimeoutError:
            return _new_audio_frame(b"\x00" * 960 * 2, 48000)


def _strip_wav_header(data: bytes) -> bytes:
    if len(data) < 44:
        return data
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return data[44:]
    return data


def _extract_wav_info(data: bytes) -> tuple[int, int]:
    if len(data) < 44 or data[:4] != b"RIFF":
        return 16000, 16
    sample_rate = struct.unpack_from("<I", data, 24)[0]
    bits_per_sample = struct.unpack_from("<H", data, 34)[0]
    return sample_rate, bits_per_sample


def _new_audio_frame(data: bytes, sample_rate: int) -> AudioFrame:
    """Build a 16-bit mono AudioFrame compatible with installed aiortc.

    aiortc 1.15 (av-backed) rejects keyword construction (data=...); use the
    geometry constructor + plane copy, and fall back for older releases that
    only accept the keyword form.
    """
    try:
        frame = AudioFrame(format="s16", layout="mono", samples=len(data) // 2)
        frame.sample_rate = sample_rate
        frame.planes[0].update(data)
        return frame
    except TypeError:
        return AudioFrame(data=data, sample_rate=sample_rate, channels=1)


def _candidate_to_payload(cand: RTCIceCandidate) -> dict:
    return {
        "candidate": "candidate:" + candidate_to_sdp(cand),
        "sdpMid": cand.sdpMid or "0",
        "sdpMLineIndex": 0 if cand.sdpMLineIndex is None else cand.sdpMLineIndex,
    }


def _candidate_from_payload(cand: dict) -> RTCIceCandidate:
    """Build an RTCIceCandidate from a signaling payload.

    Browsers serialize candidates as a single SDP line under the "candidate"
    key (e.g. ``"candidate:3853… 1 udp 2122260223 192.168.1.97 48867 typ host …"``).
    Earlier integrations sent a flat dict of fields instead. Support both, and
    default to the single audio transceiver (sdpMid "0", index 0) when the
    client omits them; aiortc rejects candidates that carry neither.
    """
    candidate_str = cand.get("candidate")
    if isinstance(candidate_str, str) and candidate_str:
        if candidate_str.startswith("candidate:"):
            candidate_str = candidate_str.split(":", 1)[1]
        try:
            parsed = candidate_from_sdp(candidate_str)
            parsed.sdpMid = cand.get("sdpMid") or "0"
            parsed.sdpMLineIndex = cand.get("sdpMLineIndex", 0)
            return parsed
        except (AssertionError, ValueError, IndexError):
            pass
    return RTCIceCandidate(
        component=cand.get("component", 1),
        foundation=cand.get("foundation", "0"),
        ip=cand.get("ip", ""),
        port=cand.get("port", 0),
        priority=cand.get("priority", 0),
        protocol=cand.get("protocol", "udp"),
        type=cand.get("type", "host"),
        sdpMid=cand.get("sdpMid") or "0",
        sdpMLineIndex=cand.get("sdpMLineIndex", 0),
    )


@router.websocket("/signal")
async def webrtc_signal(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"webrtc session {session_id} connected")

    pipeline: StreamingPipeline | None = websocket.app.state.pipeline
    if pipeline is None:
        await websocket.send_json({"type": "error", "text": "Pipeline not initialized"})
        await websocket.close(1011)
        return

    pc = RTCPeerConnection(configuration=_make_ice_configuration())
    tts_track = TTSTrack()
    pc.addTrack(tts_track)

    session = SessionState(session_id=session_id)
    memory = SessionMemory()
    retrieval = RetrievalModule()
    pipeline.register_session(session_id, memory, retrieval)
    register_session(session)

    # Send welcome message on connection
    await websocket.send_json({
        "type": "system",
        "text": "Welcome! Greetings, I am Shinde Vinayak Rao Patil."
    })

    async def pump_output():
        interrupted = False
        async for msg in pipeline.output_stream(session_id):
            try:
                if msg.session_id != session_id:
                    continue

                if msg.event == PipelineEvent.SPEECH_START:
                    interrupted = True
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
                        # Playable WAV per chunk over the signaling WS so
                        # programmatic (API) clients can play TTS audio without
                        # decoding Opus/RTP. Mirrors /ws/audio behavior.
                        await websocket.send_bytes(msg.data)
                        pcm = _strip_wav_header(msg.data)
                        sr, _ = _extract_wav_info(msg.data)
                        if sr == 0:
                            sr = 16000
                        # WebRTC expects 48kHz Opus; resample Piper output to 48kHz
                        if sr != 48000:
                            pcm = resample_pcm(pcm, sr, 48000)
                            sr = 48000
                        tts_track.push_pcm(pcm, sr)

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
                    tts_track.flush()
                    session.set_state(DialogueState.LISTENING)
                    await websocket.send_json({"type": "interrupt"})

                elif msg.event == PipelineEvent.ERROR:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "text": str(msg.data),
                        }
                    )

            except Exception as e:
                logger.error(
                    f"session {session_id} output pump error: {e}"
                )
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

                arr = frame.to_ndarray()
                pcm = arr.tobytes()

                if frame.sample_rate != 16000:
                    from utils.audio import resample_pcm

                    pcm = resample_pcm(pcm, frame.sample_rate, 16000)

                if 0 < len(pcm) <= MAX_FRAME_BYTES:
                    total = _session_bytes.get(session_id, 0) + len(pcm)
                    if total > MAX_SESSION_BYTES:
                        logger.warning(
                            f"session {session_id} exceeded audio budget; "
                            f"dropping {len(pcm)} bytes"
                        )
                    else:
                        _session_bytes[session_id] = total
                        await pipeline.push_audio(pcm, session_id)
            except (asyncio.CancelledError, Exception) as e:
                logger.debug(f"session {session_id} audio track done: {e}")
                break

    @pc.on("icecandidate")
    async def on_ice_candidate(candidate: RTCIceCandidate) -> None:
        if candidate is None:
            return
        try:
            await websocket.send_json(
                {"type": "ice", "candidate": _candidate_to_payload(candidate)}
            )
        except Exception:
            pass

    @pc.on("iceconnectionstatechange")
    async def on_ice_state() -> None:
        session.ice_connection_state = pc.iceConnectionState
        logger.info(f"session {session_id} ice state: {pc.iceConnectionState}")
        if pc.iceConnectionState in ("failed", "closed", "disconnected"):
            await pc.close()

    try:
        while True:
            raw = await websocket.receive()

            if raw.get("type") == "websocket.disconnect":
                break

            if "text" in raw:
                try:
                    data = json.loads(raw["text"])
                    msg_type = data.get("type")

                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

                    elif msg_type == "interrupt":
                        tts_track.flush()
                        await pipeline.signal_interrupt(session_id)

                    elif msg_type == "text":
                        text = data.get("text", "")
                        if isinstance(text, str):
                            text = text.strip()
                        if not text:
                            await websocket.send_json(
                                {"type": "error", "text": "empty text message"}
                            )
                        elif len(text) > MAX_TEXT_CHARS:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "text": f"text exceeds {MAX_TEXT_CHARS} chars",
                                }
                            )
                        else:
                            logger.info(
                                f"session {session_id} text input: {text[:80]!r}"
                            )
                            asyncio.create_task(
                                pipeline.push_text(text, session_id)
                            )

                    elif msg_type == "offer":
                        offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                        await pc.setRemoteDescription(offer)
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        await websocket.send_json(
                            {
                                "type": "answer",
                                "sdp": pc.localDescription.sdp,
                            }
                        )
                        logger.info(f"session {session_id} webrtc connected")

                    elif msg_type == "ice":
                        candidate = _candidate_from_payload(data["candidate"])
                        await pc.addIceCandidate(candidate)

                except json.JSONDecodeError:
                    logger.debug(f"session {session_id} dropped invalid JSON frame")

    except WebSocketDisconnect:
        logger.info(f"session {session_id} disconnected")
    except Exception as e:
        logger.error(f"session {session_id} error: {e}")
    finally:
        pump_task.cancel()
        pipeline.unregister_session(session_id)
        _session_bytes.pop(session_id, None)
        unregister_session(session_id)
        await pc.close()
        logger.info(f"session {session_id} cleaned up")
