from __future__ import annotations

import time

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request):
    pipeline = request.app.state.pipeline
    start_time = getattr(request.app.state, "start_time", time.time())

    status = "healthy" if pipeline is not None else "degraded"

    return {
        "status": status,
        "pipeline": pipeline is not None,
        "uptime_seconds": round(time.time() - start_time, 1),
        "version": "0.2.0",
    }


@router.get("/health/ready")
async def readiness(request: Request):
    pipeline = request.app.state.pipeline
    if pipeline is None:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "pipeline": False},
        )
    # Basic provider readiness checks
    try:
        tts_ready = hasattr(pipeline.tts, "voice")
        # Trigger voice load
        _ = pipeline.tts.voice
        stt_ready = pipeline.stt is not None
        vad_ready = pipeline.vad is not None
        ready = pipeline._running and tts_ready and stt_ready and vad_ready
        status = "ready" if ready else "degraded"
        code = 200 if ready else 503
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=code,
            content={
                "status": status,
                "pipeline": True,
                "running": pipeline._running,
                "tts_ready": tts_ready,
                "stt_ready": stt_ready,
                "vad_ready": vad_ready,
            },
        )
    except Exception:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "pipeline": True},
        )


@router.get("/health/live")
async def liveness():
    return {"status": "alive"}
