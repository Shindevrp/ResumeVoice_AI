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
    return {"status": "ready", "pipeline": True}


@router.get("/health/live")
async def liveness():
    return {"status": "alive"}
