from __future__ import annotations

import time

from fastapi import APIRouter, Request

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics(request: Request):
    pipeline = request.app.state.pipeline
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = time.time() - start_time

    result = {
        "uptime_seconds": round(uptime, 1),
        "pipeline_initialized": pipeline is not None,
    }

    if pipeline:
        result["latency"] = pipeline.latency_report()

    return result
