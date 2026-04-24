# runtime/routes/scheduler.py
from fastapi import APIRouter, Request

router = APIRouter(prefix="/v1", tags=["scheduler"])


@router.get("/scheduler/status")
def scheduler_status(req: Request) -> dict:
    sch = getattr(req.app.state, "scheduler", None)
    if not sch:
        return {"running": False, "ticks": 0, "interval": None}
    return {
        "running": bool(sch._running),
        "ticks": sch.ticks,
        "interval": sch.interval,
    }
