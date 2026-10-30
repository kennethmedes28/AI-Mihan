"""GET /health (is it running, and what works) and GET /health/ready (is it fit to receive traffic)."""

from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import HealthOut

router = APIRouter(tags=["health"])


def run_checks(request: Request) -> dict[str, str]:
    """Check each dependency once. Failures are reported by name, never with their details."""
    state = request.app.state
    checks = {}

    get_connection = getattr(state, "get_connection", None)
    try:
        get_connection().execute("SELECT 1")
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "unavailable"

    checks["chat_graph"] = "ok" if getattr(state, "graph", None) is not None else "unavailable"
    checks["chat_model"] = "configured" if getattr(state, "llm_ready", False) else "missing (weather and document answers are off)"
    checks["document_search"] = "ready" if getattr(state, "retriever_ready", False) else "off"
    return checks


@router.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    """Always 200 while the process runs; ``status`` says whether anything is wrong."""
    checks = run_checks(request)
    healthy = checks["database"] == "ok" and checks["chat_graph"] == "ok"
    return HealthOut(status="ok" if healthy else "degraded", checks=checks)


@router.get("/health/ready", response_model=HealthOut)
def ready(request: Request) -> HealthOut:
    """503 unless the database and the chat graph work. Use this one for a load balancer."""
    result = health(request)
    if result.status != "ok":
        raise HTTPException(status_code=503, detail=result.model_dump())
    return result
