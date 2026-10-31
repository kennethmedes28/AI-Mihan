"""The FastAPI application.

    uv run uvicorn main:app --reload        # development
    uv run uvicorn main:app --host 0.0.0.0  # behind a proxy; set TRUST_PROXY_HEADERS=true there

Endpoints (see docs/backend_architecture.md section 4):

    POST /chat                  one message through the chat graph
    POST /reports, GET /reports community flood reports (unverified)
    POST /requests              supply and assistance requests (write only)
    GET  /evacuation-centers    search the centers
    GET  /advisories/latest     current PAGASA bulletins
    GET  /health, /health/ready what works
    /staff/...                  review reports and requests (only when STAFF_API_KEY is set)
    WS   /ws/voice, /ws/camera  live speech-to-text and the webcam flood check

Interactive documentation is served at /docs.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.rate_limit import RateLimiter
from src.api.routers import advisories, chat, evacuation, health, help_requests, reports, staff
from src.api.settings import Settings, load_settings
from src.api.startup import real_services
from video.camera_router import create_camera_router
from voice.websocket_router import create_voice_router


async def validation_error_response(request: Request, error: RequestValidationError) -> JSONResponse:
    """The 422 answer, with only where each problem is and what it is.

    FastAPI's default also repeats the value that was sent. Here that is left out, because it can be
    a phone number or a place, and because a value such as NaN cannot be written back as JSON at all.
    """
    problems = [{"loc": item["loc"], "msg": item["msg"], "type": item["type"]} for item in error.errors()]
    return JSONResponse(status_code=422, content={"detail": problems})


def create_app(settings: Settings | None = None, lifespan=real_services) -> FastAPI:
    """Build the app. Tests pass their own ``settings`` and a ``lifespan`` that sets fakes on ``app.state``."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Flood-Gentic API",
        version="0.1.0",
        description="Flood assistant for Metro Manila. Reports and requests are unverified community input.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_exception_handler(RequestValidationError, validation_error_response)
    app.state.limiters = {
        "chat": RateLimiter(settings.chat_per_minute, window_seconds=60),
        "reports": RateLimiter(settings.reports_per_hour, window_seconds=3600),
        "requests": RateLimiter(settings.requests_per_hour, window_seconds=3600),
        "staff": RateLimiter(settings.staff_per_minute, window_seconds=60),
    }

    # Browsers may call the API only from the listed origins. This does not cover the WebSockets.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    for module in (health, chat, reports, help_requests, evacuation, advisories):
        app.include_router(module.router)
    if settings.staff_api_key:  # without a key the staff endpoints simply do not exist
        app.include_router(staff.router)
    app.include_router(create_voice_router())
    app.include_router(create_camera_router())
    return app
