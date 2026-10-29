"""Tiny development server for trying the webcam flood check in a browser.

    cd backend
    uv run uvicorn video.demo_app:app --port 8000

Then open http://localhost:8000 (a webcam needs https or localhost). This is only a demo; the real
app will mount ``create_camera_router()`` in ``main.py``.
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse

from video.camera_router import create_camera_router

load_dotenv()

app = FastAPI(title="Flood camera demo")
app.include_router(create_camera_router())


@app.get("/")
def camera_page() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "camera.html")
