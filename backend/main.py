"""Entry point of the Flood-Gentic API.

    uv run uvicorn main:app --reload

The application itself is built in ``src/api/app.py``.
"""

from src.api.app import create_app

app = create_app()
