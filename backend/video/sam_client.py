"""Client for Meta's SAM 3.1 model: finds every object that matches a short phrase in a photo or video.

SAM answers with a box and a pixel mask for each match, plus a stable object id that follows the
object from frame to frame in a video. Use it for "where is the flood water?" or "how many people
are in the picture?", not for reading text or describing a scene.

    async with SamClient(api_key) as sam:
        result = await sam.segment("flood water", ImageInput(path="street.jpg"))
        for record in result.records:        # boxes and masks, see meta_sam_parser
            ...

One request looks for ONE phrase (SAM's own rule), so ask once per kind of object.
API notes (https://dev.meta.ai/docs/sam/segmenting): ``POST /v1/responses``, model ``sam-3.1``, the
answer is streamed as text that ``meta_sam_parser`` turns into typed records. Videos must be MP4 or
MOV (H.264, H.265 or MPEG-4), up to about 10 minutes, and are billed per frame (at least 50 frames).
"""

import base64
import json
import mimetypes
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
from meta_sam_parser import (
    ImageSegmentationResult,
    VideoSegmentationResult,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)

BASE_URL = "https://api.meta.ai/v1"
MODEL = "sam-3.1"
REQUEST_TIMEOUT_SECONDS = 300.0  # a long video streams for a while
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # guard for what we are willing to base64-encode ourselves


@dataclass(frozen=True)
class ImageInput:
    """A photo: a public ``url``, a local ``path`` or raw ``data`` bytes (give exactly one)."""

    url: str | None = None
    path: str | Path | None = None
    data: bytes | None = None
    mime_type: str = "image/jpeg"  # only used with ``data``

    def image_url(self) -> str:
        """The value SAM expects: the public URL, or the file as a base64 data URL."""
        given = [value for value in (self.url, self.path, self.data) if value is not None]
        if len(given) != 1:
            raise ValueError("Give exactly one of url, path or data.")
        if self.url is not None:
            return self.url

        if self.path is not None:
            raw = Path(self.path).read_bytes()
            mime_type = mimetypes.guess_type(str(self.path))[0] or "image/jpeg"
        else:
            raw, mime_type = self.data, self.mime_type
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("Image is too large; resize it first.")
        return f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"


@dataclass(frozen=True)
class VideoInput:
    """A video: a public ``url`` or a local ``path`` (uploaded once, then used by file id)."""

    url: str | None = None
    path: str | Path | None = None
    file_id: str | None = None  # set when the file was uploaded before


class SamError(RuntimeError):
    """SAM refused the request or the stream failed."""


class SamClient:
    """Async client for ``sam-3.1``. Use it in an ``async with`` block."""

    def __init__(self, api_key: str, base_url: str = BASE_URL):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)

    async def __aenter__(self) -> "SamClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._http.aclose()

    async def segment(
        self, concept: str, media: ImageInput | VideoInput
    ) -> ImageSegmentationResult | VideoSegmentationResult:
        """Find every ``concept`` (a short noun phrase such as "flood water") in the media."""
        if isinstance(media, VideoInput):
            media_part = {"type": "input_video", **await self._video_reference(media)}
            parser_format = video_segmentation_format()
        else:
            media_part = {"type": "input_image", "image_url": media.image_url()}
            parser_format = image_segmentation_format()

        body = {
            "model": MODEL,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": concept}, media_part],
                }
            ],
            "stream": True,
            # Confidence lets callers ignore weak matches. Sent as strings, as the API requires.
            "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
        }
        parsed = parse_responses_stream(self._stream_events(body), parser_format)
        async for _snapshot in parsed:  # reading to the end is what finishes the parse
            pass
        return await parsed.final_result()

    async def upload_video(self, path: str | Path) -> str:
        """Upload a local video once and return its file id (reuse it for several concepts)."""
        with open(path, "rb") as file:
            response = await self._http.post(
                f"{self._base_url}/files",
                headers=self._headers(),
                data={"purpose": "user_data"},
                files={"file": (Path(path).name, file)},
            )
        if response.status_code != 200:
            raise SamError(f"Video upload failed ({response.status_code}): {response.text[:300]}")
        return response.json()["id"]

    async def _video_reference(self, video: VideoInput) -> dict:
        if video.file_id:
            return {"file_id": video.file_id}
        if video.url:
            return {"video_url": video.url}
        if video.path:
            return {"file_id": await self.upload_video(video.path)}
        raise ValueError("Give a url, path or file_id for the video.")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _stream_events(self, body: dict) -> AsyncIterator[dict]:
        """Yield the server-sent events of one ``/responses`` request as dictionaries."""
        headers = {**self._headers(), "Content-Type": "application/json", "Accept": "text/event-stream"}
        async with self._http.stream("POST", f"{self._base_url}/responses", headers=headers, json=body) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode("utf-8", errors="replace")[:300]
                raise SamError(f"SAM request failed ({response.status_code}): {detail}")
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break
                yield json.loads(data)
