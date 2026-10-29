"""Client for Meta's realtime speech-to-text WebSocket (model ``muse-voice-transcribe-1.0``).

One ``RealtimeTranscriber`` is one transcription session:

    async with RealtimeTranscriber(api_key, mode="ENDPOINTING", sample_rate=24000) as transcriber:
        await transcriber.send_audio(pcm_bytes)   # raw 16-bit mono PCM, about real time
        await transcriber.end_stream()            # no more audio
        async for event in transcriber.events():  # until the server closes the socket
            print(event)

How the API works (https://dev.meta.ai/docs/speech-to-text):

    * The first message is a JSON handshake with the key and settings; wait for the
      ``{"sessionId": ...}`` reply before sending audio.
    * Audio goes as binary frames of signed 16-bit little-endian mono PCM.
    * ``{"type": "endStream"}`` ends the whole session. The server then sends the
      remaining events and closes with code 1000.
    * Modes: PUSH_TO_TALK (one transcript per session) and ENDPOINTING (the server
      splits speech into turns, so one session can hold several questions).
"""

import asyncio
import json
from collections.abc import AsyncIterator

from websockets.asyncio.client import ClientConnection, connect

REALTIME_URL = "wss://api.meta.ai/v1/asr/realtime"
MODEL = "muse-voice-transcribe-1.0"
HANDSHAKE_TIMEOUT_SECONDS = 10.0  # the server closes sessions that do not start within 10 s

# Sample rate (Hz) -> the name the API uses for it.
AUDIO_ENCODINGS = {16000: "PCM_16KHZ", 24000: "PCM_24KHZ"}
MODES = ("PUSH_TO_TALK", "ENDPOINTING")


class RealtimeTranscriber:
    """One realtime transcription session with Meta's speech-to-text API."""

    def __init__(
        self,
        api_key: str,
        mode: str = "ENDPOINTING",
        sample_rate: int = 24000,
        language_bias: list[str] | None = None,
        keywords: list[str] | None = None,
        url: str = REALTIME_URL,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if sample_rate not in AUDIO_ENCODINGS:
            raise ValueError(f"sample_rate must be one of {sorted(AUDIO_ENCODINGS)}, got {sample_rate}")

        self._api_key = api_key
        self._url = url
        self._settings = {
            "audioEncoding": AUDIO_ENCODINGS[sample_rate],
            "model": MODEL,
            "mode": mode,
            "partialMode": "CUMULATIVE",  # each partial replaces the previous one
            "emitAudioProgress": False,   # progress events are not needed
        }
        if language_bias:
            self._settings["languageBias"] = language_bias
        if keywords:
            self._settings["keywords"] = keywords

        self.session_id: str | None = None
        self._socket: ClientConnection | None = None

    async def __aenter__(self) -> "RealtimeTranscriber":
        """Open the socket, send the handshake and wait for the session id."""
        self._socket = await connect(self._url, open_timeout=HANDSHAKE_TIMEOUT_SECONDS)
        try:
            handshake = {"authorization": {"accessToken": f"Bearer {self._api_key}"}, **self._settings}
            await self._socket.send(json.dumps(handshake))

            async with asyncio.timeout(HANDSHAKE_TIMEOUT_SECONDS):
                reply = json.loads(await self._socket.recv())
            if "sessionId" not in reply:
                raise RuntimeError(f"Speech-to-text handshake failed: {reply.get('message', reply)}")
            self.session_id = reply["sessionId"]
        except BaseException:
            await self._socket.close()
            raise
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._socket is not None:
            await self._socket.close()

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Send raw PCM (16-bit little-endian, mono, at the sample rate chosen above)."""
        await self._socket.send(pcm_bytes)

    async def end_stream(self) -> None:
        """Tell the server no more audio is coming. It ends the whole session."""
        await self._socket.send(json.dumps({"type": "endStream"}))

    async def events(self) -> AsyncIterator[dict]:
        """Yield the server's JSON events until it closes the socket."""
        async for message in self._socket:
            if isinstance(message, str):
                yield json.loads(message)


def to_client_event(event: dict) -> dict | None:
    """Turn a Meta event into the small message the browser receives, or None to skip it.

    Browser messages:
        {"type": "partial", "text": ...}                 text so far; replaces the last partial
        {"type": "final", "text": ..., "turnId": ...}    one finished question
        {"type": "error", "message": ...}
    """
    kind = event.get("type")

    if kind == "transcript":
        if event.get("final"):  # push-to-talk: the one transcript is complete
            return {"type": "final", "text": event["transcript"]}
        return {"type": "partial", "text": event["transcript"]}

    if kind == "speechComplete":  # endpointing: the finished transcript of one turn
        return {"type": "final", "text": event["transcript"], "turnId": event.get("turnId")}

    if kind == "error":
        return {"type": "error", "message": event.get("message", "Speech-to-text error")}

    return None  # speechStart, speechEnd, audioProgress, ... are not needed by the browser
