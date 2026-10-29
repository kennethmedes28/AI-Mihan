# Flood Media Check

How a photo or video of a street is checked for flood water. Status: **detection and agent built; not connected to the chat graph yet**.

## 1. What it does

SAM 3.1 (Meta's Segment Anything model, `sam-3.1`) finds every object that matches a short phrase in a picture and returns a box
and a pixel mask for each. The flood media agent asks for three things and turns the answers into a short summary:

| Phrase | Used for |
|---|---|
| `flood water` | Share of the picture covered by water, and the share of video frames that show it |
| `person` | How many people are visible, and how many stand in the water |
| `vehicle` | How many vehicles are visible, and how many are in the water |

SAM takes one phrase per request, so this is three requests (sent in parallel). A person or vehicle counts as "in the water" when at least half
of the ground around its base is water; the base is checked, not the body, because SAM leaves the object out of the water mask.

```mermaid
flowchart LR
    M[Photo or video] --> S["SamClient<br/>video/sam_client.py"]
    S -->|"flood water / person / vehicle"| A["SAM 3.1<br/>api.meta.ai/v1/responses"]
    A -->|boxes + masks| P[meta_sam_parser]
    P --> F["flood_scene.py<br/>coverage, counts, level"]
    F --> G["flood_media_agent.py<br/>fixed bilingual text"]
```

## 2. Code

| File | Role |
|---|---|
| `backend/video/sam_client.py` | `SamClient`: sends a phrase plus an image (URL, file or bytes) or a video (URL or file, uploaded once) to `POST /v1/responses` and parses the streamed answer with `meta_sam_parser` |
| `backend/video/flood_scene.py` | Pairs boxes with masks, draws water on a coarse grid, and builds a `FloodSceneReport`. `analyze_flood_media()` is the blocking entry point |
| `backend/agents/flood_media_agent.py` | `answer_flood_media_question(media, context)` returns an `AgentReply` with English and Filipino text. No chat model is used |
| `backend/tests/test_flood_media.py` | Tests with made-up detections (no SAM calls) |

`FloodSceneReport` fields: `media_kind`, `frames_checked`, `water_coverage` (typical share under water, 0 to 1), `peak_water_coverage`,
`frames_with_water`, `level` (`none`, `some`, `widespread`), `people`, `people_in_water`, `vehicles`, `vehicles_in_water`, `notes`.
Levels use the typical coverage: under 3% is `none`, 3% to 20% is `some`, 20% and above is `widespread`. These are starting values, to be tuned on real photos.
Matches with a confidence under 0.5 are ignored.

## 3. SAM API facts used (reviewed 2026-10-29)

- `POST https://api.meta.ai/v1/responses`, model `sam-3.1`, same `MUSE_API` key as the chat models, streamed. The answer is special-token text that `meta_sam_parser` (`pip install meta-sam-parser`) turns into typed records.
- Images: public URL or base64 data URL. Videos: MP4 or MOV with H.264, H.265 or MPEG-4 (WebM and VP9 are rejected), up to about 10 minutes, uploaded with `POST /v1/files` (`purpose=user_data`) or given by URL.
- Video is billed per frame with a 50-frame minimum per request, and each of the three phrases is a separate request, so keep clips short (a few seconds). Pricing and rate limits are on Meta's pricing page, not reviewed here.
- A request with no match is valid and returns nothing.

## 4. Tested so far

- A photo of a flooded street in Quezon City (CC licence, Wikimedia Commons, 960x720): water found at 33% coverage, 17 people, 10 of them in the water, no vehicles. This matched the picture by eye.
- A 4-second clip (40 frames) of the 2009 Ondoy flood in Quezon City, converted to H.264 MP4: water in all 40 frames at about 40% coverage.
- Not tested: night or low-light pictures, muddy or very shallow water, vehicles (none in the test photo), long videos, accuracy across many photos.

## 5. Webcam check

SAM works on pictures and clips, not on a live stream, so the webcam check is near-live: the browser sends one frame every few seconds.

- **Browser:** `backend/video/static/camera.html` asks for the webcam (`getUserMedia`), shrinks each frame to 640 pixels wide, sends it as a JPEG over a WebSocket and waits for the answer before sending the next one. It shows the level, the water coverage and the bilingual text. Tick "also count people and vehicles" to add them (3 SAM requests per frame instead of 1).
- **Server:** `backend/video/camera_router.py` is the `/ws/camera` endpoint (message list in its docstring). At most one check every 3 seconds, frames over about 1 MB are refused, a failed frame is reported and the session goes on, and a connection closes after 5 minutes.
- **Try it:** from `backend/`, run `uv run uvicorn video.demo_app:app --port 8000` and open `http://localhost:8000`. A webcam needs `localhost` or https. Inside WSL the browser runs on Windows and reaches the server through localhost.
- **Tested:** the WebSocket with the real SAM API (a frame in, a "widespread" report out) and with fake analysers in `tests/test_camera_websocket.py`. The browser page and a real webcam have **not** been tried.
- **Cost:** the pause is measured from the start of the previous check, so a check that takes longer than 3 seconds is followed straight away by the next frame. Left running, this is one SAM request (or three) every few seconds; use it for a minute or two, not all day.

## 6. Rough water depth (optional)

SAM cannot measure depth: it outlines things and has no sense of distance or real size. A depth estimate needs two more pieces, so it is optional (`estimate_depth=True`, photos only):

1. **A depth model** (`video/depth_model.py`): Depth Anything V2 Metric Outdoor (small), a transformer from Hugging Face, guesses the distance in metres of every pixel of ONE picture. About 100 MB, downloaded on first use; runs on the GPU if there is one, otherwise the CPU. Choose another metric model with `DEPTH_MODEL`.
2. **People as a ruler** (`video/flood_depth.py`): an adult is about 1.7 m tall. For each clearly visible person standing in the water, `visible height = pixel height x distance / focal length in pixels`, and `depth = 1.7 m - visible height`. The report uses the middle value over all such people.

The focal length comes from the photo's EXIF data when present, otherwise a typical phone camera (26 mm in 35 mm terms) is assumed and a note says so.

| Report field | Meaning |
|---|---|
| `depth_cm`, `depth_low_cm`, `depth_high_cm` | Middle estimate and a plausible range (allows for body heights of 1.5 to 1.8 m and a 25% error in the model's distances) |
| `depth_band` | `none` (under 5 cm), `ankle` (5 to 15), `knee` (15 to 50), `waist` (50 to 100), `chest_or_deeper` |
| `depth_confidence` | `low`, or `medium` only when the photo states its zoom and at least five people agree closely. Never `high` |
| `depth_people_used` | How many people the estimate is based on |

People are used only if they are at least 20 pixels tall, standing-shaped, not cut off by the picture's edge, in the water, and between 1 and 60 m from the camera. With no such person the report has no depth and a note says why.

**How far to trust it.** On the Quezon City test photo it gave about 52 cm (range 3 to 92 cm, "waist", low confidence) from 7 people; the Muse Spark vision model independently said "waist, 50 to 100 cm, low confidence" for the same photo. That agreement is one picture judged by eye. There are no photos with measured depths to test against, so the accuracy is unknown. The depth model's distances are rough (for a person about 1.5 m away it said 8 m, so close-up people are unreliable) and the range is wide on purpose. Do not use the number to decide whether a road can be crossed. Needs only that one extra model pass: about 1.5 s on the GPU after a 20 s first load.

Turning it on: `analyze_flood_media(media, api_key, estimate_depth=True)`, `answer_flood_media_question(..., estimate_depth=True)`, or `?depth=1` on `/ws/camera` (the demo page has a checkbox).

## 7. Safety rules

- The result is unverified evidence. It never overrides PAGASA or NDRRMC information, and it never says a road is safe. A depth, when asked for, is shown as a guess with its range.
- Every reply says it is an automatic check, that the picture may be old, that any depth is only a guess, and to call 911 if someone is in danger.
- A place given by the user is repeated as "not confirmed from the picture". The picture does not set the user's location.

## 8. Not built yet

- Sending a photo or video through the chat: the graph state carries text only, so the agent has no entry point from a chat message yet (an upload endpoint and a `media` intent are needed).
- Saving a result as a community report (`reports` table) with the picture's evidence.
- Using the water coverage as an input to the flood risk agent or to the flooded areas that routes avoid.
- A check of what the user uploads (size, type, content) and a per-user limit, since every check costs SAM requests.
- Tuning the thresholds and measuring accuracy on a labelled set of Philippine flood photos (`docs/evaluation_rubric.md`).
- Depth accuracy: photos with a measured depth (a gauge or marked pole in view) to compare against; using the dry people in a scene to calibrate the scale; vehicles and fixed objects as extra rulers; depth for videos (only photos today).
