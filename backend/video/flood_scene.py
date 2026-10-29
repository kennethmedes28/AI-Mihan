"""Turn SAM 3.1 detections into a short flood summary of a photo or video.

Three things are looked for, one SAM request each (SAM takes one phrase per request):

    * "flood water"  - how much of the picture is covered, and in how many frames
    * "person"       - how many people, and how many stand in the water
    * "vehicle"      - how many vehicles, and how many sit in the water

The result is a ``FloodSceneReport``. It is only what the camera shows: it cannot tell how DEEP the
water is or whether a road is safe, and a photo can be old or from somewhere else. Treat it as
unverified evidence, never as an official warning.

    report = analyze_flood_media(ImageInput(path="street.jpg"), api_key)
    report.level            # "none", "some" or "widespread"
    report.water_coverage   # share of the picture covered by water, 0 to 1
"""

import asyncio
import re
import threading
from dataclasses import asdict, dataclass, field

import numpy as np
from meta_sam_parser import decode_mask_to_raster

from video.sam_client import ImageInput, SamClient, VideoInput

WATER_CONCEPT = "flood water"
PEOPLE_CONCEPT = "person"
VEHICLE_CONCEPT = "vehicle"

MIN_CONFIDENCE = 0.5  # matches SAM is less sure about are ignored

# Share of the picture covered by water (starting values, to be tuned on real photos).
SOME_WATER_FROM = 0.03
WIDESPREAD_WATER_FROM = 0.20

# An object counts as "in the water" when this much of the ground around its base is water.
# SAM leaves the object itself out of the water mask, so the area at its feet is checked, not its body.
IN_WATER_OVERLAP = 0.5
FEET_ABOVE_BASE = 0.1  # the checked strip starts this far (share of the box height) above the bottom edge...
FEET_BELOW_BASE = 0.3  # ...and ends this far below it
FEET_SIDE_MARGIN = 0.25  # ...and reaches this far (share of the box width) past each side

GRID_COLUMNS = 192  # water is drawn on a coarse grid (a few pixels per cell); fine enough for shares and overlaps

LEVEL_NONE = "none"
LEVEL_SOME = "some"
LEVEL_WIDESPREAD = "widespread"


@dataclass
class Detection:
    """One object SAM found in one frame (a photo is frame 0)."""

    object_id: str
    frame: int
    box: tuple[float, float, float, float]  # left, top, right, bottom in source pixels
    confidence: float | None
    raster: bytes | None = None  # mask pixels, row by row, 0 or 1; covers the box
    mask_size: tuple[int, int] | None = None  # (rows, columns) of ``raster``


@dataclass
class FloodSceneReport:
    """What the camera shows, in numbers."""

    media_kind: str  # "image" or "video"
    frames_checked: int  # 1 for a photo
    water_coverage: float  # typical share of the picture under water, 0 to 1 (median over frames with water)
    peak_water_coverage: float  # the highest share in any frame
    frames_with_water: float  # share of frames that show flood water (1.0 or 0.0 for a photo)
    level: str  # none / some / widespread
    people: int  # most people seen in one frame
    people_in_water: int
    vehicles: int
    vehicles_in_water: int
    notes: list[str] = field(default_factory=list)  # things a reader should know about this result
    # Rough water depth, only when asked for (photos only; see flood_depth.py). All None when not estimated.
    depth_cm: float | None = None
    depth_low_cm: float | None = None
    depth_high_cm: float | None = None
    depth_band: str | None = None
    depth_confidence: str | None = None
    depth_people_used: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# --- Reading SAM's answer ----------------------------------------------------


def frame_size_from_output(raw_output: str) -> tuple[int, int] | None:
    """The (width, height) of the source frame, repeated in every box token as ``w=..;h=..``."""
    match = re.search(r"w=(\d+);h=(\d+)", raw_output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def to_detections(result) -> list[Detection]:
    """Pair each box with its mask (same object and frame) and drop weak matches."""
    masks = {}
    for record in result.records:
        if record.kind == "mask":
            masks[(record.object_id, _frame_index(record))] = record

    detections = []
    for record in result.records:
        if record.kind != "box":
            continue
        if record.confidence is not None and record.confidence < MIN_CONFIDENCE:
            continue
        frame = _frame_index(record)
        detection = Detection(
            object_id=record.object_id,
            frame=frame,
            box=(record.left, record.top, record.right, record.bottom),
            confidence=record.confidence,
        )
        mask_record = masks.get((record.object_id, frame))
        if mask_record is not None:
            detection.raster = decode_mask_to_raster(mask_record.mask)
            detection.mask_size = (mask_record.mask.height, mask_record.mask.width)
        detections.append(detection)
    return detections


def _frame_index(record) -> int:
    return record.frame.frame_index if record.frame is not None else 0


# --- Measuring ---------------------------------------------------------------


def grid_shape(frame_size: tuple[int, int]) -> tuple[int, int]:
    """(rows, columns) of the coarse grid for a frame of (width, height) pixels."""
    width, height = frame_size
    return max(1, round(GRID_COLUMNS * height / width)), GRID_COLUMNS


def paint_water(grid: np.ndarray, detection: Detection, frame_size: tuple[int, int]) -> None:
    """Mark the grid cells covered by one water mask (nearest-pixel sampling)."""
    if detection.raster is None or detection.mask_size is None:
        return
    rows, columns = grid.shape
    width, height = frame_size
    left, top, right, bottom = detection.box
    mask_rows, mask_columns = detection.mask_size
    mask = np.frombuffer(detection.raster, dtype=np.uint8).reshape(mask_rows, mask_columns)

    # Frame position of every grid cell centre, then the mask pixel under it.
    x = (np.arange(columns) + 0.5) * width / columns
    y = (np.arange(rows) + 0.5) * height / rows
    inside_x = (x >= left) & (x < right)
    inside_y = (y >= top) & (y < bottom)
    if not inside_x.any() or not inside_y.any():
        return
    mask_x = np.clip(((x[inside_x] - left) / (right - left) * mask_columns).astype(int), 0, mask_columns - 1)
    mask_y = np.clip(((y[inside_y] - top) / (bottom - top) * mask_rows).astype(int), 0, mask_rows - 1)
    covered = mask[np.ix_(mask_y, mask_x)] == 1
    grid[np.ix_(inside_y, inside_x)] |= covered


def share_in_water(grid: np.ndarray, box: tuple[float, float, float, float], frame_size: tuple[int, int]) -> float:
    """Share of the ground around an object's base (feet, wheels) that lies on water."""
    rows, columns = grid.shape
    width, height = frame_size
    left, top, right, bottom = box
    box_width, box_height = right - left, bottom - top

    x_start, x_end = left - FEET_SIDE_MARGIN * box_width, right + FEET_SIDE_MARGIN * box_width
    y_start, y_end = bottom - FEET_ABOVE_BASE * box_height, bottom + FEET_BELOW_BASE * box_height
    x0, x1 = int(x_start / width * columns), int(np.ceil(x_end / width * columns))
    y0, y1 = int(y_start / height * rows), int(np.ceil(y_end / height * rows))

    region = grid[max(y0, 0):min(max(y1, y0 + 1), rows), max(x0, 0):min(max(x1, x0 + 1), columns)]
    return float(region.mean()) if region.size else 0.0


def build_report(
    media_kind: str,
    frame_size: tuple[int, int] | None,
    water: list[Detection],
    people: list[Detection],
    vehicles: list[Detection],
) -> FloodSceneReport:
    """Combine the detections of the three concepts into one report."""
    notes = ["Automatic check of the picture only: road safety cannot be judged from it, and any water depth is a rough guess."]
    frames = sorted({d.frame for d in water + people + vehicles})
    if frame_size is None:  # SAM found nothing at all, so there is nothing to measure
        return FloodSceneReport(media_kind, 1, 0.0, 0.0, 0.0, LEVEL_NONE, 0, 0, 0, 0,
                                notes + ["No flood water, people or vehicles were found."])

    frames_checked = max(frames) + 1 if media_kind == "video" else 1
    coverage_by_frame, grids = {}, {}
    for frame in {d.frame for d in water}:
        grid = np.zeros(grid_shape(frame_size), dtype=bool)
        for detection in (d for d in water if d.frame == frame):
            paint_water(grid, detection, frame_size)
        grids[frame] = grid
        coverage_by_frame[frame] = float(grid.mean())

    wet_coverages = [value for value in coverage_by_frame.values() if value > 0]
    water_coverage = float(np.median(wet_coverages)) if wet_coverages else 0.0
    peak_coverage = max(wet_coverages, default=0.0)
    frames_with_water = len(wet_coverages) / frames_checked

    def counts(objects: list[Detection]) -> tuple[int, int]:
        """Most objects seen in one frame, and the most of those standing in water."""
        most, most_in_water = 0, 0
        for frame in {d.frame for d in objects}:
            here = [d for d in objects if d.frame == frame]
            grid = grids.get(frame)
            in_water = 0 if grid is None else sum(
                share_in_water(grid, d.box, frame_size) >= IN_WATER_OVERLAP for d in here
            )
            most, most_in_water = max(most, len(here)), max(most_in_water, in_water)
        return most, most_in_water

    people_count, people_in_water = counts(people)
    vehicle_count, vehicles_in_water = counts(vehicles)

    level = LEVEL_NONE
    if water_coverage >= WIDESPREAD_WATER_FROM:
        level = LEVEL_WIDESPREAD
    elif water_coverage >= SOME_WATER_FROM:
        level = LEVEL_SOME
    if media_kind == "video":
        notes.append(f"{frames_checked} frames were checked; water appeared in {frames_with_water:.0%} of them.")

    return FloodSceneReport(media_kind, frames_checked, water_coverage, peak_coverage, frames_with_water,
                            level, people_count, people_in_water, vehicle_count, vehicles_in_water, notes)


# --- Running it ----------------------------------------------------------------


async def analyze_flood_media_async(
    media: ImageInput | VideoInput,
    api_key: str,
    include_people_and_vehicles: bool = True,
    estimate_depth: bool = False,
) -> FloodSceneReport:
    """Ask SAM for water, people and vehicles (in parallel) and summarise what it found.

    With ``include_people_and_vehicles=False`` only the water is looked for: one SAM request
    instead of three, which is what a repeating webcam check uses to keep the cost down.

    ``estimate_depth=True`` (photos only) also runs the depth model and adds a rough water depth
    to the report. It needs people in the picture, so it always looks for them.
    """
    estimate_depth = estimate_depth and isinstance(media, ImageInput)
    include_people_and_vehicles = include_people_and_vehicles or estimate_depth
    media_kind = "video" if isinstance(media, VideoInput) else "image"
    async with SamClient(api_key) as sam:
        if isinstance(media, VideoInput) and media.path and not media.file_id:
            media = VideoInput(file_id=await sam.upload_video(media.path))  # upload once, reuse for each concept
        concepts = [WATER_CONCEPT] + ([PEOPLE_CONCEPT, VEHICLE_CONCEPT] if include_people_and_vehicles else [])
        results = await asyncio.gather(*(sam.segment(concept, media) for concept in concepts))

    sizes = [size for size in (frame_size_from_output(r.raw_output) for r in results) if size]
    detections = [to_detections(result) for result in results]
    water, people, vehicles = detections + [[]] * (3 - len(detections))
    report = build_report(media_kind, sizes[0] if sizes else None, water, people, vehicles)
    if estimate_depth and sizes:
        await add_depth_estimate(report, media, water, people, sizes[0])
    return report


async def add_depth_estimate(
    report: FloodSceneReport, media: ImageInput, water: list[Detection], people: list[Detection], frame_size: tuple[int, int]
) -> None:
    """Fill in the report's depth fields. A failure only adds a note; the rest of the report stays."""
    # Imported here: flood_depth needs this module's helpers, and the depth model is heavy to load.
    from video.depth_model import get_depth_estimator
    from video.flood_depth import estimate_flood_depth, exif_focal_mm_35

    if report.level == LEVEL_NONE:
        return
    try:
        image_bytes = await asyncio.to_thread(media.read_bytes)
        depth_map = await asyncio.to_thread(get_depth_estimator().estimate, image_bytes)  # blocking model work
        estimate = estimate_flood_depth(people, water, depth_map, frame_size, exif_focal_mm_35(image_bytes))
    except Exception as error:  # download, model or memory problems must not lose the detection result
        report.notes.append(f"Depth could not be estimated ({type(error).__name__}).")
        return

    if estimate is None:
        report.notes.append("Depth was not estimated: no person was clear enough standing in the water.")
        return
    report.depth_cm, report.depth_low_cm, report.depth_high_cm = estimate.depth_cm, estimate.low_cm, estimate.high_cm
    report.depth_band, report.depth_confidence, report.depth_people_used = estimate.band, estimate.confidence, estimate.people_used
    report.notes.extend(estimate.notes)


def analyze_flood_media(
    media: ImageInput | VideoInput, api_key: str, estimate_depth: bool = False
) -> FloodSceneReport:
    """Blocking version of ``analyze_flood_media_async`` for the (synchronous) agents."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(analyze_flood_media_async(media, api_key, estimate_depth=estimate_depth))

    # Already inside an event loop (for example an async server): run in a helper thread instead.
    outcome = {}

    def work() -> None:
        try:
            outcome["report"] = asyncio.run(analyze_flood_media_async(media, api_key, estimate_depth=estimate_depth))
        except BaseException as error:  # handed back to the caller below
            outcome["error"] = error

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["report"]
