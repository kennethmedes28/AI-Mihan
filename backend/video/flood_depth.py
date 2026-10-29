"""Estimate how deep flood water is from a photo, using people as a ruler. A ROUGH estimate, never a measurement.

Idea: an adult is about 1.7 m tall. If a person stands in the water and we can see only the top part, the part
under water is the missing height:

    camera distance  Z      from the depth model (video/depth_model.py), in metres
    visible height          pixels_tall * Z / focal_length_in_pixels        (pinhole camera)
    water depth             1.7 m - visible height

The camera's focal length is read from the photo's EXIF data when it has some; otherwise a typical phone
camera is assumed (26 mm in 35 mm terms). Only people who are clearly visible, fully inside the picture and
standing in the water are used, and the middle of all of them is reported.

How far to trust it: every step has an error. Depth models are off by tens of percent in distance, the focal
length may be wrong, and people differ in height (about 1.5 to 1.8 m for adults) or crouch, wade at an angle or
are partly hidden. The result therefore carries a wide range, and its confidence is "low" unless the photo states its camera
zoom (EXIF) and at least five people agree closely, which gives "medium". It was only compared with one
photo by eye. It says nothing about how safe a road is.

Bands (centimetres): none under 5, ankle 5 to 15, knee 15 to 50, waist 50 to 100, chest or deeper from 100.
"""

import io
from dataclasses import asdict, dataclass, field

import numpy as np

from video.flood_scene import IN_WATER_OVERLAP, Detection, grid_shape, paint_water, share_in_water

ADULT_HEIGHT_M = 1.7
ADULT_HEIGHT_RANGE_M = (1.5, 1.8)
DEFAULT_FOCAL_MM_35 = 26.0  # 35 mm-equivalent focal length of a typical phone main camera
FULL_FRAME_LONG_SIDE_MM = 36.0

MIN_PERSON_HEIGHT_PX = 20  # smaller people are too coarse to measure
MIN_HEIGHT_TO_WIDTH = 1.2  # box shape of a standing person; wider boxes are crouched, hidden or groups
EDGE_MARGIN_PX = 3  # a person touching the picture's edge is cut off
DISTANCE_RANGE_M = (1.0, 60.0)  # distances outside this are not trusted
DISTANCE_ERROR = 0.25  # assumed relative error of the depth model, used for the range
MEDIUM_CONFIDENCE_MIN_PEOPLE = 5
MEDIUM_CONFIDENCE_MAX_SPREAD_M = 0.15  # people must agree within this (interquartile half-range)

BAND_NONE = "none"
BAND_ANKLE = "ankle"
BAND_KNEE = "knee"
BAND_WAIST = "waist"
BAND_CHEST = "chest_or_deeper"


@dataclass
class DepthEstimate:
    """A rough water depth with its range and how it was made."""

    depth_cm: float  # middle estimate
    low_cm: float  # plausible range, after allowing for body height and depth-model error
    high_cm: float
    band: str
    confidence: str  # "low" or "medium"; never "high"
    people_used: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def depth_band(depth_cm: float) -> str:
    """Name the depth in body-part terms."""
    if depth_cm < 5:
        return BAND_NONE
    if depth_cm < 15:
        return BAND_ANKLE
    if depth_cm < 50:
        return BAND_KNEE
    if depth_cm < 100:
        return BAND_WAIST
    return BAND_CHEST


def exif_focal_mm_35(image_bytes: bytes) -> float | None:
    """The 35 mm-equivalent focal length stored in the photo, if the camera wrote one."""
    from PIL import Image

    try:
        exif = Image.open(io.BytesIO(image_bytes)).getexif()
        value = exif.get_ifd(0x8769).get(0xA405)  # FocalLengthIn35mmFilm
    except Exception:  # a damaged or missing EXIF block is normal; fall back to the default
        return None
    return float(value) if value else None


def focal_length_pixels(frame_size: tuple[int, int], focal_mm_35: float) -> float:
    """Focal length in pixels: the long side of the picture stands for 36 mm of a full-frame sensor."""
    return max(frame_size) * focal_mm_35 / FULL_FRAME_LONG_SIDE_MM


def usable_people(
    people: list[Detection], water_grid: np.ndarray, depth_map: np.ndarray, frame_size: tuple[int, int]
) -> list[tuple[float, float]]:
    """For each clearly visible person standing in water: (pixel height, distance in metres)."""
    width, height = frame_size
    usable = []
    for person in people:
        left, top, right, bottom = person.box
        box_height, box_width = bottom - top, right - left
        if box_height < MIN_PERSON_HEIGHT_PX or box_height / max(box_width, 1) < MIN_HEIGHT_TO_WIDTH:
            continue
        if left < EDGE_MARGIN_PX or top < EDGE_MARGIN_PX or right > width - EDGE_MARGIN_PX or bottom > height - EDGE_MARGIN_PX:
            continue
        if share_in_water(water_grid, person.box, frame_size) < IN_WATER_OVERLAP:
            continue

        # Distance of the person: the middle of the lower fifth of the box, where body meets water.
        feet = depth_map[int(bottom - 0.2 * box_height):int(bottom), int(left):int(right) + 1]
        distance = float(np.median(feet)) if feet.size else float("nan")
        if not DISTANCE_RANGE_M[0] <= distance <= DISTANCE_RANGE_M[1]:
            continue
        usable.append((box_height, distance))
    return usable


def estimate_flood_depth(
    people: list[Detection],
    water: list[Detection],
    depth_map: np.ndarray,
    frame_size: tuple[int, int],
    focal_mm_35: float | None = None,
) -> DepthEstimate | None:
    """Estimate the water depth from the people standing in it, or None when no person can be used."""
    notes = []
    zoom_known = focal_mm_35 is not None
    if focal_mm_35 is None:
        focal_mm_35 = DEFAULT_FOCAL_MM_35
        notes.append(f"Camera zoom unknown, assumed a typical phone camera ({DEFAULT_FOCAL_MM_35:g} mm).")
    focal_pixels = focal_length_pixels(frame_size, focal_mm_35)

    water_grid = np.zeros(grid_shape(frame_size), dtype=bool)
    for detection in water:
        paint_water(water_grid, detection, frame_size)

    samples = usable_people(people, water_grid, depth_map, frame_size)
    if not samples:
        return None

    visible_m = np.array([pixels * distance / focal_pixels for pixels, distance in samples])
    depths_m = np.clip(ADULT_HEIGHT_M - visible_m, 0.0, ADULT_HEIGHT_M)

    middle_visible = float(np.median(visible_m))
    low_m = max(0.0, ADULT_HEIGHT_RANGE_M[0] - middle_visible * (1 + DISTANCE_ERROR))
    high_m = min(ADULT_HEIGHT_RANGE_M[1], ADULT_HEIGHT_RANGE_M[1] - middle_visible * (1 - DISTANCE_ERROR))
    depth_m = float(np.median(depths_m))

    spread_m = float((np.percentile(depths_m, 75) - np.percentile(depths_m, 25)) / 2)
    confident = zoom_known and len(samples) >= MEDIUM_CONFIDENCE_MIN_PEOPLE and spread_m <= MEDIUM_CONFIDENCE_MAX_SPREAD_M
    notes.append(f"Based on {len(samples)} people standing in the water, assuming an adult is {ADULT_HEIGHT_M} m tall.")

    return DepthEstimate(
        depth_cm=round(depth_m * 100),
        low_cm=round(low_m * 100),
        high_cm=round(max(high_m, depth_m) * 100),
        band=depth_band(depth_m * 100),
        confidence="medium" if confident else "low",
        people_used=len(samples),
        notes=notes,
    )
