"""Tests for the flood media check, using made-up detections (no SAM calls)."""

import numpy as np

from agents import flood_media_agent
from agents.flood_media_agent import answer_flood_media_question, format_report
from agents.flood_types import ChatContext, Location
from video.flood_scene import (
    LEVEL_NONE,
    LEVEL_SOME,
    LEVEL_WIDESPREAD,
    Detection,
    build_report,
    frame_size_from_output,
    share_in_water,
)
from video.sam_client import ImageInput

FRAME = (200, 100)  # width, height


def water_detection(left, top, right, bottom, frame=0) -> Detection:
    """A water mask that fills its whole box."""
    rows, columns = int(bottom - top), int(right - left)
    return Detection("0", frame, (left, top, right, bottom), 0.9, bytes([1]) * (rows * columns), (rows, columns))


def person(left, top, right, bottom, frame=0) -> Detection:
    return Detection("p", frame, (left, top, right, bottom), 0.9)


def test_frame_size_is_read_from_box_tokens():
    assert frame_size_from_output("<0f>0<|box;x1=1;y1=2;x2=3;y2=4;w=537;h=561|>") == (537, 561)
    assert frame_size_from_output("") is None


def test_coverage_and_levels():
    half = build_report("image", FRAME, [water_detection(0, 50, 200, 100)], [], [])
    assert abs(half.water_coverage - 0.5) < 0.03 and half.level == LEVEL_WIDESPREAD

    strip = build_report("image", FRAME, [water_detection(0, 94, 200, 100)], [], [])
    assert strip.level == LEVEL_SOME  # about 6% of the picture

    dry = build_report("image", FRAME, [], [person(10, 10, 20, 40)], [])
    assert dry.level == LEVEL_NONE and dry.people == 1 and dry.people_in_water == 0


def test_people_in_water_use_the_ground_at_their_feet():
    water = [water_detection(0, 60, 200, 100)]
    in_water = person(50, 40, 60, 75)  # feet at y=75, inside the water
    on_dry_ground = person(100, 5, 110, 40)  # feet at y=40, above the water
    report = build_report("image", FRAME, water, [in_water, on_dry_ground], [])
    assert report.people == 2 and report.people_in_water == 1


def test_no_detections_at_all():
    report = build_report("image", None, [], [], [])
    assert report.level == LEVEL_NONE and report.people == 0


def test_video_counts_frames_with_water():
    water = [water_detection(0, 50, 200, 100, frame=0), water_detection(0, 50, 200, 100, frame=1)]
    people = [person(10, 10, 20, 40, frame=3)]  # a later frame with no water
    report = build_report("video", FRAME, water, people, [])
    assert report.frames_checked == 4 and report.frames_with_water == 0.5


def test_share_in_water_is_zero_without_water():
    grid = np.zeros((50, 100), dtype=bool)
    assert share_in_water(grid, (10, 10, 20, 40), FRAME) == 0.0


def test_text_is_bilingual_and_never_says_a_road_is_safe():
    report = build_report("image", FRAME, [water_detection(0, 50, 200, 100)], [person(50, 40, 60, 75)], [])
    place = Location(name="Marikina", lat=14.65, lng=121.10, city="marikina")
    text = format_report(report, ChatContext(location=place))
    assert "Flood water covers" in text and "May baha" in text and "911" in text
    assert "Marikina" in text and "not confirmed" in text
    assert "safe to" not in text.lower()


def test_agent_without_key_or_on_failure(monkeypatch):
    monkeypatch.delenv("MUSE_API", raising=False)
    assert "cannot check pictures" in answer_flood_media_question(ImageInput(url="x"), ChatContext()).text

    def broken(media, api_key, estimate_depth=False):
        raise OSError("network down")

    monkeypatch.setattr(flood_media_agent, "analyze_flood_media", broken)
    reply = answer_flood_media_question(ImageInput(url="x"), ChatContext(), api_key="key")
    assert "could not check" in reply.text
