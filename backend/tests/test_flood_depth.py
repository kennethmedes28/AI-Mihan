"""Tests for the water-depth estimate, using made-up depth maps and detections (no models, no network)."""

import numpy as np

from agents.flood_media_agent import format_report
from agents.flood_types import ChatContext
from video import depth_model
from video.depth_model import DepthEstimator
from video.flood_depth import (
    BAND_ANKLE,
    BAND_CHEST,
    BAND_KNEE,
    BAND_NONE,
    BAND_WAIST,
    depth_band,
    estimate_flood_depth,
    focal_length_pixels,
)
from video.flood_scene import Detection, build_report

FRAME = (960, 720)
F_MM = 26.0
F_PIX = focal_length_pixels(FRAME, F_MM)  # 693 px


def water_everywhere() -> Detection:
    rows, columns = 720, 960
    return Detection("w", 0, (0, 300, 960, 720), 0.9, bytes([1]) * ((720 - 300) * columns), (720 - 300, columns))


def person_with_visible_height(visible_m: float, distance_m: float, left: int = 300, bottom: int = 600) -> Detection:
    """A person whose visible part is ``visible_m`` tall at ``distance_m`` (pinhole camera)."""
    pixels = visible_m * F_PIX / distance_m
    return Detection("p", 0, (left, bottom - pixels, left + pixels / 2.5, bottom), 0.9)


def flat_depth(distance_m: float) -> np.ndarray:
    return np.full((FRAME[1], FRAME[0]), distance_m, dtype=np.float32)


def test_bands():
    assert [depth_band(cm) for cm in (2, 10, 30, 70, 120)] == [BAND_NONE, BAND_ANKLE, BAND_KNEE, BAND_WAIST, BAND_CHEST]


def test_people_waist_deep_give_about_the_missing_height():
    # People of whom only 1.1 m shows: about 0.6 m is under water.
    people = [person_with_visible_height(1.1, 20, left=100 + 80 * i) for i in range(6)]
    estimate = estimate_flood_depth(people, [water_everywhere()], flat_depth(20), FRAME, F_MM)
    assert abs(estimate.depth_cm - 60) <= 2
    assert estimate.band == BAND_WAIST and estimate.people_used == 6
    assert estimate.low_cm < estimate.depth_cm < estimate.high_cm
    assert estimate.confidence == "medium"  # known zoom, six people who agree


def test_unknown_zoom_is_flagged_and_never_more_than_low_confidence():
    people = [person_with_visible_height(1.1, 20, left=100 + 80 * i) for i in range(6)]
    estimate = estimate_flood_depth(people, [water_everywhere()], flat_depth(20), FRAME, None)
    assert estimate.confidence == "low"
    assert any("zoom unknown" in note for note in estimate.notes)


def test_fully_visible_people_mean_no_depth():
    people = [person_with_visible_height(1.7, 20)]
    estimate = estimate_flood_depth(people, [water_everywhere()], flat_depth(20), FRAME, F_MM)
    assert estimate.depth_cm == 0 and estimate.band == BAND_NONE


def test_unusable_people_are_skipped():
    on_dry_ground = person_with_visible_height(1.1, 20, bottom=200)  # feet above the water
    cut_off = Detection("c", 0, (0, 300, 40, 600), 0.9)  # touches the left edge
    tiny = Detection("t", 0, (400, 590, 405, 600), 0.9)  # 10 pixels tall
    too_far = person_with_visible_height(1.1, 90, left=500)
    assert estimate_flood_depth([on_dry_ground, cut_off, tiny], [water_everywhere()], flat_depth(20), FRAME, F_MM) is None
    assert estimate_flood_depth([too_far], [water_everywhere()], flat_depth(90), FRAME, F_MM) is None


def test_report_text_gives_the_range_and_calls_it_a_guess():
    people = [person_with_visible_height(1.1, 20, left=100 + 80 * i) for i in range(6)]
    estimate = estimate_flood_depth(people, [water_everywhere()], flat_depth(20), FRAME, F_MM)
    report = build_report("image", FRAME, [water_everywhere()], people, [])
    report.depth_cm, report.depth_low_cm, report.depth_high_cm = estimate.depth_cm, estimate.low_cm, estimate.high_cm
    report.depth_band, report.depth_confidence, report.depth_people_used = estimate.band, estimate.confidence, estimate.people_used
    text = format_report(report, ChatContext())
    assert "Rough depth estimate" in text and "waist-deep" in text and "not a measurement" in text
    assert "Tantiyang lalim" in text and "hula lang" in text.lower()


def test_depth_estimator_resizes_to_the_picture_and_loads_once(monkeypatch):
    import io

    from PIL import Image

    calls = {"loaded": 0}

    class FakePipeline:
        def __call__(self, image):
            import torch

            return {"predicted_depth": torch.full((1, 40, 60), 12.0)}  # smaller than the picture

    def fake_load(self):
        calls["loaded"] += 1
        return FakePipeline()

    monkeypatch.setattr(DepthEstimator, "_load", fake_load)
    buffer = io.BytesIO()
    Image.new("RGB", (120, 80)).save(buffer, format="JPEG")

    estimator = DepthEstimator("fake-model")
    first = estimator.estimate(buffer.getvalue())
    estimator.estimate(buffer.getvalue())
    assert first.shape == (80, 120) and float(first.mean()) == 12.0
    assert calls["loaded"] == 1
    assert depth_model.DEFAULT_MODEL.endswith("Metric-Outdoor-Small-hf")
