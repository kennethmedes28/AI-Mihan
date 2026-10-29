"""Monocular depth estimation with a transformer: how far from the camera is each part of a picture, in metres.

Uses Depth Anything V2 (metric, outdoor) through Hugging Face ``transformers``. It looks at ONE picture
and guesses the distance of every pixel. The guess is good for relative distances and rough for
absolute ones (it is often off by tens of percent, and worse for things very close to the camera),
so ``flood_depth.py`` treats it as an estimate with a wide margin.

    estimator = get_depth_estimator()
    metres = estimator.estimate(jpeg_bytes)      # numpy array (height, width), distance in metres

The model (about 100 MB) is downloaded on first use and then cached by Hugging Face. It runs on the
GPU when there is one, otherwise on the CPU (a few seconds per picture). Choose another model with the
``DEPTH_MODEL`` environment variable; it must be a METRIC depth model, or the numbers are not metres.
"""

import io
import os
import threading

import numpy as np

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class DepthEstimator:
    """Loads the depth model once (on first use) and turns pictures into distance maps."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("DEPTH_MODEL", DEFAULT_MODEL)
        self._pipeline = None
        self._lock = threading.Lock()  # loading and running the model is done one request at a time

    def _load(self):
        # Imported here so the rest of the app works (and starts fast) when depth is never asked for.
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        return pipeline("depth-estimation", model=self.model_name, device=device)

    def estimate(self, image_bytes: bytes) -> np.ndarray:
        """Return the distance in metres of every pixel, as an array shaped (height, width)."""
        import torch
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with self._lock:
            if self._pipeline is None:
                self._pipeline = self._load()
            predicted = self._pipeline(image)["predicted_depth"]

        depth = predicted.float().reshape(1, 1, *predicted.shape[-2:])
        if tuple(depth.shape[-2:]) != (image.height, image.width):  # the model may answer at its own size
            depth = torch.nn.functional.interpolate(depth, size=(image.height, image.width), mode="bilinear", align_corners=False)
        return depth[0, 0].cpu().numpy()


_estimator: DepthEstimator | None = None
_estimator_lock = threading.Lock()


def get_depth_estimator() -> DepthEstimator:
    """The shared estimator, so the model is loaded only once per process."""
    global _estimator
    with _estimator_lock:
        if _estimator is None:
            _estimator = DepthEstimator()
        return _estimator
