"""
The ONE place that turns a face crop into a model input.

Both the dataset builder (prepare_dataset.py) and inference (predict.py,
app.py) call `preprocess_face`. Keeping a single implementation rules out
train/serve skew - a different resize interpolation or colour order between
training and the web app is the most common reason a model with 95% validation
accuracy falls apart on real uploads.

Contract:
    input : BGR uint8 crop of any size (what OpenCV gives you)
    output: RGB float32 array of shape (IMG_SIZE, IMG_SIZE, 3) in the 0-255 range

Why 0-255 and not 0-1? Because any channel normalisation the backbone needs is
built *into the model* (see train.py / build_model). That way a saved .keras
file carries its own normalisation and cannot be fed the wrong range.
"""

from __future__ import annotations

import cv2
import numpy as np

import config


def resize_square(image_bgr: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """Resize to size x size. INTER_AREA when shrinking, INTER_CUBIC when enlarging."""
    h, w = image_bgr.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC
    return cv2.resize(image_bgr, (size, size), interpolation=interp)


def preprocess_face(crop_bgr: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """BGR uint8 crop -> RGB float32 (size, size, 3), values 0-255."""
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Empty face crop passed to preprocess_face")
    resized = resize_square(crop_bgr, size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)


def preprocess_batch(crops_bgr: list[np.ndarray], size: int = config.IMG_SIZE) -> np.ndarray:
    """List of BGR crops -> (N, size, size, 3) float32 batch."""
    return np.stack([preprocess_face(c, size) for c in crops_bgr], axis=0)


def to_display_bgr(model_input_rgb: np.ndarray) -> np.ndarray:
    """Inverse of preprocess_face for drawing: RGB float 0-255 -> BGR uint8."""
    rgb = np.clip(model_input_rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
