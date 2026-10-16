"""
Drawing / saving helpers. Nothing here touches the model input - that lives in
preprocessing/preprocessing.py so the model always sees identical tensors.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(path: str | Path, image_bgr: np.ndarray, quality: int = 92) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        cv2.imwrite(str(path), image_bgr)
    return path


def heatmap_to_color(heatmap: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """[0,1] float heatmap -> BGR JET colour image (optionally resized to (w, h))."""
    hm = np.clip(heatmap, 0.0, 1.0)
    if size is not None:
        hm = cv2.resize(hm, size, interpolation=cv2.INTER_CUBIC)
    hm8 = np.uint8(255 * hm)
    return cv2.applyColorMap(hm8, cv2.COLORMAP_JET)


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a Grad-CAM heatmap over an image. Returns BGR uint8."""
    h, w = image_bgr.shape[:2]
    color = heatmap_to_color(heatmap, (w, h))
    base = image_bgr.astype(np.float32)
    blended = (1 - alpha) * base + alpha * color.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_box(image_bgr: np.ndarray, box: tuple[int, int, int, int], label: str | None = None,
             color: tuple[int, int, int] = (0, 200, 255), thickness: int = 2) -> np.ndarray:
    """Draw an (x, y, w, h) box with an optional label. Returns a copy."""
    out = image_bgr.copy()
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(0, y - th - 8)
        cv2.rectangle(out, (x, y0), (x + tw + 8, y0 + th + 8), color, -1)
        cv2.putText(out, label, (x + 4, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return out


def draw_landmarks(image_bgr: np.ndarray, landmarks: np.ndarray | None,
                   color: tuple[int, int, int] = (255, 255, 255), radius: int = 3) -> np.ndarray:
    out = image_bgr.copy()
    if landmarks is None:
        return out
    for (px, py) in landmarks:
        cv2.circle(out, (int(px), int(py)), radius, color, -1, cv2.LINE_AA)
    return out


def side_by_side(*images: np.ndarray, pad: int = 8) -> np.ndarray:
    """Horizontally concatenate images of equal height with a white gap."""
    h = max(im.shape[0] for im in images)
    resized = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in images]
    gap = np.full((h, pad, 3), 255, dtype=np.uint8)
    parts: list[np.ndarray] = []
    for i, im in enumerate(resized):
        if i:
            parts.append(gap)
        parts.append(im)
    return np.concatenate(parts, axis=1)
