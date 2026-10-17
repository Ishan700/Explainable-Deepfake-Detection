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


_COLORMAP = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)


def _resize_heat(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    hm = cv2.resize(heatmap.astype(np.float32), size, interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0.0, None)
    if hm.max() > 0:
        hm = hm / hm.max()
    return hm


def heatmap_to_color(heatmap: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """[0,1] float heatmap -> BGR colour image (optionally resized to (w, h))."""
    hm = np.clip(heatmap, 0.0, 1.0)
    if size is not None:
        hm = _resize_heat(hm, size)
    return cv2.applyColorMap(np.uint8(255 * hm), _COLORMAP)


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.65,
                    floor: float = 0.18, contour: float | None = 0.6) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap over an image so that only the *active* area is
    coloured: opacity ramps from 0 at `floor` to `alpha` at 1.0, so cold regions
    stay as the original photo instead of being tinted blue. A thin contour at
    the `contour` level outlines the strongest evidence. Returns BGR uint8.
    """
    h, w = image_bgr.shape[:2]
    hm = _resize_heat(heatmap, (w, h))
    color = cv2.applyColorMap(np.uint8(255 * hm), _COLORMAP).astype(np.float32)
    # per-pixel opacity: smoothstep ramp from floor..1
    t = np.clip((hm - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)
    a = (t * t * (3 - 2 * t)) * alpha
    a = cv2.GaussianBlur(a, (0, 0), 2)[..., None]
    base = image_bgr.astype(np.float32)
    out = (1 - a) * base + a * color
    out = np.clip(out, 0, 255).astype(np.uint8)
    if contour is not None:
        mask = (hm >= contour).astype(np.uint8)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 2, cv2.LINE_AA)
    return out


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
