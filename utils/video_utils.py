"""
Video -> frames.

We never process every frame: neighbouring frames are near-duplicates, so a
handful of uniformly spaced samples carries almost all the information at a
fraction of the cost. `sample_frames` is the single function used by both
prepare_dataset.py (training data) and predict.py (inference), so the two
paths see frames chosen the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

import config


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in config.VIDEO_EXTENSIONS


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in config.IMAGE_EXTENSIONS


def get_video_info(path: str | Path) -> dict:
    """Return basic metadata; frame_count may be 0 for badly muxed files."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        return {
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        cap.release()


def _uniform_indices(frame_count: int, n: int) -> list[int]:
    if frame_count <= 0:
        return []
    n = max(1, min(n, frame_count))
    # Avoid the very first/last frames (often black or fade) when we can.
    lo, hi = (1, frame_count - 2) if frame_count > n + 2 else (0, frame_count - 1)
    return sorted({int(round(v)) for v in np.linspace(lo, hi, n)})


def sample_frames(path: str | Path, n: int = config.FRAMES_PER_VIDEO) -> list[tuple[int, np.ndarray]]:
    """
    Uniformly sample up to `n` frames from a video (or 1 "frame" from an image).

    Returns a list of (frame_index, BGR ndarray). Seeking with
    CAP_PROP_POS_FRAMES is fast but unreliable for some codecs, so if a seek
    fails we fall back to a sequential scan that keeps only the wanted indices.
    """
    path = Path(path)
    if is_image(path):
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Cannot read image: {path}")
        return [(0, img)]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        wanted = _uniform_indices(frame_count, n)

        frames: list[tuple[int, np.ndarray]] = []
        seek_ok = True
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                seek_ok = False
                break
            frames.append((idx, frame))

        if seek_ok and frames:
            return frames

        # Fallback: sequential read (also handles frame_count == 0).
        cap.release()
        cap = cv2.VideoCapture(str(path))
        frames = []
        all_frames: list[np.ndarray] = []
        if frame_count <= 0:
            # Unknown length: read everything, then subsample.
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                all_frames.append(frame)
            for idx in _uniform_indices(len(all_frames), n):
                frames.append((idx, all_frames[idx]))
            return frames

        wanted_set = set(wanted)
        idx = 0
        while idx <= max(wanted_set):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted_set:
                frames.append((idx, frame))
            idx += 1
        return frames
    finally:
        cap.release()


def iter_media_files(root: str | Path) -> Iterator[Path]:
    """Recursively yield every video or image file under `root`, sorted."""
    root = Path(root)
    exts = config.VIDEO_EXTENSIONS | config.IMAGE_EXTENSIONS
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p
