"""
Link faces across sampled frames into "people" (tracks).

Frames are sampled sparsely (e.g. every 15th frame), so classic IoU tracking is
too strict: a person who turns or walks moves their box a lot between samples.
We match on two lenient rules instead - the box centre must not have moved
further than TRACK_MAX_CENTRE_DIST * box size, and the box size must not have
changed by more than TRACK_SIZE_RATIO. Faces in a frame are assigned greedily,
largest first, each to the closest free track; unmatched faces start a new
track. Good enough for a handful of people in a clip; not a re-identification
system (two people swapping places could swap tracks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import config
from preprocessing.face_detection import Face


@dataclass
class Track:
    """One person: every sampled frame in which they were found."""
    track_id: int
    frames: list[int] = field(default_factory=list)          # frame indices
    faces: list[Face] = field(default_factory=list)          # detection per frame
    crops: list[np.ndarray] = field(default_factory=list)    # BGR face crop per frame
    frame_images: list[np.ndarray] = field(default_factory=list)  # full frame per detection

    @property
    def last_box(self) -> tuple[int, int, int, int]:
        return self.faces[-1].box

    @property
    def mean_area(self) -> float:
        return float(np.mean([f.area for f in self.faces])) if self.faces else 0.0

    def __len__(self) -> int:
        return len(self.faces)


def _centre_size(box: tuple[int, int, int, int]) -> tuple[float, float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0, math.sqrt(max(1.0, w * h))


def _distance(box_a, box_b) -> float | None:
    """Normalised centre distance if the pair is a plausible match, else None."""
    cxa, cya, sa = _centre_size(box_a)
    cxb, cyb, sb = _centre_size(box_b)
    ratio = max(sa, sb) / max(1.0, min(sa, sb))
    if ratio > config.TRACK_SIZE_RATIO:
        return None
    d = math.hypot(cxa - cxb, cya - cyb) / max(sa, sb)
    return d if d < config.TRACK_MAX_CENTRE_DIST else None


class FaceTracker:
    def __init__(self, max_faces: int = config.MAX_FACES):
        self.max_faces = max_faces
        self.tracks: list[Track] = []

    def update(self, frame_idx: int, frame: np.ndarray, faces: list[Face], crops: list[np.ndarray]) -> None:
        """Assign this frame's faces (largest first) to existing tracks or new ones."""
        used: set[int] = set()
        for face, crop in list(zip(faces, crops))[: self.max_faces]:
            best, best_d = None, float("inf")
            for t in self.tracks:
                if t.track_id in used or t.frames[-1] == frame_idx:
                    continue
                d = _distance(t.last_box, face.box)
                if d is not None and d < best_d:
                    best, best_d = t, d
            if best is None:
                best = Track(track_id=len(self.tracks))
                self.tracks.append(best)
            used.add(best.track_id)
            best.frames.append(frame_idx)
            best.faces.append(face)
            best.crops.append(crop)
            best.frame_images.append(frame)

    def people(self, min_frames: int = 1) -> list[Track]:
        """Tracks worth reporting, most prominent first (seen often and large)."""
        keep = [t for t in self.tracks if len(t) >= min_frames]
        keep.sort(key=lambda t: (len(t), t.mean_area), reverse=True)
        for i, t in enumerate(keep):
            t.track_id = i
        return keep[: self.max_faces]
