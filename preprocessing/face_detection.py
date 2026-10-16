"""
Face detection and cropping.

Default detector is YuNet, OpenCV's lightweight DNN face detector. It runs
fast on CPU, ships as a single 230 KB ONNX file, and returns five facial
landmarks (eyes, nose, mouth corners) alongside each box. Those landmarks are
what lets explainability/explain.py say "the model focused on the mouth"
instead of just drawing a heatmap.

A Haar-cascade fallback is included for environments where the ONNX model is
unavailable, and a "none" mode handles datasets that are already face crops.

Crops are taken with a margin (config.FACE_MARGIN). This matters: face-swap
blending seams sit at the border of the swapped region, often just outside a
tight detector box. A tight crop throws away exactly the evidence we want the
model (and Grad-CAM) to find.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import config


@dataclass
class Face:
    """One detected face in *frame* coordinates."""
    box: tuple[int, int, int, int]            # x, y, w, h (tight detector box)
    score: float
    landmarks: np.ndarray | None = None       # (5, 2): right eye, left eye, nose, right mouth, left mouth
    crop_box: tuple[int, int, int, int] = field(default=(0, 0, 0, 0))  # box after margin + squaring

    @property
    def area(self) -> int:
        return int(self.box[2]) * int(self.box[3])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_yunet_model(path: Path = config.YUNET_MODEL_PATH) -> Path:
    """Download the YuNet ONNX file if it is missing or corrupted."""
    if path.exists() and _sha256(path) == config.YUNET_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[face_detection] downloading YuNet model to {path} ...")
    urllib.request.urlretrieve(config.YUNET_URL, path)
    digest = _sha256(path)
    if digest != config.YUNET_SHA256:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"YuNet download failed checksum: {digest}")
    return path


class FaceDetector:
    """
    Wraps the chosen backend behind one interface:

        detector = FaceDetector()           # uses config.FACE_DETECTOR
        faces = detector.detect(frame_bgr)  # list[Face], largest first
        crop = detector.crop(frame_bgr, faces[0])
    """

    def __init__(self, backend: str | None = None,
                 score_threshold: float = config.FACE_SCORE_THRESHOLD,
                 margin: float = config.FACE_MARGIN,
                 min_size: int = config.MIN_FACE_SIZE):
        self.backend = (backend or config.FACE_DETECTOR).lower()
        self.score_threshold = score_threshold
        self.margin = margin
        self.min_size = min_size
        self._yunet = None
        self._haar = None
        self._yunet_size: tuple[int, int] | None = None

        if self.backend == "yunet":
            try:
                model_path = ensure_yunet_model()
                self._yunet = cv2.FaceDetectorYN.create(
                    str(model_path), "", (320, 320),
                    score_threshold=self.score_threshold, nms_threshold=0.3, top_k=50)
            except Exception as exc:  # pragma: no cover - network / build dependent
                if getattr(cv2, "CascadeClassifier", None) is None:
                    raise RuntimeError(f"YuNet face detector unavailable ({exc}) and this OpenCV build "
                                       "has no Haar cascade fallback.") from exc
                print(f"[face_detection] YuNet unavailable ({exc}); falling back to Haar cascade.")
                self.backend = "haar"

        if self.backend == "haar":
            # OpenCV 4.x ships CascadeClassifier + the XML files in cv2.data;
            # OpenCV 5 moved them out of the core wheel, so check explicitly.
            cascade = Path(getattr(cv2.data, "haarcascades", "")) / "haarcascade_frontalface_default.xml"
            ctor = getattr(cv2, "CascadeClassifier", None)
            if ctor is None or not cascade.exists():
                raise RuntimeError(
                    "Haar cascades are not available in this OpenCV build "
                    f"(opencv {cv2.__version__}). Use the default 'yunet' detector instead.")
            self._haar = ctor(str(cascade))
            if self._haar.empty():
                raise RuntimeError("Could not load Haar cascade from OpenCV data directory.")

        if self.backend not in {"yunet", "haar", "none"}:
            raise ValueError(f"Unknown face detector backend: {self.backend}")

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        h, w = frame_bgr.shape[:2]
        if self.backend == "none":
            # Treat the whole frame as the face (already-cropped datasets).
            face = Face(box=(0, 0, w, h), score=1.0)
            face.crop_box = self._square_box(face.box, w, h, margin=0.0)
            return [face]

        faces: list[Face] = []
        if self.backend == "yunet":
            if self._yunet_size != (w, h):
                self._yunet.setInputSize((w, h))
                self._yunet_size = (w, h)
            _, dets = self._yunet.detect(frame_bgr)
            if dets is not None:
                for d in dets:
                    x, y, bw, bh = d[:4]
                    lm = d[4:14].reshape(5, 2)
                    faces.append(Face(box=(int(x), int(y), int(bw), int(bh)),
                                      score=float(d[14]), landmarks=lm.astype(np.float32)))
        else:  # haar
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            rects = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                minSize=(self.min_size, self.min_size))
            for (x, y, bw, bh) in rects:
                faces.append(Face(box=(int(x), int(y), int(bw), int(bh)), score=1.0))

        faces = [f for f in faces if f.box[2] >= self.min_size and f.box[3] >= self.min_size]
        for f in faces:
            f.crop_box = self._square_box(f.box, w, h, margin=self.margin)
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    def largest_face(self, frame_bgr: np.ndarray) -> Face | None:
        faces = self.detect(frame_bgr)
        return faces[0] if faces else None

    # ------------------------------------------------------------------ #
    # Cropping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _square_box(box: tuple[int, int, int, int], frame_w: int, frame_h: int,
                    margin: float) -> tuple[int, int, int, int]:
        """Expand by `margin`, make square around the centre, clamp to the frame."""
        x, y, w, h = box
        cx, cy = x + w / 2.0, y + h / 2.0
        side = max(w, h) * (1.0 + 2.0 * margin)
        side = min(side, frame_w, frame_h)
        x0 = int(round(cx - side / 2.0))
        y0 = int(round(cy - side / 2.0))
        x0 = max(0, min(x0, frame_w - int(side)))
        y0 = max(0, min(y0, frame_h - int(side)))
        return (x0, y0, int(side), int(side))

    def crop(self, frame_bgr: np.ndarray, face: Face) -> np.ndarray:
        x, y, w, h = face.crop_box
        return frame_bgr[y:y + h, x:x + w].copy()

    @staticmethod
    def landmarks_in_crop(face: Face) -> np.ndarray | None:
        """Translate frame-space landmarks into crop-space coordinates."""
        if face.landmarks is None:
            return None
        x0, y0, _, _ = face.crop_box
        return face.landmarks - np.array([x0, y0], dtype=np.float32)
