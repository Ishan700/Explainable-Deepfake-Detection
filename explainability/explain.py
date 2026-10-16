"""
From "where" to "why".

A Grad-CAM heatmap over the mouth tells you the network attended to the
mouth. It does not tell a human what that means. This module converts a
heatmap into a region-level attribution ("48% of the evidence is in the mouth
region, 21% around the left eye") and attaches the artefact class that each
facial region is known to carry in deepfake forensics literature
(FaceForensics++, DFDC analyses, Li et al. on blending boundaries, etc.).

Regions are built from the five YuNet landmarks when available (eyes, nose,
mouth corners). Without landmarks (Haar / pre-cropped inputs) a canonical
face template is used instead, and the output says so.

The hints are heuristics, not ground truth - they describe what attention on a
region *usually* corresponds to. The README is explicit about this.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

import config

# Ordered by assignment priority: a pixel inside two regions goes to the first.
REGION_ORDER = ["left_eye", "right_eye", "mouth", "nose", "forehead", "cheeks", "boundary", "background"]

REGION_LABELS = {
    "left_eye": "left eye",
    "right_eye": "right eye",
    "mouth": "mouth",
    "nose": "nose",
    "forehead": "forehead / hairline",
    "cheeks": "cheeks & skin",
    "boundary": "face boundary / jawline",
    "background": "area outside the face",
}

# What strong attention on a region usually indicates in a manipulated face.
FAKE_HINTS = {
    "left_eye": "inconsistent eye reflections, unnatural iris texture or irregular blinking - common in face-swap and reenactment outputs",
    "right_eye": "inconsistent eye reflections, unnatural iris texture or irregular blinking - common in face-swap and reenactment outputs",
    "mouth": "lip-sync and inner-mouth artefacts: blurred or merged teeth, inconsistent tongue, lips that lag the speech - typical of reenactment methods",
    "nose": "over-smoothed or mismatched skin texture around the nose where generators struggle with fine pore detail",
    "forehead": "blending along the hairline where the synthetic face is pasted onto the original head",
    "cheeks": "skin that is too smooth, or colour and lighting that do not match the neck and ears",
    "boundary": "blending seams along the jawline where the swapped face meets the original frame - the classic face-swap tell",
    "background": "cues outside the face itself - compression patterns or frame-level statistics rather than facial detail - so treat this verdict with caution",
}

# Why a region looks authentic when the verdict is REAL.
REAL_HINTS = {
    "left_eye": "natural eye detail and consistent reflections",
    "right_eye": "natural eye detail and consistent reflections",
    "mouth": "sharp, consistent teeth and lip detail",
    "nose": "fine, consistent skin texture",
    "forehead": "a hairline that blends naturally with the skin",
    "cheeks": "consistent skin texture and lighting across the face",
    "boundary": "no visible seam between the face and the surrounding frame",
    "background": "context outside the face (weak evidence either way)",
}


@dataclass
class RegionScore:
    name: str
    label: str
    share: float          # fraction of total heatmap mass in this region (0-1)
    peak: float           # max heatmap value inside the region (0-1)
    hint: str


@dataclass
class Explanation:
    verdict: str                     # "fake" | "real"
    confidence: float                # probability of the verdict class
    regions: list[RegionScore]       # sorted by share, descending
    concentration: float             # share of mass in the top 10% of pixels
    concentration_label: str         # "highly localised" | "moderately localised" | "diffuse"
    landmark_source: str             # "detector" | "template"
    summary: str                     # human-readable paragraph(s)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regions"] = [asdict(r) for r in self.regions]
        return d


# --------------------------------------------------------------------------- #
# Landmark handling
# --------------------------------------------------------------------------- #
def template_landmarks(size: int, margin: float = config.FACE_MARGIN) -> np.ndarray:
    """
    Approximate landmark positions for a centred face crop taken with `margin`.
    Order matches YuNet: right eye, left eye, nose, right mouth, left mouth
    ("right" = the subject's right = image left).
    """
    face = size / (1.0 + 2.0 * margin)          # face box side inside the crop
    top = (size - face) / 2.0
    left = (size - face) / 2.0
    cx = size / 2.0
    eye_y = top + 0.40 * face
    return np.array(
        [
            [cx - 0.20 * face, eye_y],            # right eye (image left)
            [cx + 0.20 * face, eye_y],            # left eye (image right)
            [cx, top + 0.62 * face],              # nose tip
            [cx - 0.17 * face, top + 0.80 * face],  # right mouth corner
            [cx + 0.17 * face, top + 0.80 * face],  # left mouth corner
        ],
        dtype=np.float32,
    )


def _scale_landmarks(landmarks: np.ndarray, from_size: int, to_size: int) -> np.ndarray:
    return landmarks * (float(to_size) / float(from_size))


# --------------------------------------------------------------------------- #
# Region geometry
# --------------------------------------------------------------------------- #
def region_geometry(landmarks: np.ndarray, size: int) -> dict[str, dict]:
    """
    Ellipses (centre, axes) and bands describing each region, in pixel
    coordinates of a size x size image. Derived from the inter-ocular distance
    so it adapts to the face scale inside the crop.
    """
    re, le, nose, rm, lm = landmarks
    ed = float(np.linalg.norm(le - re)) or size * 0.25     # inter-ocular distance
    mouth_c = (rm + lm) / 2.0
    mouth_w = float(np.linalg.norm(lm - rm)) or 0.8 * ed
    eye_c = (re + le) / 2.0

    face_c = np.array([(re[0] + le[0] + nose[0] + rm[0] + lm[0]) / 5.0, (eye_c[1] + mouth_c[1]) / 2.0])
    face_axes = (1.35 * ed, 1.75 * ed)

    return {
        "right_eye": {"type": "ellipse", "center": re, "axes": (0.42 * ed, 0.28 * ed)},
        "left_eye": {"type": "ellipse", "center": le, "axes": (0.42 * ed, 0.28 * ed)},
        "nose": {"type": "ellipse", "center": nose, "axes": (0.32 * ed, 0.42 * ed)},
        "mouth": {"type": "ellipse", "center": mouth_c, "axes": (0.70 * mouth_w, 0.38 * mouth_w)},
        "forehead": {"type": "band", "y_max": eye_c[1] - 0.55 * ed,
                     "x_min": face_c[0] - 1.1 * ed, "x_max": face_c[0] + 1.1 * ed},
        "face": {"type": "ellipse", "center": face_c, "axes": face_axes},
        "outer": {"type": "ellipse", "center": face_c, "axes": (1.30 * face_axes[0], 1.25 * face_axes[1])},
    }


def _ellipse_mask(shape: tuple[int, int], center, axes) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(mask, (int(round(center[0])), int(round(center[1]))),
                (max(1, int(round(axes[0]))), max(1, int(round(axes[1])))), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def region_masks(landmarks: np.ndarray, size: int) -> dict[str, np.ndarray]:
    """One boolean mask per region; masks are disjoint and cover the image."""
    geo = region_geometry(landmarks, size)
    shape = (size, size)
    yy, xx = np.mgrid[0:size, 0:size]

    raw = {
        "left_eye": _ellipse_mask(shape, geo["left_eye"]["center"], geo["left_eye"]["axes"]),
        "right_eye": _ellipse_mask(shape, geo["right_eye"]["center"], geo["right_eye"]["axes"]),
        "mouth": _ellipse_mask(shape, geo["mouth"]["center"], geo["mouth"]["axes"]),
        "nose": _ellipse_mask(shape, geo["nose"]["center"], geo["nose"]["axes"]),
    }
    face = _ellipse_mask(shape, geo["face"]["center"], geo["face"]["axes"])
    outer = _ellipse_mask(shape, geo["outer"]["center"], geo["outer"]["axes"])
    fb = geo["forehead"]
    raw["forehead"] = outer & (yy < fb["y_max"]) & (xx >= fb["x_min"]) & (xx <= fb["x_max"])
    raw["cheeks"] = face
    raw["boundary"] = outer & ~face
    raw["background"] = ~outer

    assigned = np.zeros(shape, dtype=bool)
    masks: dict[str, np.ndarray] = {}
    for name in REGION_ORDER:
        m = raw[name] & ~assigned
        masks[name] = m
        assigned |= m
    # Anything left (shouldn't happen) counts as background.
    masks["background"] |= ~assigned
    return masks


def draw_region_outlines(image_bgr: np.ndarray, landmarks: np.ndarray, highlight: list[str],
                         color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Outline the highlighted regions on a copy of the image (for the result page)."""
    out = image_bgr.copy()
    size = image_bgr.shape[0]
    geo = region_geometry(landmarks, size)
    for name in highlight:
        if name in ("cheeks",):
            g = geo["face"]
        elif name in ("boundary",):
            g = geo["outer"]
        elif name in ("background", "forehead"):
            continue
        else:
            g = geo[name]
        cv2.ellipse(out, (int(g["center"][0]), int(g["center"][1])),
                    (int(g["axes"][0]), int(g["axes"][1])), 0, 0, 360, color, 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def upsample_heatmap(cam: np.ndarray, size: int) -> np.ndarray:
    hm = cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    hm = np.clip(hm, 0.0, None)
    if hm.max() > 0:
        hm = hm / hm.max()
    return hm


def concentration_score(heatmap: np.ndarray, top_fraction: float = 0.10) -> float:
    """Share of total activation mass contained in the top `top_fraction` of pixels."""
    flat = np.sort(heatmap.ravel())[::-1]
    total = flat.sum()
    if total <= 0:
        return 0.0
    k = max(1, int(len(flat) * top_fraction))
    return float(flat[:k].sum() / total)


def _concentration_label(c: float) -> str:
    if c >= 0.45:
        return "highly localised"
    if c >= 0.25:
        return "moderately localised"
    return "diffuse"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def explain(cam: np.ndarray, verdict: str, confidence: float,
            landmarks_crop: np.ndarray | None, crop_size: int,
            analysis_size: int = config.IMG_SIZE) -> Explanation:
    """
    cam            : raw Grad-CAM output (small, e.g. 7x7), values 0-1
    verdict        : "fake" or "real" (the class the CAM was computed for)
    confidence     : softmax probability of that class
    landmarks_crop : (5,2) landmarks in crop coordinates, or None
    crop_size      : side length of the square crop the landmarks refer to
    """
    size = analysis_size
    heat = upsample_heatmap(cam, size)

    if landmarks_crop is not None:
        lms = _scale_landmarks(np.asarray(landmarks_crop, dtype=np.float32), crop_size, size)
        source = "detector"
    else:
        lms = template_landmarks(size)
        source = "template"

    masks = region_masks(lms, size)
    total = float(heat.sum()) or 1.0
    hints = FAKE_HINTS if verdict == config.CLASS_FAKE else REAL_HINTS

    regions = []
    for name in REGION_ORDER:
        m = masks[name]
        share = float(heat[m].sum() / total) if m.any() else 0.0
        peak = float(heat[m].max()) if m.any() else 0.0
        regions.append(RegionScore(name=name, label=REGION_LABELS[name], share=share, peak=peak, hint=hints[name]))
    regions.sort(key=lambda r: r.share, reverse=True)

    conc = concentration_score(heat)
    conc_label = _concentration_label(conc)
    summary = _write_summary(verdict, confidence, regions, conc_label, source)

    return Explanation(verdict=verdict, confidence=float(confidence), regions=regions,
                       concentration=conc, concentration_label=conc_label,
                       landmark_source=source, summary=summary)


def _write_summary(verdict: str, confidence: float, regions: list[RegionScore],
                   conc_label: str, source: str) -> str:
    top = regions[0]
    second = regions[1] if len(regions) > 1 else None
    is_fake = verdict == config.CLASS_FAKE

    parts: list[str] = []
    parts.append(
        f"The model classified this face as {verdict.upper()} with {_pct(confidence)} confidence. "
        f"Its attention is {conc_label}: {_pct(top.share)} of the Grad-CAM activation falls on the "
        f"{top.label}"
        + (f" and {_pct(second.share)} on the {second.label}." if second and second.share >= 0.10 else ".")
    )

    if is_fake:
        parts.append(f"Attention on the {top.label} usually points to {top.hint}.")
        if second and second.share >= 0.15 and second.hint != top.hint:
            parts.append(f"The secondary focus on the {second.label} suggests {second.hint}.")
        if top.name == "background":
            parts.append("Because the strongest evidence lies outside the face, this prediction is less "
                         "trustworthy than one grounded in facial detail.")
        elif conc_label == "diffuse":
            parts.append("A diffuse map means no single artefact dominates; the network is responding to "
                         "texture statistics spread across the face, which is common for heavily "
                         "compressed or low-resolution fakes.")
    else:
        parts.append(f"The strongest evidence for authenticity comes from the {top.label}: {top.hint}.")
        if conc_label == "diffuse":
            parts.append("A spread-out map is expected for real faces - there is no single artefact to "
                         "latch onto, so the network draws on consistent detail across the whole face.")

    if source == "template":
        parts.append("Facial regions were estimated from a standard face template because no landmarks "
                     "were available for this crop, so region boundaries are approximate.")
    return " ".join(parts)
