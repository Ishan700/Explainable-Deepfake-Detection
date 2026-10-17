#!/usr/bin/env python3
"""
Video / image -> verdict + explanation.

    python predict.py path/to/video.mp4
    python predict.py photo.jpg --out results/my_run
    python predict.py clip.mp4 --json           # machine-readable output only

Inference is VIDEO-level: FRAMES_PER_VIDEO frames are sampled, the largest
face in each is scored, and the mean P(fake) decides the verdict. Grad-CAM is
then run once, on the single frame that most strongly supports the verdict,
and explainability.explain turns that heatmap into a region breakdown and a
plain-English reason.

`DeepfakeDetector` is also what app.py uses, so the CLI and the web demo can
never disagree.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

import config
from explainability.explain import Explanation, draw_region_outlines, explain, template_landmarks
from explainability.gradcam import GradCAM
from preprocessing.face_detection import Face, FaceDetector
from preprocessing.preprocessing import preprocess_batch
from utils.image_utils import draw_box, draw_landmarks, overlay_heatmap, save_image
from utils.video_utils import get_video_info, is_video, sample_frames


class DeepfakeDetector:
    """Loads the model once; call `analyze()` per upload."""

    def __init__(self, model_path: Path = config.MODEL_PATH,
                 class_names_path: Path = config.CLASS_NAMES_PATH,
                 detector_backend: str | None = None,
                 threshold: float = config.THRESHOLD):
        import keras  # imported here so `import predict` stays cheap for the web app's error page

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}. Train one with train.py first.")
        self.model = keras.models.load_model(model_path)
        with open(class_names_path) as f:
            self.class_names: list[str] = json.load(f)
        self.fake_idx = self.class_names.index(config.CLASS_FAKE)
        self.real_idx = self.class_names.index(config.CLASS_REAL)
        self.threshold = threshold
        self.face_detector = FaceDetector(backend=detector_backend)
        self.gradcam = GradCAM(self.model)
        # Warm-up so the first real request isn't slow.
        self.model.predict(np.zeros((1, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32), verbose=0)

    # ------------------------------------------------------------------ #
    def analyze(self, media_path: str | Path, frames_per_video: int = config.FRAMES_PER_VIDEO,
                out_dir: str | Path | None = None, with_explanation: bool = True) -> dict:
        t0 = time.time()
        media_path = Path(media_path)
        kind = "video" if is_video(media_path) else "image"
        info = get_video_info(media_path) if kind == "video" else {}

        frames = sample_frames(media_path, frames_per_video)
        detections: list[tuple[int, np.ndarray, Face, np.ndarray]] = []
        for frame_idx, frame in frames:
            face = self.face_detector.largest_face(frame)
            if face is None:
                continue
            crop = self.face_detector.crop(frame, face)
            if crop.size:
                detections.append((frame_idx, frame, face, crop))

        result: dict = {
            "status": "ok",
            "input": {"name": media_path.name, "type": kind, "frames_sampled": len(frames),
                      "faces_detected": len(detections), **info},
            "threshold": self.threshold,
            "class_names": self.class_names,
        }
        if not detections:
            result.update(status="no_face", message="No face was detected in the sampled frames.",
                          timing_s=round(time.time() - t0, 2))
            return result

        # ---- score every detected face ------------------------------------ #
        batch = preprocess_batch([d[3] for d in detections])
        probs = self.model.predict(batch, batch_size=config.BATCH_SIZE, verbose=0)
        p_fake = probs[:, self.fake_idx]

        mean_fake = float(np.mean(p_fake))
        verdict = config.CLASS_FAKE if mean_fake >= self.threshold else config.CLASS_REAL
        confidence = mean_fake if verdict == config.CLASS_FAKE else 1.0 - mean_fake

        result.update({
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "p_fake": round(mean_fake, 4),
            "frame_scores": [
                {"frame_index": int(d[0]), "p_fake": round(float(p), 4), "face_score": round(float(d[2].score), 3)}
                for d, p in zip(detections, p_fake)
            ],
            "stats": {
                "mean": round(mean_fake, 4),
                "std": round(float(np.std(p_fake)), 4),
                "min": round(float(np.min(p_fake)), 4),
                "max": round(float(np.max(p_fake)), 4),
                "frames_above_threshold": int(np.sum(p_fake >= self.threshold)),
            },
        })
        result["reliability"] = self._reliability(kind, len(detections), float(np.std(p_fake)), mean_fake)

        # ---- explain the most representative frame ------------------------ #
        if with_explanation:
            key = int(np.argmax(p_fake)) if verdict == config.CLASS_FAKE else int(np.argmin(p_fake))
            frame_idx, frame, face, crop = detections[key]
            class_idx = self.fake_idx if verdict == config.CLASS_FAKE else self.real_idx
            cam, frame_probs = self.gradcam.compute(batch[key], class_idx)
            landmarks_crop = self.face_detector.landmarks_in_crop(face)
            exp: Explanation = explain(cam, verdict, float(frame_probs[class_idx]), landmarks_crop, crop.shape[0])
            result["explanation"] = {**exp.to_dict(), "key_frame_index": int(frame_idx),
                                     "key_frame_p_fake": round(float(frame_probs[self.fake_idx]), 4)}
            if out_dir is not None:
                result["images"] = self._render(Path(out_dir), frame, face, crop, cam, landmarks_crop, exp, verdict)

        result["timing_s"] = round(time.time() - t0, 2)
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _reliability(kind: str, n_faces: int, std: float, mean_fake: float) -> dict:
        reasons: list[str] = []
        level = "high"
        if kind == "video" and n_faces < config.MIN_FACES_FOR_VERDICT:
            level = "low"
            reasons.append(f"only {n_faces} face(s) found across the sampled frames")
        if std > 0.30:
            level = "low"
            reasons.append("frame scores disagree strongly with each other")
        elif std > 0.15 and level != "low":
            level = "medium"
            reasons.append("frame scores vary noticeably")
        if abs(mean_fake - config.THRESHOLD) < 0.10:
            level = "low" if level == "low" else "medium"
            reasons.append("mean score is close to the decision threshold")
        if kind == "image":
            reasons.append("single image: no temporal consistency check possible")
            if level == "high":
                level = "medium"
        if not reasons:
            reasons.append("consistent scores across frames")
        return {"level": level, "reasons": reasons}

    def _render(self, out_dir: Path, frame: np.ndarray, face: Face, crop: np.ndarray, cam: np.ndarray,
                landmarks_crop: np.ndarray | None, exp: Explanation, verdict: str) -> dict[str, str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        disp = 512  # display size for the crop-based panels
        crop_disp = cv2.resize(crop, (disp, disp), interpolation=cv2.INTER_CUBIC)
        heat_disp = overlay_heatmap(crop_disp, cam)

        lms = landmarks_crop
        if lms is None:
            lms = template_landmarks(crop.shape[0])
        lms_disp = lms * (disp / crop.shape[0])
        top_regions = [r.name for r in exp.regions[:2] if r.share >= 0.10]
        # Regions view: soft heat (no contour) + outlines of the attributed regions + landmarks
        regions_img = draw_region_outlines(overlay_heatmap(crop_disp, cam, alpha=0.4, contour=None),
                                           lms_disp, top_regions)
        if landmarks_crop is not None:
            regions_img = draw_landmarks(regions_img, lms_disp, radius=4)

        label = f"{verdict.upper()} {exp.confidence * 100:.0f}%"
        color = (28, 35, 180) if verdict == config.CLASS_FAKE else (63, 112, 25)
        frame_img = draw_box(frame, face.crop_box, label, color=color, thickness=3)
        h, w = frame_img.shape[:2]
        if w > 960:
            frame_img = cv2.resize(frame_img, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA)

        paths = {
            "frame": save_image(out_dir / "key_frame.jpg", frame_img),
            "crop": save_image(out_dir / "face_crop.jpg", crop_disp),
            "heatmap": save_image(out_dir / "gradcam.jpg", heat_disp),
            "regions": save_image(out_dir / "gradcam_regions.jpg", regions_img),
        }
        return {k: str(v) for k, v in paths.items()}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_summary(res: dict) -> None:
    print(f"\nInput : {res['input']['name']} ({res['input']['type']}, "
          f"{res['input']['faces_detected']}/{res['input']['frames_sampled']} frames with a face)")
    if res["status"] != "ok":
        print(f"Status: {res['status']} - {res.get('message', '')}")
        return
    print(f"Verdict: {res['verdict'].upper()}  (confidence {res['confidence'] * 100:.1f}%, "
          f"mean P(fake)={res['p_fake']:.3f}, threshold {res['threshold']})")
    s = res["stats"]
    print(f"Frames : P(fake) min {s['min']:.2f} / mean {s['mean']:.2f} / max {s['max']:.2f}  "
          f"(std {s['std']:.2f}); {s['frames_above_threshold']} frame(s) above threshold")
    r = res["reliability"]
    print(f"Reliability: {r['level']} - {'; '.join(r['reasons'])}")
    if "explanation" in res:
        e = res["explanation"]
        print(f"\nWhy (key frame #{e['key_frame_index']}, attention {e['concentration_label']}):")
        for reg in e["regions"][:4]:
            if reg["share"] >= 0.03:
                print(f"  {reg['label']:<24}{reg['share'] * 100:5.1f}%")
        print(f"\n{e['summary']}")
    if "images" in res:
        print("\nImages:")
        for k, v in res["images"].items():
            print(f"  {k:<8} {v}")
    print(f"\n({res['timing_s']} s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("media", type=Path, help="video or image file")
    ap.add_argument("--out", type=Path, default=None, help="folder for images + result.json "
                                                            "(default results/predictions/<name>/)")
    ap.add_argument("--frames", type=int, default=config.FRAMES_PER_VIDEO)
    ap.add_argument("--detector", choices=["yunet", "haar", "none"], default=None)
    ap.add_argument("--threshold", type=float, default=config.THRESHOLD)
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args()

    if not args.media.exists():
        raise SystemExit(f"File not found: {args.media}")
    out_dir = args.out or (config.RESULTS_DIR / "predictions" / args.media.stem)

    det = DeepfakeDetector(detector_backend=args.detector, threshold=args.threshold)
    res = det.analyze(args.media, frames_per_video=args.frames, out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump(res, f, indent=2)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        _print_summary(res)
        print(f"Full result: {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
