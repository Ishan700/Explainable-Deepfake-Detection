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
from preprocessing.tracking import FaceTracker
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
                out_dir: str | Path | None = None, with_explanation: bool = True,
                max_faces: int = config.MAX_FACES) -> dict:
        """
        Analyse a video or image. Every face in every sampled frame is scored;
        faces are linked across frames into people, and each person gets a
        verdict, per-frame scores, reliability and an explanation.

        The top-level fields (verdict, p_fake, frame_scores, explanation, ...)
        describe the *primary* person (seen most often / largest) so existing
        callers keep working; `people` holds everyone, and `overall` is the
        clip-level verdict (fake if any person is fake).
        """
        t0 = time.time()
        media_path = Path(media_path)
        kind = "video" if is_video(media_path) else "image"
        info = get_video_info(media_path) if kind == "video" else {}

        frames = sample_frames(media_path, frames_per_video)
        tracker = FaceTracker(max_faces=max_faces)
        n_detections = 0
        for frame_idx, frame in frames:
            faces = self.face_detector.detect(frame)[:max_faces]
            crops = [self.face_detector.crop(frame, f) for f in faces]
            keep = [(f, c) for f, c in zip(faces, crops) if c.size]
            n_detections += len(keep)
            if keep:
                tracker.update(frame_idx, frame, [f for f, _ in keep], [c for _, c in keep])

        min_frames = config.MIN_TRACK_FRAMES if kind == "video" and len(frames) >= config.MIN_TRACK_FRAMES else 1
        people = tracker.people(min_frames=min_frames)
        if not people and tracker.tracks:          # everyone was brief; keep the best anyway
            people = tracker.people(min_frames=1)

        result: dict = {
            "status": "ok",
            "input": {"name": media_path.name, "type": kind, "frames_sampled": len(frames),
                      "faces_detected": sum(len(t) for t in people), "people_found": len(people),
                      "detections_total": n_detections, **info},
            "threshold": self.threshold,
            "class_names": self.class_names,
        }
        if not people:
            result.update(status="no_face", message="No face was detected in the sampled frames.",
                          timing_s=round(time.time() - t0, 2))
            return result

        # ---- score every crop of every person in one batch ---------------- #
        all_crops = [c for t in people for c in t.crops]
        batch = preprocess_batch(all_crops)
        probs = self.model.predict(batch, batch_size=config.BATCH_SIZE, verbose=0)
        p_fake_all = probs[:, self.fake_idx]

        people_out: list[dict] = []
        offset = 0
        for t in people:
            n = len(t)
            idx = slice(offset, offset + n)
            person = self._person_result(kind, t, batch[idx], p_fake_all[idx], with_explanation,
                                         None if out_dir is None else Path(out_dir) / f"person{t.track_id + 1}")
            people_out.append(person)
            offset += n

        # ---- clip-level verdict: fake if anyone is fake --------------------- #
        fakes = [p for p in people_out if p["verdict"] == config.CLASS_FAKE]
        overall_verdict = config.CLASS_FAKE if fakes else config.CLASS_REAL
        overall_conf = max(p["confidence"] for p in fakes) if fakes else min(p["confidence"] for p in people_out)
        result["overall"] = {
            "verdict": overall_verdict,
            "confidence": round(float(overall_conf), 4),
            "people": len(people_out),
            "fake_people": len(fakes),
            "summary": self._overall_summary(people_out),
        }
        result["people"] = people_out

        # primary person = first (most prominent); mirror its fields at the top level
        primary = people_out[0]
        for key in ("verdict", "confidence", "p_fake", "frame_scores", "stats", "reliability", "explanation", "images"):
            if key in primary:
                result[key] = primary[key]

        if out_dir is not None and len(people_out) > 1:
            result["overview_image"] = str(self._render_overview(Path(out_dir), people, people_out))

        result["timing_s"] = round(time.time() - t0, 2)
        return result

    # ------------------------------------------------------------------ #
    def _person_result(self, kind: str, track, batch: np.ndarray, p_fake: np.ndarray,
                       with_explanation: bool, out_dir: Path | None) -> dict:
        mean_fake = float(np.mean(p_fake))
        verdict = config.CLASS_FAKE if mean_fake >= self.threshold else config.CLASS_REAL
        confidence = mean_fake if verdict == config.CLASS_FAKE else 1.0 - mean_fake
        person: dict = {
            "person_id": int(track.track_id + 1),
            "label": f"Person {track.track_id + 1}",
            "frames_seen": len(track),
            "mean_face_size": int(round(track.mean_area ** 0.5)),
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "p_fake": round(mean_fake, 4),
            "frame_scores": [
                {"frame_index": int(fi), "p_fake": round(float(p), 4), "face_score": round(float(f.score), 3)}
                for fi, f, p in zip(track.frames, track.faces, p_fake)
            ],
            "stats": {
                "mean": round(mean_fake, 4),
                "std": round(float(np.std(p_fake)), 4),
                "min": round(float(np.min(p_fake)), 4),
                "max": round(float(np.max(p_fake)), 4),
                "frames_above_threshold": int(np.sum(p_fake >= self.threshold)),
            },
            "reliability": self._reliability(kind, len(track), float(np.std(p_fake)), mean_fake),
        }
        if with_explanation:
            key = int(np.argmax(p_fake)) if verdict == config.CLASS_FAKE else int(np.argmin(p_fake))
            frame_idx, frame, face, crop = track.frames[key], track.frame_images[key], track.faces[key], track.crops[key]
            class_idx = self.fake_idx if verdict == config.CLASS_FAKE else self.real_idx
            cam, frame_probs = self.gradcam.compute(batch[key], class_idx)
            landmarks_crop = self.face_detector.landmarks_in_crop(face)
            exp: Explanation = explain(cam, verdict, float(frame_probs[class_idx]), landmarks_crop, crop.shape[0])
            person["explanation"] = {**exp.to_dict(), "key_frame_index": int(frame_idx),
                                     "key_frame_p_fake": round(float(frame_probs[self.fake_idx]), 4)}
            if out_dir is not None:
                person["images"] = self._render(out_dir, frame, face, crop, cam, landmarks_crop, exp, verdict,
                                                label=person["label"])
        return person

    @staticmethod
    def _overall_summary(people: list[dict]) -> str:
        if len(people) == 1:
            p = people[0]
            return f"One face found: {p['verdict']} ({p['confidence'] * 100:.0f}% confidence)."
        parts = [f"{p['label']} {p['verdict']} {p['confidence'] * 100:.0f}%" for p in people]
        n_fake = sum(p["verdict"] == config.CLASS_FAKE for p in people)
        head = (f"{len(people)} faces found, {n_fake} judged fake. " if n_fake
                else f"{len(people)} faces found, all judged real. ")
        return head + "; ".join(parts) + "."

    def _render_overview(self, out_dir: Path, tracks, people_out: list[dict]) -> Path:
        """One full frame with every person's box and verdict - the frame where most people are visible."""
        counts: dict[int, int] = {}
        for t in tracks:
            for fi in t.frames:
                counts[fi] = counts.get(fi, 0) + 1
        best_frame = max(counts, key=lambda k: (counts[k], -k))
        frame = None
        for t in tracks:
            if best_frame in t.frames:
                frame = t.frame_images[t.frames.index(best_frame)]
                break
        img = frame.copy()
        for t, p in zip(tracks, people_out):
            if best_frame not in t.frames:
                continue
            face = t.faces[t.frames.index(best_frame)]
            fake = p["verdict"] == config.CLASS_FAKE
            color = (28, 35, 180) if fake else (63, 112, 25)
            img = draw_box(img, face.crop_box, f"{p['label']}: {p['verdict'].upper()} {p['confidence'] * 100:.0f}%",
                           color=color, thickness=3)
        h, w = img.shape[:2]
        if w > 1280:
            img = cv2.resize(img, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
        return save_image(out_dir / "overview.jpg", img)

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
                landmarks_crop: np.ndarray | None, exp: Explanation, verdict: str,
                label: str | None = None) -> dict[str, str]:
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

        label = f"{label + ': ' if label else ''}{verdict.upper()} {exp.confidence * 100:.0f}%"
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
def _print_person(p: dict, threshold: float) -> None:
    print(f"\n{p['label']}  (seen in {p['frames_seen']} frame(s))")
    print(f"Verdict: {p['verdict'].upper()}  (confidence {p['confidence'] * 100:.1f}%, "
          f"mean P(fake)={p['p_fake']:.3f}, threshold {threshold})")
    s = p["stats"]
    print(f"Frames : P(fake) min {s['min']:.2f} / mean {s['mean']:.2f} / max {s['max']:.2f}  "
          f"(std {s['std']:.2f}); {s['frames_above_threshold']} frame(s) above threshold")
    r = p["reliability"]
    print(f"Reliability: {r['level']} - {'; '.join(r['reasons'])}")
    if "explanation" in p:
        e = p["explanation"]
        print(f"Why (key frame #{e['key_frame_index']}, attention {e['concentration_label']}):")
        for reg in e["regions"][:4]:
            if reg["share"] >= 0.03:
                print(f"  {reg['label']:<24}{reg['share'] * 100:5.1f}%")
        print(f"{e['summary']}")
    if "images" in p:
        print("Images: " + ", ".join(p["images"].values()))


def _print_summary(res: dict) -> None:
    print(f"\nInput : {res['input']['name']} ({res['input']['type']}, {res['input']['frames_sampled']} frames sampled, "
          f"{res['input'].get('people_found', 0)} person(s) found)")
    if res["status"] != "ok":
        print(f"Status: {res['status']} - {res.get('message', '')}")
        return
    o = res["overall"]
    print(f"Overall: {o['verdict'].upper()}  ({o['summary']})")
    for p in res["people"]:
        _print_person(p, res["threshold"])
    if "overview_image" in res:
        print(f"\nOverview image: {res['overview_image']}")
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
