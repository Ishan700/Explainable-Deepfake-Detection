#!/usr/bin/env python3
"""
Evaluate the trained model on a held-out split.

    python evaluate.py                                  # dataset/test
    python evaluate.py --split validation
    python evaluate.py --dataset /path/to/other_dataset # cross-dataset test (same folder layout)
    python evaluate.py --gradcam-samples 6              # also dump Grad-CAM overlays

Reports both FRAME-level and VIDEO-level metrics. Video-level numbers are the
ones that matter for the application (we classify uploads, not frames) and
they come from averaging P(fake) over every crop that shares a source video in
manifest.csv. Accuracy is reported, but AUC is the headline figure because the
classes are rarely balanced.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import keras
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)

import config
from train import load_split


def _metrics(y_true: np.ndarray, p_fake: np.ndarray, threshold: float) -> dict:
    """y_true: 1 = fake, 0 = real. p_fake: P(fake)."""
    y_pred = (p_fake >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, p_fake))
        out["log_loss"] = float(log_loss(y_true, np.clip(p_fake, 1e-7, 1 - 1e-7)))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["confusion_matrix"] = {"labels": ["real", "fake"], "matrix": cm.tolist()}
    return out


def _plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["real", "fake"]); ax.set_yticklabels(["real", "fake"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(title)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130); plt.close(fig)


def _plot_roc(y_true: np.ndarray, p_fake: np.ndarray, path: Path, auc: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y_true, p_fake)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC (frame level)"); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _save_gradcam_samples(model, ds_dir: Path, class_names: list[str], n_per_class: int, out_dir: Path) -> None:
    import cv2
    from explainability.gradcam import GradCAM
    from explainability.explain import explain
    from preprocessing.preprocessing import preprocess_face
    from utils.image_utils import overlay_heatmap, save_image, side_by_side

    cam = GradCAM(model)
    fake_idx = class_names.index(config.CLASS_FAKE)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in config.CLASS_FOLDERS:
        files = sorted(p for p in (ds_dir / cls).glob("*") if p.suffix.lower() in config.IMAGE_EXTENSIONS)[:n_per_class]
        for p in files:
            img = cv2.imread(str(p))
            x = preprocess_face(img)
            probs = model.predict(x[None], verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            heat, _ = cam.compute(x, pred_idx)
            exp = explain(heat, class_names[pred_idx], float(probs[pred_idx]), None, img.shape[0])
            panel = side_by_side(cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE)),
                                 overlay_heatmap(cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE)), heat))
            label = f"true={cls} pred={class_names[pred_idx]} p_fake={probs[fake_idx]:.2f} top={exp.regions[0].label}"
            cv2.putText(panel, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            save_image(out_dir / f"{cls}_{p.stem}.jpg", panel)
    print(f"Grad-CAM samples written to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--threshold", type=float, default=config.THRESHOLD)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--gradcam-samples", type=int, default=0, help="save N Grad-CAM panels per class")
    args = ap.parse_args()

    if not args.model_path.exists():
        raise SystemExit(f"No model at {args.model_path}. Run train.py first.")
    with open(config.CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    fake_idx = class_names.index(config.CLASS_FAKE)

    print(f"Loading model {args.model_path} ...")
    model = keras.models.load_model(args.model_path)

    ds = load_split(args.split, args.batch_size, shuffle=False, augment_jpeg=False, dataset_dir=args.dataset)
    print(f"Evaluating on {args.dataset / args.split}: {ds.counts}")

    # Collect predictions and labels in the dataset's (deterministic) file order.
    probs = model.predict(ds, verbose=1)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in ds])
    y_true_fake = (y_true == fake_idx).astype(int)
    p_fake = probs[:, fake_idx]

    # File paths in the same order as the predictions (shuffle=False keeps Keras' listing order).
    file_paths = list(getattr(ds, "file_paths", []) or [])
    if len(file_paths) != len(p_fake):
        file_paths = []
        for cls in config.CLASS_FOLDERS:
            file_paths += sorted(str(p) for p in (args.dataset / args.split / cls).glob("*")
                                 if p.suffix.lower() in config.IMAGE_EXTENSIONS)

    report = {"model": str(args.model_path), "dataset": str(args.dataset), "split": args.split,
              "class_names": class_names, "frame_level": _metrics(y_true_fake, p_fake, args.threshold)}

    # ---- video level ------------------------------------------------------- #
    manifest = args.dataset / config.MANIFEST_PATH.name
    if manifest.exists() and len(file_paths) == len(p_fake):
        mf = pd.read_csv(manifest)
        mf = mf[mf["split"] == args.split]
        crop_to_video = {str((args.dataset / c).resolve()): v for c, v in zip(mf["crop"], mf["source_video"])}
        per_video_scores: dict[str, list[float]] = defaultdict(list)
        per_video_label: dict[str, int] = {}
        for path, p, y in zip(file_paths, p_fake, y_true_fake):
            vid = crop_to_video.get(str(Path(path).resolve()))
            if vid is None:
                continue
            per_video_scores[vid].append(float(p))
            per_video_label[vid] = int(y)
        if per_video_scores:
            vids = sorted(per_video_scores)
            v_true = np.array([per_video_label[v] for v in vids])
            v_p = np.array([np.mean(per_video_scores[v]) for v in vids])
            report["video_level"] = _metrics(v_true, v_p, args.threshold)
            report["video_level"]["frames_per_video_mean"] = float(np.mean([len(per_video_scores[v]) for v in vids]))
    else:
        print("No manifest for this dataset - skipping video-level metrics.")

    # ---- output ------------------------------------------------------------ #
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}" if args.dataset == config.DATASET_DIR else f"{args.dataset.name}_{args.split}"
    with open(args.out / f"eval_{tag}.json", "w") as f:
        json.dump(report, f, indent=2)
    _plot_confusion(np.array(report["frame_level"]["confusion_matrix"]["matrix"]),
                    args.out / f"confusion_{tag}_frames.png", "Frame-level")
    if "auc" in report["frame_level"]:
        _plot_roc(y_true_fake, p_fake, args.out / f"roc_{tag}.png", report["frame_level"]["auc"])
    if "video_level" in report:
        _plot_confusion(np.array(report["video_level"]["confusion_matrix"]["matrix"]),
                        args.out / f"confusion_{tag}_videos.png", "Video-level")

    def show(name: str, m: dict) -> None:
        line = f"  {name:<12} n={m['n']:<5} acc={m['accuracy']:.3f}"
        if "auc" in m:
            line += f"  auc={m['auc']:.3f}"
        line += f"  P(fake)={m['precision_fake']:.3f}  R(fake)={m['recall_fake']:.3f}  F1={m['f1_fake']:.3f}"
        print(line)

    print("\nResults")
    show("frame-level", report["frame_level"])
    if "video_level" in report:
        show("video-level", report["video_level"])
    print(f"\nReport + plots written to {args.out}/ (eval_{tag}.json)")

    if args.gradcam_samples > 0:
        _save_gradcam_samples(model, args.dataset / args.split, class_names, args.gradcam_samples,
                              args.out / f"gradcam_samples_{tag}")


if __name__ == "__main__":
    main()
