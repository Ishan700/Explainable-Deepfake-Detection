#!/usr/bin/env python3
"""
Build dataset/{train,validation,test}/{real,fake}/ from raw videos.

    python prepare_dataset.py --real_dir data_raw/real --fake_dir data_raw/fake

For each video: sample FRAMES_PER_VIDEO frames uniformly -> detect the largest
face -> crop with margin -> resize to IMG_SIZE -> save as JPEG.

Two things this script gets right that are easy to get wrong:

1. Splitting happens at the VIDEO level, before any frames are extracted.
   Frames from one video are near-identical, so if the split were done on
   frames the test set would contain near-copies of training images and the
   accuracy would be meaningless. The resulting manifest.csv records which
   video every crop came from so you can prove the split is clean.

2. Crops are resized here with the same function used at inference
   (preprocessing.resize_square). Training then reads 224x224 JPEGs, so Keras'
   own resize is a no-op and the train and serve paths see identical pixels.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

import config
from preprocessing.face_detection import FaceDetector
from preprocessing.preprocessing import resize_square
from utils.video_utils import iter_media_files, sample_frames


def assign_splits(items: list[Path], ratios: dict[str, float], seed: int) -> dict[Path, str]:
    """Deterministically assign each video to a split, respecting the ratios."""
    items = sorted(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["validation"]))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    out: dict[Path, str] = {}
    for i, p in enumerate(items):
        if i < n_train:
            out[p] = "train"
        elif i < n_train + n_val:
            out[p] = "validation"
        else:
            out[p] = "test"
    return out


def process_video(path: Path, label: str, split: str, detector: FaceDetector,
                  frames_per_video: int, out_root: Path, writer: csv.writer,
                  jpeg_quality: int) -> tuple[int, int]:
    """Returns (frames_sampled, faces_saved)."""
    try:
        frames = sample_frames(path, frames_per_video)
    except Exception as exc:
        print(f"  ! skipping {path.name}: {exc}", file=sys.stderr)
        return 0, 0

    saved = 0
    out_dir = out_root / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame in frames:
        face = detector.largest_face(frame)
        if face is None:
            continue
        crop = detector.crop(frame, face)
        if crop.size == 0:
            continue
        crop = resize_square(crop, config.IMG_SIZE)
        name = f"{path.stem}_f{frame_idx:05d}.jpg"
        out_path = out_dir / name
        cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        writer.writerow([str(out_path.relative_to(out_root)), str(path), label, split,
                         frame_idx, f"{face.score:.3f}", face.landmarks is not None])
        saved += 1
    return len(frames), saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dir", type=Path, default=config.DATA_RAW_DIR / "real")
    ap.add_argument("--fake_dir", type=Path, default=config.DATA_RAW_DIR / "fake")
    ap.add_argument("--out", type=Path, default=config.DATASET_DIR)
    ap.add_argument("--frames", type=int, default=config.FRAMES_PER_VIDEO, help="frames sampled per video")
    ap.add_argument("--detector", choices=["yunet", "haar", "none"], default=config.FACE_DETECTOR)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--max-videos", type=int, default=None, help="per class, for quick experiments")
    ap.add_argument("--clean", action="store_true", help="delete the existing dataset/ folder first")
    args = ap.parse_args()

    sources = {config.CLASS_REAL: list(iter_media_files(args.real_dir)),
               config.CLASS_FAKE: list(iter_media_files(args.fake_dir))}
    for label, files in sources.items():
        if not files:
            sys.exit(f"No videos/images found for class '{label}'. Put them under "
                     f"{args.real_dir if label == config.CLASS_REAL else args.fake_dir}")
        if args.max_videos:
            rng = random.Random(args.seed)
            rng.shuffle(files)
            sources[label] = sorted(files[: args.max_videos])

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in config.SPLIT_RATIOS:
        for label in config.CLASS_FOLDERS:
            (args.out / split / label).mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(backend=args.detector)
    print(f"Face detector: {detector.backend} | frames/video: {args.frames} | seed: {args.seed}")

    manifest_path = args.out / config.MANIFEST_PATH.name
    totals: dict[tuple[str, str], Counter] = defaultdict(Counter)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["crop", "source_video", "label", "split", "frame_index", "face_score", "has_landmarks"])
        for label, files in sources.items():
            splits = assign_splits(files, config.SPLIT_RATIOS, args.seed)
            for path in tqdm(files, desc=f"{label:>4}", unit="video"):
                split = splits[path]
                sampled, saved = process_video(path, label, split, detector, args.frames,
                                               args.out, writer, args.jpeg_quality)
                totals[(split, label)]["videos"] += 1
                totals[(split, label)]["frames"] += sampled
                totals[(split, label)]["faces"] += saved

    print("\nSummary (videos / frames sampled / faces saved):")
    for split in config.SPLIT_RATIOS:
        for label in config.CLASS_FOLDERS:
            c = totals[(split, label)]
            print(f"  {split:<11}{label:<6}{c['videos']:>6}{c['frames']:>9}{c['faces']:>9}")
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
