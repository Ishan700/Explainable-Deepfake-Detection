#!/usr/bin/env python3
"""
Copy (or move) videos from a downloaded dataset into data_raw/{real,fake}/.

Supported layouts:

  celebdf   Celeb-DF v2 as distributed (and as mirrored on Kaggle):
              Celeb-real/        -> real
              YouTube-real/      -> real
              Celeb-synthesis/   -> fake
            python scripts/import_dataset.py --source celebdf /path/to/Celeb-DF-v2 --limit 300

  dfdc      Deepfake Detection Challenge sample (train_sample_videos/ with metadata.json):
            python scripts/import_dataset.py --source dfdc /path/to/train_sample_videos --limit 200

  ffpp      FaceForensics++ as downloaded by its script:
              original_sequences/youtube/<c23>/videos/    -> real
              manipulated_sequences/<Method>/<c23>/videos -> fake
            python scripts/import_dataset.py --source ffpp /path/to/FaceForensics++ --limit 300

  folders   Any two folders you already sorted yourself:
            python scripts/import_dataset.py --source folders /path/real /path/fake

--limit N keeps at most N videos PER CLASS (random, seeded), which is the easy way to get a
balanced set quickly: Celeb-DF has 890 real vs 5,639 fake, DFDC ~80/20 fake-heavy.
Files are copied by default; pass --move to move them instead (saves disk space).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from utils.video_utils import iter_media_files  # noqa: E402


def _find_dir(root: Path, *names: str) -> Path | None:
    """Case-insensitive search for a sub-folder named like any of `names` (any depth)."""
    wanted = {n.lower() for n in names}
    if root.name.lower() in wanted:
        return root
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in wanted:
            return p
    return None


def collect_celebdf(root: Path) -> tuple[list[Path], list[Path]]:
    real_dirs = [d for d in (_find_dir(root, "Celeb-real"), _find_dir(root, "YouTube-real")) if d]
    fake_dir = _find_dir(root, "Celeb-synthesis")
    if not real_dirs or fake_dir is None:
        sys.exit(f"Could not find Celeb-real / YouTube-real / Celeb-synthesis under {root}")
    real = [p for d in real_dirs for p in iter_media_files(d)]
    fake = list(iter_media_files(fake_dir))
    return real, fake


def collect_dfdc(root: Path) -> tuple[list[Path], list[Path]]:
    metas = list(root.rglob("metadata.json"))
    if not metas:
        sys.exit(f"No metadata.json found under {root}")
    real, fake = [], []
    for meta in metas:
        with open(meta) as f:
            entries = json.load(f)
        for name, info in entries.items():
            path = meta.parent / name
            if not path.exists():
                continue
            (fake if str(info.get("label", "")).upper() == "FAKE" else real).append(path)
    return real, fake


def collect_ffpp(root: Path, compression: str) -> tuple[list[Path], list[Path]]:
    orig = _find_dir(root, "original_sequences")
    manip = _find_dir(root, "manipulated_sequences")
    if orig is None or manip is None:
        sys.exit(f"Could not find original_sequences / manipulated_sequences under {root}")

    def videos_under(base: Path) -> list[Path]:
        out = []
        for p in iter_media_files(base):
            parts = {q.lower() for q in p.parts}
            if compression.lower() in parts or compression == "any":
                out.append(p)
        return out

    real = videos_under(orig)
    fake = videos_under(manip)
    return real, fake


def collect_folders(real_dir: Path, fake_dir: Path) -> tuple[list[Path], list[Path]]:
    return list(iter_media_files(real_dir)), list(iter_media_files(fake_dir))


def place(files: list[Path], dest: Path, move: bool) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in files:
        # Keep the parent folder name in the file name so Celeb-real/id0_0000.mp4 and
        # YouTube-real/00000.mp4 (or FF++ method folders) never collide.
        target = dest / f"{src.parent.name}__{src.name}"
        if target.exists():
            continue
        (shutil.move if move else shutil.copy2)(str(src), str(target))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["celebdf", "dfdc", "ffpp", "folders"], required=True)
    ap.add_argument("paths", nargs="+", type=Path, help="dataset root (or REAL_DIR FAKE_DIR for --source folders)")
    ap.add_argument("--out", type=Path, default=config.DATA_RAW_DIR)
    ap.add_argument("--limit", type=int, default=None, help="max videos per class (random, seeded)")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    ap.add_argument("--compression", default="c23", help="ffpp only: c0 / c23 / c40 / any")
    args = ap.parse_args()

    if args.source == "celebdf":
        real, fake = collect_celebdf(args.paths[0])
    elif args.source == "dfdc":
        real, fake = collect_dfdc(args.paths[0])
    elif args.source == "ffpp":
        real, fake = collect_ffpp(args.paths[0], args.compression)
    else:
        if len(args.paths) != 2:
            sys.exit("--source folders needs two paths: REAL_DIR FAKE_DIR")
        real, fake = collect_folders(args.paths[0], args.paths[1])

    print(f"Found {len(real)} real and {len(fake)} fake videos")
    if not real or not fake:
        sys.exit("One class is empty - check the path you passed.")

    rng = random.Random(args.seed)
    if args.limit:
        real = sorted(rng.sample(real, min(args.limit, len(real))))
        fake = sorted(rng.sample(fake, min(args.limit, len(fake))))
        print(f"Keeping {len(real)} real and {len(fake)} fake (--limit {args.limit})")

    verb = "Moved" if args.move else "Copied"
    n_real = place(real, args.out / config.CLASS_REAL, args.move)
    n_fake = place(fake, args.out / config.CLASS_FAKE, args.move)
    print(f"{verb} {n_real} real -> {args.out / config.CLASS_REAL}")
    print(f"{verb} {n_fake} fake -> {args.out / config.CLASS_FAKE}")
    print("\nNext: python prepare_dataset.py")


if __name__ == "__main__":
    main()
