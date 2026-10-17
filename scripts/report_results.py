#!/usr/bin/env python3
"""
Fill the README "Results" section from results/eval_test.json.

    python scripts/report_results.py
    python scripts/report_results.py --dataset-name "Celeb-DF v2 (890 videos per class)"

Reads results/eval_<split>.json (written by evaluate.py), writes a metrics table
between the <!-- RESULTS:START --> / <!-- RESULTS:END --> markers in README.md,
and copies the ROC / confusion / training-curve plots into docs/results/ so they
render on GitHub (results/ itself is git-ignored).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

PLOTS = {
    "training_curves.png": "Training curves (loss / accuracy / AUC per epoch)",
    "roc_{tag}.png": "ROC curve, frame level",
    "confusion_{tag}_videos.png": "Confusion matrix, video level",
}


def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def table(report: dict) -> str:
    rows = []
    for key, label in (("frame_level", "Frame"), ("video_level", "Video")):
        m = report.get(key)
        if not m:
            continue
        rows.append(f"| {label} | {m['n']} | {fmt(m['accuracy'])} | {fmt(m.get('auc'))} | "
                    f"{fmt(m['precision_fake'])} | {fmt(m['recall_fake'])} | {fmt(m['f1_fake'])} |")
    return ("| Level | n | Accuracy | AUC | Precision (fake) | Recall (fake) | F1 (fake) |\n"
            "|---|---|---|---|---|---|---|\n" + "\n".join(rows))


def confusion_text(m: dict) -> str:
    cm = m["confusion_matrix"]["matrix"]  # rows: actual real, fake; cols: predicted real, fake
    return (f"real→real {cm[0][0]}, real→fake {cm[0][1]}, fake→real {cm[1][0]}, fake→fake {cm[1][1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--results", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--readme", type=Path, default=config.BASE_DIR / "README.md")
    ap.add_argument("--dataset-name", default="Celeb-DF v2 (Kaggle mirror)")
    ap.add_argument("--note", default="", help="one extra sentence to add under the table")
    args = ap.parse_args()

    tag = args.split
    report_path = args.results / f"eval_{tag}.json"
    if not report_path.exists():
        sys.exit(f"{report_path} not found. Run evaluate.py first (or copy results/ from Colab).")
    report = json.loads(report_path.read_text())

    # ---- copy plots that exist ------------------------------------------- #
    docs_dir = config.BASE_DIR / "docs" / "results"
    docs_dir.mkdir(parents=True, exist_ok=True)
    images: list[tuple[str, str]] = []
    for pattern, caption in PLOTS.items():
        name = pattern.format(tag=tag)
        src = args.results / name
        if src.exists():
            shutil.copy2(src, docs_dir / name)
            images.append((caption, f"docs/results/{name}"))

    # ---- build the section ------------------------------------------------ #
    fl, vl = report.get("frame_level", {}), report.get("video_level")
    arch = config.ARCHITECTURE
    lines = [
        START,
        f"Trained on **{args.dataset_name}** with `{arch}` (ImageNet init), split by video, "
        f"evaluated on the held-out `{tag}` split. Frames per video: "
        f"{int(vl['frames_per_video_mean']) if vl and 'frames_per_video_mean' in vl else config.FRAMES_PER_VIDEO}. "
        f"Threshold {fl.get('threshold', config.THRESHOLD)}. Updated {date.today().isoformat()}.",
        "",
        table(report),
        "",
    ]
    if vl:
        lines.append(f"Video-level confusion: {confusion_text(vl)}.")
    if args.note:
        lines.append(args.note)
    lines.append("")
    lines.append("AUC is the headline number: it does not depend on the 0.5 threshold, and the classes "
                 "are not always balanced. The in-dataset score above is optimistic by construction; "
                 "the honest check is a **cross-dataset** run (`python evaluate.py --dataset <other>`), "
                 "which is usually 10–30 points lower.")
    if images:
        lines.append("")
        for caption, rel in images:
            lines.append(f"**{caption}**  ")
            lines.append(f"![{caption}]({rel})")
            lines.append("")
    lines.append(END)
    section = "\n".join(lines)

    readme = args.readme.read_text(encoding="utf-8")
    if START in readme and END in readme:
        new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: section, readme, flags=re.S)
    else:
        # First run: replace everything under "## Results" up to the next "## "
        m = re.search(r"^## Results\s*\n(.*?)(?=^## )", readme, flags=re.S | re.M)
        if not m:
            sys.exit("README.md has no '## Results' section and no RESULTS markers; add one.")
        new = readme[: m.start(1)] + section + "\n\n" + readme[m.end(1):]
    args.readme.write_text(new, encoding="utf-8")

    print(f"README updated from {report_path}")
    for _, rel in images:
        print(f"  copied -> {rel}")
    print("\nNext: git add -A && git commit -m \"Add results\" && git push")


if __name__ == "__main__":
    main()
