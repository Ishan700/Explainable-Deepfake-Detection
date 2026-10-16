#!/usr/bin/env python3
"""
Train the deepfake classifier on dataset/ produced by prepare_dataset.py.

    python train.py                      # defaults from config.py
    python train.py --epochs 10 --fine-tune-epochs 5 --batch-size 32
    python train.py --architecture xception --no-fine-tune

Training runs in two phases (standard transfer-learning recipe):
  phase 1 - backbone frozen, only the head learns          (LEARNING_RATE)
  phase 2 - top FINE_TUNE_LAYERS of the backbone unfrozen  (FINE_TUNE_LEARNING_RATE)

The best model by validation AUC is written to model/deepfake_model.keras and
model/class_names.json records the output order so predict.py never has to
guess which softmax index means "fake".
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras

import config
from model_builder import available_architectures, build_model, set_backbone_trainable

AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _count_images(split_dir: Path) -> dict[str, int]:
    counts = {}
    for cls in config.CLASS_FOLDERS:
        d = split_dir / cls
        counts[cls] = sum(1 for p in d.glob("*") if p.suffix.lower() in config.IMAGE_EXTENSIONS) if d.exists() else 0
    return counts


def _random_jpeg(image: tf.Tensor, label: tf.Tensor):
    """Re-encode a random subset of training images at a random JPEG quality.

    Social-media deepfakes arrive heavily recompressed; a model trained only on
    pristine crops learns compression fingerprints instead of forgery cues.
    """
    def apply():
        lo, hi = config.AUG_JPEG_QUALITY_RANGE
        x = tf.image.random_jpeg_quality(image / 255.0, lo, hi)
        return tf.clip_by_value(x * 255.0, 0.0, 255.0)

    image = tf.cond(tf.random.uniform([]) < config.AUG_JPEG_PROBABILITY, apply, lambda: image)
    return image, label


def load_split(split: str, batch_size: int, shuffle: bool, augment_jpeg: bool,
               dataset_dir: Path = config.DATASET_DIR) -> tf.data.Dataset:
    split_dir = dataset_dir / split
    counts = _count_images(split_dir)
    if min(counts.values()) == 0:
        raise SystemExit(f"{split_dir} is missing images for one class ({counts}). Run prepare_dataset.py first.")

    ds = keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=list(config.CLASS_FOLDERS),   # explicit: fake -> index 0, real -> index 1
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=None,                           # we batch after per-image augmentation
        shuffle=shuffle,
        seed=config.RANDOM_SEED,
        interpolation="area",
    )
    class_names = list(ds.class_names)
    file_paths = list(getattr(ds, "file_paths", []))
    if augment_jpeg:
        ds = ds.map(_random_jpeg, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    # Keep the metadata attributes after the transformations above.
    ds.class_names = class_names
    ds.file_paths = file_paths
    ds.counts = counts
    return ds


def class_weights(counts: dict[str, int]) -> dict[int, float]:
    """Inverse-frequency weights so an imbalanced split does not bias the head."""
    total = sum(counts.values())
    n = len(counts)
    return {i: total / (n * counts[cls]) for i, cls in enumerate(config.CLASS_FOLDERS)}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")],
    )


def make_callbacks(model_path: Path, log_path: Path, patience: int, append_log: bool,
                   initial_best: float | None = None) -> list[keras.callbacks.Callback]:
    ckpt_kwargs = dict(filepath=str(model_path), monitor="val_auc", mode="max", save_best_only=True, verbose=1)
    if initial_best is not None:
        ckpt_kwargs["initial_value_threshold"] = initial_best
    return [
        keras.callbacks.ModelCheckpoint(**ckpt_kwargs),
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, patience // 2),
                                          min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(str(log_path), append=append_log),
    ]


def plot_history(log_path: Path, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:  # pragma: no cover
        return
    if not log_path.exists():
        return
    df = pd.read_csv(log_path)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["loss", "accuracy", "auc"]):
        if metric in df:
            ax.plot(df[metric].values, label=f"train {metric}")
        if f"val_{metric}" in df:
            ax.plot(df[f"val_{metric}"].values, label=f"val {metric}")
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Training curves saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--architecture", default=config.ARCHITECTURE, choices=available_architectures())
    ap.add_argument("--epochs", type=int, default=config.EPOCHS, help="phase 1 epochs (frozen backbone)")
    ap.add_argument("--fine-tune-epochs", type=int, default=config.FINE_TUNE_EPOCHS, help="phase 2 epochs (0 to skip)")
    ap.add_argument("--no-fine-tune", action="store_true")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--fine-tune-lr", type=float, default=config.FINE_TUNE_LEARNING_RATE)
    ap.add_argument("--fine-tune-layers", type=int, default=config.FINE_TUNE_LAYERS)
    ap.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights")
    ap.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    ap.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    ap.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    args = ap.parse_args()

    keras.utils.set_random_seed(config.RANDOM_SEED)
    pretrained = config.PRETRAINED and not args.no_pretrained

    print("Loading data ...")
    train_ds = load_split("train", args.batch_size, shuffle=True, augment_jpeg=True, dataset_dir=args.dataset)
    val_ds = load_split("validation", args.batch_size, shuffle=False, augment_jpeg=False, dataset_dir=args.dataset)
    print(f"  train: {train_ds.counts}   validation: {val_ds.counts}")
    weights = class_weights(train_ds.counts)
    print(f"  class weights: { {config.CLASS_FOLDERS[i]: round(w, 3) for i, w in weights.items()} }")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(train_ds.class_names, f, indent=2)
    print(f"  class_names.json -> {train_ds.class_names}  (index 0 = {train_ds.class_names[0]})")

    print(f"\nBuilding {args.architecture} (pretrained={pretrained}) ...")
    model = build_model(architecture=args.architecture, pretrained=pretrained)
    if pretrained:
        set_backbone_trainable(model, False)
        phase1_desc = "training head on frozen ImageNet backbone"
    else:
        # Freezing random weights is pointless (and a random frozen backbone
        # produces near-constant features), so train everything from scratch.
        set_backbone_trainable(model, True, freeze_bn=False)
        phase1_desc = "training the whole network from scratch"
    compile_model(model, args.lr)
    model.summary(line_length=100)

    log_path = config.TRAINING_LOG_PATH
    if log_path.exists():
        log_path.unlink()

    # ---- phase 1 ------------------------------------------------------------ #
    print(f"\n=== Phase 1: {phase1_desc} ({args.epochs} epochs) ===")
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=weights,
                      callbacks=make_callbacks(args.model_path, log_path, args.patience, append_log=False))
    best_auc = float(np.max(hist1.history.get("val_auc", [0.0])))
    print(f"Phase 1 best val AUC: {best_auc:.4f}")

    # ---- phase 2: fine-tune top of backbone (pretrained only) ---------------- #
    if not pretrained and args.fine_tune_epochs > 0 and not args.no_fine_tune:
        print("Skipping phase 2: the whole network was already trainable in phase 1.")
    elif not args.no_fine_tune and args.fine_tune_epochs > 0:
        n = set_backbone_trainable(model, True, top_layers=args.fine_tune_layers)
        compile_model(model, args.fine_tune_lr)
        print(f"\n=== Phase 2: fine-tuning {n} backbone layers ({args.fine_tune_epochs} epochs) ===")
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, class_weight=weights,
                  callbacks=make_callbacks(args.model_path, log_path, args.patience, append_log=True,
                                           initial_best=best_auc))

    # ---- wrap up ------------------------------------------------------------ #
    if not args.model_path.exists():
        # EarlyStopping restored best weights; make sure something is on disk.
        model.save(args.model_path)
    best = keras.models.load_model(args.model_path)
    val_metrics = best.evaluate(val_ds, verbose=0, return_dict=True)
    print("\nBest checkpoint on validation:", {k: round(float(v), 4) for k, v in val_metrics.items()})
    print(f"Model saved to {args.model_path}")
    plot_history(log_path, config.RESULTS_DIR / "training_curves.png")
    print("\nNext: python evaluate.py        (test-set metrics)\n      python app.py             (web demo)")


if __name__ == "__main__":
    main()
