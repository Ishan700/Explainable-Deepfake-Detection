"""
Central configuration for the Explainable Deepfake Detection project.

Every tunable value lives here so that train.py, predict.py, evaluate.py and
app.py all agree on image size, class order, thresholds and file locations.
Nothing else in the project should hard-code these numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR = BASE_DIR / "data_raw"          # where you drop the source videos
DATASET_DIR = BASE_DIR / "dataset"            # face crops produced by prepare_dataset.py
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"            # evaluation reports, plots, CLI predictions

MODEL_PATH = MODEL_DIR / "deepfake_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
TRAINING_LOG_PATH = MODEL_DIR / "training_log.csv"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"

STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
WEB_RESULTS_DIR = STATIC_DIR / "results"

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
# Folder names inside dataset/<split>/. Keras sorts them alphabetically, which
# gives fake -> 0 and real -> 1. We never rely on that ordering directly:
# class_names.json is written from the dataset object at training time and read
# back at prediction time.
CLASS_FAKE = "fake"
CLASS_REAL = "real"
CLASS_FOLDERS = (CLASS_FAKE, CLASS_REAL)

# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
IMG_SIZE = 224                 # model input is IMG_SIZE x IMG_SIZE x 3
FRAMES_PER_VIDEO = 16          # uniformly sampled frames per video
FACE_MARGIN = 0.30             # extra context around the detected box (30% each side)
MIN_FACE_SIZE = 40             # ignore detections smaller than this (pixels, in the frame)
FACE_SCORE_THRESHOLD = 0.70    # detector confidence cut-off

# Split ratios used when prepare_dataset.py assigns *videos* to splits.
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 42

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Face detector: "yunet" (OpenCV DNN, default), "haar" (classic cascade, no
# landmarks) or "none" (inputs are already face crops - just centre-crop/resize).
FACE_DETECTOR = "yunet"
YUNET_MODEL_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
# One of: efficientnetb0, efficientnetb2, mobilenetv2, xception
ARCHITECTURE = os.environ.get("DFD_ARCHITECTURE", "efficientnetb0")
PRETRAINED = os.environ.get("DFD_PRETRAINED", "1") == "1"   # ImageNet weights
DROPOUT = 0.30

# Layer inside the backbone whose activations Grad-CAM uses. "auto" picks the
# backbone's last convolutional feature map, which is the usual choice.
GRADCAM_LAYER = {
    "efficientnetb0": "top_activation",
    "efficientnetb2": "top_activation",
    "mobilenetv2": "out_relu",
    "xception": "block14_sepconv2_act",
}

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
BATCH_SIZE = 32
EPOCHS = 15                    # phase 1: frozen backbone
FINE_TUNE_EPOCHS = 10          # phase 2: top layers unfrozen
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_LAYERS = 40          # how many backbone layers (from the top) to unfreeze
EARLY_STOPPING_PATIENCE = 5

# Augmentation that mimics real-world degradation (social-media recompression).
AUG_JPEG_QUALITY_RANGE = (40, 95)
AUG_JPEG_PROBABILITY = 0.5

# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
THRESHOLD = 0.50               # video is "fake" if mean P(fake) >= THRESHOLD
MIN_FACES_FOR_VERDICT = 3      # fewer detected faces than this -> "inconclusive"

# --------------------------------------------------------------------------- #
# Web app
# --------------------------------------------------------------------------- #
MAX_UPLOAD_MB = 100
ALLOWED_UPLOAD_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
CLEANUP_AFTER_MINUTES = 30     # uploads/results older than this are deleted
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
