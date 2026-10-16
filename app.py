#!/usr/bin/env python3
"""
Flask demo: upload a video or image, get a verdict and an explanation.

    python app.py            # http://127.0.0.1:5000

Routes
    GET  /             upload page
    POST /analyze      form upload -> result page
    POST /api/analyze  form upload -> JSON (same fields as predict.py)
    GET  /health       {"model_loaded": bool}

The model is loaded lazily on the first request (so the page still opens and
explains what to do when no model has been trained yet). Uploads and result
images live under static/ so Flask can serve them, and anything older than
CLEANUP_AFTER_MINUTES is swept on every new upload - static/ is public to
anyone who can reach the server, which is fine for a localhost demo but is
the first thing to change before deploying this anywhere.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

import config

app = Flask(__name__, static_folder=str(config.STATIC_DIR), template_folder=str(config.BASE_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

_detector = None
_detector_error: str | None = None
_lock = threading.Lock()


def get_detector():
    """Load the model once, thread-safely. Returns None (and records why) if that fails."""
    global _detector, _detector_error
    if _detector is not None:
        return _detector
    with _lock:
        if _detector is None and _detector_error is None:
            try:
                from predict import DeepfakeDetector
                _detector = DeepfakeDetector()
            except Exception as exc:  # model missing, bad weights, ...
                _detector_error = f"{type(exc).__name__}: {exc}"
    return _detector


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in config.ALLOWED_UPLOAD_EXTENSIONS


def _cleanup(max_age_minutes: int = config.CLEANUP_AFTER_MINUTES) -> None:
    cutoff = time.time() - max_age_minutes * 60
    for root in (config.UPLOAD_DIR, config.WEB_RESULTS_DIR):
        root.mkdir(parents=True, exist_ok=True)
        for p in root.iterdir():
            if p.name.startswith("."):
                continue
            try:
                if p.stat().st_mtime < cutoff:
                    shutil.rmtree(p) if p.is_dir() else p.unlink()
            except OSError:
                pass


def _save_upload() -> tuple[Path, str]:
    f = request.files.get("media")
    if f is None or not f.filename:
        abort(400, "No file was uploaded.")
    if not _allowed(f.filename):
        abort(400, f"Unsupported file type. Allowed: {', '.join(sorted(config.ALLOWED_UPLOAD_EXTENSIONS))}")
    job_id = uuid.uuid4().hex[:12]
    ext = Path(secure_filename(f.filename)).suffix.lower()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = config.UPLOAD_DIR / f"{job_id}{ext}"
    f.save(path)
    return path, job_id


def _run(path: Path, job_id: str) -> dict:
    det = get_detector()
    if det is None:
        abort(503, f"Model is not available. {_detector_error or ''} Train one with train.py.")
    frames = request.form.get("frames", type=int) or config.FRAMES_PER_VIDEO
    frames = max(1, min(frames, 64))
    out_dir = config.WEB_RESULTS_DIR / job_id
    result = det.analyze(path, frames_per_video=frames, out_dir=out_dir)
    result["job_id"] = job_id
    result["original_name"] = secure_filename(request.files["media"].filename)
    result["upload_url"] = url_for("static", filename=f"uploads/{path.name}")
    if "images" in result:
        result["image_urls"] = {
            k: url_for("static", filename=f"results/{job_id}/{Path(v).name}") for k, v in result["images"].items()
        }
    return result


@app.route("/")
def index():
    det_ready = config.MODEL_PATH.exists()
    return render_template("index.html", model_ready=det_ready, model_path=config.MODEL_PATH.name,
                           max_mb=config.MAX_UPLOAD_MB, default_frames=config.FRAMES_PER_VIDEO,
                           exts=sorted(config.ALLOWED_UPLOAD_EXTENSIONS))


@app.route("/analyze", methods=["POST"])
def analyze():
    _cleanup()
    path, job_id = _save_upload()
    result = _run(path, job_id)
    return render_template("result.html", r=result, threshold=config.THRESHOLD)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    _cleanup()
    path, job_id = _save_upload()
    result = _run(path, job_id)
    result.pop("images", None)  # local paths are not useful to API clients
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"model_loaded": _detector is not None, "model_file_exists": config.MODEL_PATH.exists(),
                    "error": _detector_error})


@app.errorhandler(400)
@app.errorhandler(413)
@app.errorhandler(503)
def handle_error(err):
    message = getattr(err, "description", str(err))
    if err.code == 413:
        message = f"File too large. Maximum upload size is {config.MAX_UPLOAD_MB} MB."
    if request.path.startswith("/api/"):
        return jsonify({"error": message}), err.code
    return render_template("index.html", error=message, model_ready=config.MODEL_PATH.exists(),
                           model_path=config.MODEL_PATH.name, max_mb=config.MAX_UPLOAD_MB,
                           default_frames=config.FRAMES_PER_VIDEO,
                           exts=sorted(config.ALLOWED_UPLOAD_EXTENSIONS)), err.code


if __name__ == "__main__":
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    config.WEB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if config.MODEL_PATH.exists():
        print("Loading model ...")
        get_detector()
        print("Model ready." if _detector else f"Model failed to load: {_detector_error}")
    else:
        print(f"No model at {config.MODEL_PATH} - the page will open but analysis is disabled until you train.")
    print(f"Open http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False, threaded=True)
