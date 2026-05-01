"""Flask API and static frontend for DeepFake Tracker."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from inference.predict_image import predict_image
from inference.video_detector import predict_video

ROOT = Path(__file__).resolve().parent          # backend/
PROJECT_ROOT = ROOT.parent                       # mini proj/
FRONTEND = PROJECT_ROOT / "frontend"

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="/static")
CORS(app, resources={r"/api/*": {"origins": "*"}})  # allow any origin for API routes
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_UPLOAD_MB", "200")
) * 1024 * 1024


def _suffix(name: str) -> str:
    return Path(name).suffix.lower()


@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")

@app.route("/auth")
def auth_page():
    return send_from_directory(FRONTEND, "auth.html")

@app.route("/profile")
def profile_page():
    return send_from_directory(FRONTEND, "profile.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/detect-image")
def detect_image():
    if "file" not in request.files:
        return jsonify({"error": "No file field `file`."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400
    ext = _suffix(f.filename)
    if ext not in ALLOWED_IMAGE:
        return jsonify({"error": f"Unsupported image type: {ext}"}), 400

    safe = secure_filename(f.filename) or "upload"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        path = tmp.name
    try:
        f.save(path)
        result = predict_image(path)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        Path(path).unlink(missing_ok=True)


@app.post("/api/detect-video")
def detect_video_route():
    if "file" not in request.files:
        return jsonify({"error": "No file field `file`."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400
    ext = _suffix(f.filename)
    if ext not in ALLOWED_VIDEO:
        return jsonify({"error": f"Unsupported video type: {ext}"}), 400

    safe = secure_filename(f.filename) or "upload"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        path = tmp.name
    try:
        f.save(path)
        result = predict_video(path)
        if result.get("error"):
            return jsonify({"ok": False, **result}), 422
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
