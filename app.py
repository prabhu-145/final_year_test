from __future__ import annotations

import base64
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
SAVED_SKETCH_DIR = STATIC_DIR / "saved_sketches"
DATABASE_DIR = STATIC_DIR / "database"
PARTS_DIR = STATIC_DIR / "parts"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_NAME = "ArcFace"
TOP_K = 5

app = Flask(__name__)
app.secret_key = "forensic-sketch-secret"

for path in [UPLOAD_DIR, SAVED_SKETCH_DIR, DATABASE_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def list_parts() -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {}
    if not PARTS_DIR.exists():
        return categories

    for category_dir in sorted([p for p in PARTS_DIR.iterdir() if p.is_dir()]):
        files = []
        for image_file in sorted(category_dir.iterdir()):
            if image_file.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS:
                files.append(f"parts/{category_dir.name}/{image_file.name}")
        categories[category_dir.name] = files
    return categories


def preprocess_image(image_path: Path) -> np.ndarray | None:
    try:
        import cv2
    except Exception:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None
    resized = cv2.resize(image, (224, 224))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def compute_embedding(image_path: Path) -> np.ndarray | None:
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return None

    try:
        from deepface import DeepFace

        representation = DeepFace.represent(
            img_path=processed_image,
            model_name=MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=False,
        )
    except Exception:
        return None

    if not representation:
        return None

    return np.array(representation[0]["embedding"], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def build_database_embeddings() -> Dict[str, np.ndarray]:
    embeddings: Dict[str, np.ndarray] = {}
    for image_path in sorted(DATABASE_DIR.iterdir() if DATABASE_DIR.exists() else []):
        if image_path.suffix.lower().lstrip(".") not in ALLOWED_EXTENSIONS:
            continue
        embedding = compute_embedding(image_path)
        if embedding is not None:
            embeddings[image_path.name] = embedding
    return embeddings


DATABASE_EMBEDDINGS = build_database_embeddings()


@app.route("/")
def splash() -> str:
    return render_template("splash.html")


@app.route("/dashboard")
def dashboard() -> str:
    return render_template("dashboard.html")


@app.route("/api/parts")
def parts_api():
    return jsonify(list_parts())


@app.route("/upload")
def upload_page() -> str:
    return render_template("upload.html")


@app.route("/save_sketch", methods=["POST"])
def save_sketch():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    if not image_data or not image_data.startswith("data:image/png;base64,"):
        return jsonify({"error": "Invalid sketch data"}), 400

    encoded = image_data.split(",", 1)[1]
    try:
        decoded_image = base64.b64decode(encoded)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid base64 payload"}), 400

    filename = f"sketch_{uuid.uuid4().hex}.png"
    save_path = SAVED_SKETCH_DIR / filename
    save_path.write_bytes(decoded_image)

    return jsonify({"filename": filename, "url": url_for("static", filename=f"saved_sketches/{filename}")})


@app.route("/recognize", methods=["POST"])
def recognize():
    upload_file = request.files.get("sketch")

    if upload_file is None or upload_file.filename == "":
        flash("Please upload a sketch image.", "error")
        return redirect(url_for("upload_page"))

    if not allowed_file(upload_file.filename):
        flash("Only PNG, JPG, and JPEG files are allowed.", "error")
        return redirect(url_for("upload_page"))

    safe_name = secure_filename(upload_file.filename)
    filename = f"{uuid.uuid4().hex}_{safe_name}"
    upload_path = UPLOAD_DIR / filename
    upload_file.save(upload_path)

    query_embedding = compute_embedding(upload_path)
    if query_embedding is None:
        flash("No recognizable face features found in the uploaded sketch.", "error")
        return redirect(url_for("upload_page"))

    if not DATABASE_EMBEDDINGS:
        flash("Face database is empty. Add images to static/database.", "error")
        return redirect(url_for("upload_page"))

    scored_matches = []
    for db_filename, db_embedding in DATABASE_EMBEDDINGS.items():
        similarity = cosine_similarity(query_embedding, db_embedding)
        scored_matches.append(
            {
                "filename": db_filename,
                "similarity": similarity,
                "confidence": round(similarity * 100, 2),
                "url": url_for("static", filename=f"database/{db_filename}"),
            }
        )

    top_matches = sorted(scored_matches, key=lambda x: x["similarity"], reverse=True)[:TOP_K]

    session["result"] = {
        "query_url": url_for("static", filename=f"uploads/{filename}"),
        "matches": top_matches,
    }

    return redirect(url_for("result"))


@app.route("/result")
def result():
    payload = session.get("result")
    if not payload:
        return redirect(url_for("upload_page"))
    return render_template("result.html", query_url=payload["query_url"], matches=payload["matches"])


if __name__ == "__main__":
    app.run(debug=True)
