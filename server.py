"""Flask backend for the Leaf Disease Detector. Run: python server.py"""
import base64
import io
import os

from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image

import core

BASE = os.path.dirname(os.path.abspath(__file__))
# Works whether index.html/style.css/app.js are in a "static" folder or sit next to server.py.
STATIC_DIR = os.path.join(BASE, "static") if os.path.isdir(os.path.join(BASE, "static")) else BASE
ALLOWED_STATIC = (".html", ".css", ".js", ".png", ".jpg", ".svg", ".ico", ".webp")

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def data_url(img, fmt="JPEG", q=85):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, fmt, quality=q)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def from_data_url(s):
    return Image.open(io.BytesIO(base64.b64decode(s.split(",", 1)[1]))).convert("RGB")


def api_key():
    return os.getenv("GROQ_API_KEY") or request.headers.get("X-API-Key", "").strip()


def run_checks(file):
    """Local pre-checks. Nothing here touches the AI."""
    try:
        img = core.load_image(file)
    except Exception:
        return None, {"ok": False, "stage": "image", "message": "Could not read this file as an image."}
    is_leaf, ratio, msg = core.check_is_leaf(img)
    if not is_leaf:
        return img, {"ok": False, "stage": "leaf", "message": msg, "leaf_ratio": round(ratio * 100)}
    q_ok, metrics, issues = core.check_quality(img)
    if not q_ok:
        return img, {"ok": False, "stage": "quality", "message": "Lighting / image quality is not good enough.",
                     "issues": issues, "metrics": metrics}
    return img, {"ok": True, "stage": "passed", "message": "Leaf detected and lighting looks good.",
                 "metrics": metrics}


@app.get("/")
def index():
    if not os.path.isfile(os.path.join(STATIC_DIR, "index.html")):
        return (f"index.html not found. Looked in: {STATIC_DIR}. "
                "Put index.html, style.css and app.js in a 'static' folder next to server.py."), 404
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    if not name.lower().endswith(ALLOWED_STATIC):  # never expose .py / .env
        return "Not found", 404
    return send_from_directory(STATIC_DIR, name)


@app.get("/api/config")
def config():
    return jsonify(owner=core.OWNER, models=core.VISION_MODELS, has_key=bool(os.getenv("GROQ_API_KEY")))


@app.post("/api/check")
def check():
    f = request.files.get("image")
    if not f:
        return jsonify(ok=False, stage="image", message="No image received."), 400
    _, res = run_checks(f)
    return jsonify(res)


@app.post("/api/analyze")
def analyze():
    f = request.files.get("image")
    if not f:
        return jsonify(ok=False, stage="image", message="No image received."), 400
    img, checks = run_checks(f)
    if not checks["ok"]:
        return jsonify(checks), 422  # rejected locally, AI never called
    key = api_key()
    if not key:
        return jsonify(ok=False, stage="key", message="Groq API key is missing."), 401
    try:
        result = core.analyze(img, key)
    except Exception as e:
        return jsonify(ok=False, stage="ai", message=f"AI request failed: {e}"), 502
    if not result.get("is_leaf", True):
        return jsonify(ok=False, stage="ai", message="The AI could not identify a plant leaf in this image."), 422
    heat, pct = core.affected_heatmap(img)
    return jsonify(ok=True, result=result, original=data_url(img), heatmap=data_url(heat),
                   affected_pct=round(pct, 1))


@app.post("/api/report")
def report():
    d = request.get_json(force=True)
    try:
        pdf = core.build_pdf(d["result"], from_data_url(d["original"]), from_data_url(d["heatmap"]))
    except Exception as e:
        return jsonify(ok=False, message=f"Could not build PDF: {e}"), 400
    return send_file(io.BytesIO(pdf), mimetype="application/pdf", as_attachment=True,
                     download_name="leaf_disease_report.pdf")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=True)