"""Core logic: image pre-checks, Groq vision call, affected-area heatmap, PDF report."""
import base64
import io
import json
import os
import re
from datetime import datetime

import numpy as np
from fpdf import FPDF
from dotenv import load_dotenv
from groq import Groq
from PIL import Image, ImageFilter, ImageOps

load_dotenv()  # so GROQ_MODEL from .env is read before MODEL is chosen
OWNER = "Pramod Singh Gaira"
# Must be a VISION model. Text-only models (openai/gpt-oss-*, llama-3.3-*) reject images with
# "messages[0].content must be a string". Llama 4 Scout was shut down on 2026-07-17.
TEXT_ONLY_HINTS = ("gpt-oss", "llama-3.3", "llama-3.1", "kimi", "compound")
_env_model = os.getenv("GROQ_MODEL", "").strip()
if _env_model and any(h in _env_model for h in TEXT_ONLY_HINTS):
    _env_model = ""  # ignore a text-only model set by mistake
VISION_MODELS = [m for m in (_env_model, "qwen/qwen3.8-27b", "qwen/qwen3.6-27b") if m]
MODEL = VISION_MODELS[0]


# ----------------------------------------------------------------------------
# 1. PRE-CHECKS (run locally, BEFORE anything is sent to the AI)
# ----------------------------------------------------------------------------
def load_image(file) -> Image.Image:
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((1024, 1024))
    return img


def _rgb_to_hsv(arr):
    return np.asarray(Image.fromarray(arr).convert("HSV")).astype(np.float32)


def vegetation_mask(img: Image.Image) -> np.ndarray:
    """True where pixel looks like plant tissue: green, yellow-green, or yellow-brown
    (so diseased leaves still count) with some saturation."""
    hsv = _rgb_to_hsv(np.asarray(img))
    h = hsv[..., 0] * 360 / 255
    s = hsv[..., 1] / 255
    v = hsv[..., 2] / 255
    return (h >= 25) & (h <= 160) & (s > 0.18) & (v > 0.12)


def check_is_leaf(img: Image.Image):
    """Heuristic gate: enough plant-coloured pixels, and not a flat/uniform picture."""
    mask = vegetation_mask(img)
    ratio = float(mask.mean())
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    texture = float(gray.std())
    ok = ratio >= 0.20 and texture > 12
    if ok:
        return True, ratio, "Looks like plant foliage."
    if ratio < 0.20:
        return False, ratio, (
            f"Only {ratio*100:.0f}% of the image looks like leaf tissue. "
            "Please upload a clear photo of a single leaf."
        )
    return False, ratio, "The image is too flat/uniform. Please upload a real leaf photo."


def check_quality(img: Image.Image):
    """Lighting + sharpness + resolution. Returns (ok, metrics, issues[])."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    over = float((gray > 245).mean())    # blown-out pixels
    under = float((gray < 20).mean())    # crushed shadows
    edges = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    sharpness = float(edges.var())
    w, h = img.size

    issues = []
    if brightness < 70:
        issues.append("Too dark - move to daylight or turn on a light.")
    if brightness > 200 or over > 0.15:
        issues.append("Overexposed / glare - avoid direct sun or flash on the leaf.")
    if under > 0.35:
        issues.append("Heavy shadows - the leaf is partly hidden in shade.")
    if contrast < 25:
        issues.append("Low contrast - lighting looks flat or foggy.")
    if sharpness < 40:
        issues.append("Image looks blurry - hold steady and tap to focus.")
    if min(w, h) < 200:
        issues.append("Resolution is too low - move closer to the leaf.")

    metrics = {
        "brightness": round(brightness, 1), "contrast": round(contrast, 1),
        "overexposed_%": round(over * 100, 1), "sharpness": round(sharpness, 1),
        "size": f"{w}x{h}",
    }
    return len(issues) == 0, metrics, issues


# ----------------------------------------------------------------------------
# 2. GROQ VISION CALL
# ----------------------------------------------------------------------------
PROMPT = """You are an expert plant pathologist. Examine this leaf photo.
Return ONLY valid JSON (no markdown) with exactly this schema:
{
 "is_leaf": true/false,
 "plant": "common plant name, e.g. Strawberry",
 "condition": "disease/disorder name or 'Healthy'",
 "confidence": number 0-100 (your honest certainty),
 "severity": "None" | "Mild" | "Moderate" | "Severe",
 "overview": "2-3 sentences about the condition and its cause",
 "symptoms": ["3-6 short visible symptoms"],
 "treatment": ["3-6 practical treatment steps, include organic options"],
 "prevention": ["3-6 prevention tips"],
 "explanation": "3-5 sentences explaining which visible features in THIS image led to the diagnosis",
 "top_predictions": [{"label": "Plant - Condition", "confidence": number}, ... exactly 3, confidences sum to <= 100]
}
If the image is not a plant leaf set is_leaf=false and fill other fields with empty values.
Be honest: if unsure, lower the confidence. Do not invent chemicals; suggest generally accepted practices."""


def analyze(img: Image.Image, api_key: str) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode()

    client = Groq(api_key=api_key)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }]
    resp, last_err = None, None
    for model in VISION_MODELS:
        kwargs = dict(model=model, messages=messages, temperature=0.1, max_completion_tokens=3000)
        for use_json in (True, False):  # some models reject JSON mode together with images
            try:
                extra = {"response_format": {"type": "json_object"}} if use_json else {}
                resp = client.chat.completions.create(**kwargs, **extra)
                break
            except Exception as e:
                last_err = e
                if use_json and "response_format" in str(e):
                    continue
                break
        if resp is not None:
            break
        # model missing / not vision / no access -> try the next vision model
        if not any(k in str(last_err) for k in ("model_not_found", "must be a string", "decommissioned")):
            raise last_err
    if resp is None:
        raise RuntimeError(f"No Groq vision model worked. Last error: {last_err}")
    text = resp.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("AI returned an unreadable response. Try again.")
        return json.loads(m.group(0))


# ----------------------------------------------------------------------------
# 3. AFFECTED-AREA HEATMAP
# NOTE: Groq serves the model through an API, so real Grad-CAM (needs gradients
# from inside a CNN) is impossible. This map is computed locally from the image:
# it highlights tissue that is NOT healthy green (spots, scorch, chlorosis).
# ----------------------------------------------------------------------------
def affected_heatmap(img: Image.Image):
    arr = np.asarray(img)
    hsv = _rgb_to_hsv(arr)
    h = hsv[..., 0] * 360 / 255
    s = hsv[..., 1] / 255
    v = hsv[..., 2] / 255

    leaf = vegetation_mask(img) | ((h < 60) & (s > 0.25) & (v > 0.1))  # include brown/yellow lesions
    healthy_green = (h >= 75) & (h <= 150) & (s > 0.3) & (v > 0.2)
    dark_lesion = (v < 0.25)
    score = np.zeros(h.shape, dtype=np.float32)
    score[leaf & ~healthy_green] = 0.8
    score[leaf & dark_lesion] = 1.0
    score[~leaf & ~dark_lesion] = 0.0

    m = Image.fromarray((score * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(14))
    m = np.asarray(m, dtype=np.float32) / 255.0
    m = m / (m.max() + 1e-6)

    # simple jet-like colormap
    r = np.clip(1.5 - np.abs(4 * m - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * m - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * m - 1), 0, 1)
    cmap = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    alpha = (m[..., None] * 0.55)
    overlay = (arr * (1 - alpha) + cmap * alpha).astype(np.uint8)
    affected_pct = float(((score > 0.5) & leaf).sum() / max(leaf.sum(), 1) * 100)
    return Image.fromarray(overlay), affected_pct


# ----------------------------------------------------------------------------
# 4. PDF REPORT
# ----------------------------------------------------------------------------
def _t(x) -> str:
    """fpdf core fonts are latin-1 only."""
    return str(x).replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-") \
        .encode("latin-1", "replace").decode("latin-1")


def build_pdf(result: dict, img: Image.Image, heat: Image.Image) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(22, 101, 52)
    pdf.cell(0, 12, "Leaf Disease Report", ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100)
    pdf.cell(0, 6, _t(f"Generated {datetime.now():%d %b %Y, %H:%M}  |  Owner: {OWNER}"), ln=1)
    pdf.ln(3)

    def img_to_buf(i):
        b = io.BytesIO(); i.convert("RGB").save(b, "JPEG", quality=85); b.seek(0); return b

    y = pdf.get_y()
    pdf.image(img_to_buf(img), x=10, y=y, w=90)
    pdf.image(img_to_buf(heat), x=105, y=y, w=90)
    pdf.set_y(y + 70)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(95, 5, "Uploaded leaf")
    pdf.cell(95, 5, "Affected-area map (colour analysis)", ln=1)
    pdf.ln(2)

    pdf.set_text_color(0)
    for k, v in [("Plant", result.get("plant")), ("Condition", result.get("condition")),
                 ("Confidence", f"{float(result.get('confidence', 0)):.1f}%"),
                 ("Severity (typical)", result.get("severity"))]:
        pdf.set_font("Helvetica", "B", 11); pdf.cell(45, 7, _t(k))
        pdf.set_font("Helvetica", "", 11); pdf.cell(0, 7, _t(v), ln=1)

    def section(title, body):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13); pdf.set_text_color(22, 101, 52)
        pdf.cell(0, 8, title, ln=1)
        pdf.set_text_color(0); pdf.set_font("Helvetica", "", 10)
        if isinstance(body, list):
            for item in body:
                pdf.set_x(12); pdf.multi_cell(0, 5.5, _t(f"-  {item}"))
        else:
            pdf.multi_cell(0, 5.5, _t(body))

    section("Overview", result.get("overview", ""))
    section("Symptoms", result.get("symptoms", []))
    section("Treatment", result.get("treatment", []))
    section("Prevention", result.get("prevention", []))
    section("AI Explanation", result.get("explanation", ""))
    section("Top Predictions", [f"{p.get('label')} - {float(p.get('confidence', 0)):.1f}%"
                                for p in result.get("top_predictions", [])])
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8); pdf.set_text_color(120)
    pdf.multi_cell(0, 4.5, _t("Disclaimer: AI-generated estimate from a single photo. Confidence is the "
                              "model's self-reported certainty, not a lab result. Confirm with a local "
                              "agriculture expert before applying treatments."))
    return bytes(pdf.output())
