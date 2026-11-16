from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import io
import json
import base64
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Import shared utils
from utils.config import REPO_ID, MODEL_FILENAME, LOCAL_MODEL_PATH, LOCAL_MODEL_META, BENCHMARK_REPORT
from utils.logger import logger
from utils.model_architecture import CUSTOM_OBJECTS
from utils.processing import prepare_ultrasound, to_model_input
from utils.gradcam import find_target_layer, make_gradcam_heatmap, save_and_display_gradcam
from utils.report_generator import generate_docx_report

# Create Router
router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

# Global model state
MODEL = None
# Calibration + operating points selected on the validation split (see training/evaluate.py).
MODEL_META = {"calibration": {"a": 1.0, "b": 0.0}, "thresholds": {"balanced": 0.5, "rule_out": 0.5}, "source": "unloaded"}


def load_model():
    """Loads the locally trained, calibrated model; falls back to the Hugging Face-hosted model."""
    global MODEL, MODEL_META
    if MODEL is not None:
        return
    try:
        if Path(LOCAL_MODEL_PATH).exists():
            logger.info(f"Loading local model from {LOCAL_MODEL_PATH}")
            MODEL = tf.keras.models.load_model(LOCAL_MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
            meta = json.loads(Path(LOCAL_MODEL_META).read_text())
            MODEL_META = {**meta, "source": "FibonacciNet (TN5000, calibrated)"}
        else:
            logger.warning("Local model not found; loading fallback model from Hugging Face (uncalibrated)")
            model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
            MODEL = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
            MODEL_META = {**MODEL_META, "source": "fallback model (uncalibrated)"}
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")


def get_image_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def calibrate(p, meta):
    """Platt recalibration sigmoid(a * logit(p) + b); older exports carry a temperature T (a = 1/T, b = 0)."""
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    cal = meta.get("calibration") or {"a": 1.0 / meta.get("temperature", 1.0), "b": 0.0}
    return float(1 / (1 + np.exp(-(cal["a"] * np.log(p / (1 - p)) + cal["b"]))))


def risk_band(prob):
    """
    Three-tier triage using validation-selected thresholds: below rule-out the model runs at
    >=90% sensitivity, above rule-in at >=90% specificity, and in between it defers to a specialist.
    """
    thr = MODEL_META["thresholds"]
    rule_in = thr.get("rule_in", thr["balanced"])
    if prob >= max(rule_in, thr["rule_out"]):
        return "high", "High suspicion", "Findings favour malignancy. Correlate with ACR TI-RADS features; FNA is typically considered per size criteria."
    if prob >= thr["rule_out"]:
        return "intermediate", "Indeterminate", "Above the rule-out threshold. Specialist review and TI-RADS scoring recommended before any decision."
    return "low", "Low suspicion", "Below the high-sensitivity rule-out threshold. Routine TI-RADS-guided follow-up as clinically indicated."


def run_inference(contents):
    """Shared by /analyze and /report so both always agree."""
    image = Image.open(io.BytesIO(contents))
    image.load()
    view = prepare_ultrasound(image)  # exactly what the model sees
    batch = to_model_input(view)[None, ...]

    raw = float(MODEL(batch, training=False).numpy()[0][0])
    prob = calibrate(raw, MODEL_META)
    band, band_label, guidance = risk_band(prob)
    is_malignant = band == "high"

    gradcam_img = None
    try:
        target = find_target_layer(MODEL)
        if target:
            heatmap = make_gradcam_heatmap(batch, MODEL, target)
            if heatmap is not None:
                gradcam_img = save_and_display_gradcam(view, heatmap)
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")

    return {
        "image": image,
        "view": Image.fromarray(view),
        "gradcam": gradcam_img,
        "raw_score": raw,
        "probability": prob,
        "is_malignant": bool(is_malignant),
        "band": band,
        "band_label": band_label,
        "guidance": guidance,
        "label": {"high": "Suspicious for malignancy", "intermediate": "Indeterminate",
                  "low": "Likely benign"}[band],
    }


async def _read_upload(file):
    contents = await file.read()
    try:
        Image.open(io.BytesIO(contents)).verify()
    except (UnidentifiedImageError, OSError):
        return None
    return contents


# --- Routes ---

@router.on_event("startup")
async def startup_event():
    load_model()


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@router.get("/benchmark", response_class=HTMLResponse)
async def benchmark_page(request: Request):
    return templates.TemplateResponse(request=request, name="benchmark.html", context={})


@router.get("/api/benchmark")
async def benchmark_data():
    if not Path(BENCHMARK_REPORT).exists():
        return JSONResponse(status_code=404, content={"error": "Benchmark results are not available yet."})
    return json.loads(Path(BENCHMARK_REPORT).read_text())


@router.get("/api/model")
async def model_info():
    return MODEL_META


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    logger.info(f"Analyze request received for file: {file.filename}")
    if MODEL is None:
        load_model()
        if MODEL is None:
            return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    contents = await _read_upload(file)
    if contents is None:
        return JSONResponse(status_code=400, content={"error": "Unreadable image. Upload a PNG or JPEG ultrasound still."})

    r = run_inference(contents)
    return {
        "label": r["label"],
        "score": r["raw_score"],
        "probability": r["probability"],
        "percent": r["probability"] * 100,
        "class_id": int(r["is_malignant"]),
        "is_malignant": r["is_malignant"],
        "risk_band": r["band"],
        "risk_label": r["band_label"],
        "guidance": r["guidance"],
        "thresholds": MODEL_META["thresholds"],
        "model_source": MODEL_META["source"],
        "original_image": get_image_base64(r["view"]),
        "gradcam_image": get_image_base64(r["gradcam"]) if r["gradcam"] else None,
    }


@router.post("/report")
async def get_report(file: UploadFile = File(...)):
    logger.info(f"Report request received for file: {file.filename}")
    try:
        if MODEL is None:
            load_model()
            if MODEL is None:
                return JSONResponse(status_code=503, content={"error": "Model not loaded"})

        contents = await _read_upload(file)
        if contents is None:
            return JSONResponse(status_code=400, content={"error": "Unreadable image."})
        r = run_inference(contents)

        def to_buffer(img):
            if img is None:
                return None
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return buf

        report_buffer = generate_docx_report(
            image_buffer=to_buffer(r["view"]),
            prediction_label=r["label"],
            confidence_score=r["probability"],
            confidence_percent=r["probability"] * 100,
            gradcam_buffer=to_buffer(r["gradcam"]),
            risk_label=r["band_label"],
            guidance=r["guidance"],
            model_meta=MODEL_META,
        )
        report_buffer.seek(0)

        headers = {'Content-Disposition': 'attachment; filename="thyroid_analysis_report.docx"'}
        return StreamingResponse(
            report_buffer,
            headers=headers,
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
