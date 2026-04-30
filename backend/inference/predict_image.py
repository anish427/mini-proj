
from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageEnhance, ImageOps

# ---------------------------------------------------------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy singletons for each model
_models: dict[str, dict] = {
    "siglip": {"processor": None, "model": None, "ready": False},
    "vit":    {"processor": None, "model": None, "ready": False},
    "sdxl":   {"processor": None, "model": None, "ready": False},
}

ROOT = Path(__file__).resolve().parent.parent

# Model IDs
SIGLIP_ID = os.environ.get("DEEPFAKE_SIGLIP", "prithivMLmods/Deepfake-Detect-Siglip2")
VIT_ID    = os.environ.get("DEEPFAKE_VIT",    "dima806/deepfake_vs_real_image_detection")
SDXL_ID   = os.environ.get("DEEPFAKE_SDXL",   "Organika/sdxl-detector")


# ===================================================================
#  GENERIC MODEL LOADER + PREDICTOR
# ===================================================================

def _load_model(key: str, model_id: str):
    """Load any HuggingFace image-classification model generically."""
    slot = _models[key]
    if slot["ready"]:
        return
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    slot["processor"] = AutoImageProcessor.from_pretrained(model_id)
    slot["model"] = AutoModelForImageClassification.from_pretrained(model_id)
    slot["model"].to(_device).eval()
    slot["ready"] = True


def _predict_one(key: str, model_id: str, image: Image.Image) -> dict:
    """Run a single model and return {"prob_real": ..., "prob_fake": ...}."""
    _load_model(key, model_id)
    slot = _models[key]
    proc, model = slot["processor"], slot["model"]

    inputs = proc(images=image, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits.double(), dim=1)[0]

    id2label = model.config.id2label
    pf, pr = 0.0, 0.0
    for i in range(probs.shape[0]):
        raw = str(id2label.get(i, f"LABEL_{i}")).lower()
        p = float(probs[i].item())
        if "fake" in raw or "artificial" in raw or "ai" in raw or "deepfake" in raw:
            pf += p
        else:
            pr += p
    s = pr + pf
    if s > 0:
        pr /= s; pf /= s
    return {"prob_real": pr, "prob_fake": pf}


def _predict_multi(key: str, model_id: str, image: Image.Image) -> dict:
    """Run model on original + mirror → average (reduces orientation bias)."""
    s1 = _predict_one(key, model_id, image)
    s2 = _predict_one(key, model_id, ImageOps.mirror(image))
    return {
        "prob_real": (s1["prob_real"] + s2["prob_real"]) / 2,
        "prob_fake": (s1["prob_fake"] + s2["prob_fake"]) / 2,
    }


# ===================================================================
#  FORENSIC CHECKS
# ===================================================================

# --- ELA ---
def _run_ela(image: Image.Image, quality: int = 90) -> dict:
    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf).convert("RGB")
        ela_np = np.array(ImageChops.difference(image, resaved), dtype=np.float32)
        ela_gray = np.mean(ela_np, axis=2)
        ela_mean = float(np.mean(ela_gray))
        ela_std  = float(np.std(ela_gray))
        norm_mean = min(ela_mean / 25.0, 1.0)
        cv = (ela_std / max(ela_mean, 1e-6)) if ela_mean > 0.5 else 1.0
        uniformity = max(0.0, 1.0 - min(cv / 1.5, 1.0))
        score = 0.6 * uniformity + 0.4 * norm_mean
        return {"ela_score": round(max(0, min(1, score)), 4),
                "ela_mean": round(ela_mean, 4), "ela_std": round(ela_std, 4)}
    except Exception:
        return {"ela_score": 0.5, "ela_mean": 0, "ela_std": 0}


# --- Metadata ---
_AI_KW = [
    "stable diffusion","midjourney","dall-e","dalle","novelai","comfyui",
    "automatic1111","diffusion","openai","runway","adobe firefly","firefly",
    "bing image creator","copilot","gemini","ideogram","flux","leonardo",
    "playground","craiyon","invoke","dream",
]

def _run_metadata(image: Image.Image) -> dict:
    findings, score = [], 0.0
    try:
        exif = image.getexif()
    except Exception:
        exif = {}
    if not exif:
        findings.append("No EXIF data found"); score += 0.35
    else:
        make = exif.get(271, ""); model_t = exif.get(272, "")
        software = exif.get(305, ""); dt = exif.get(36867, "")
        if not make and not model_t:
            findings.append("No camera make/model"); score += 0.25
        if not dt:
            findings.append("No original datetime"); score += 0.10
        sw = (software or "").lower()
        for kw in _AI_KW:
            if kw in sw:
                findings.append(f"AI software: '{software}'"); score += 0.50; break
        if software and not make:
            findings.append(f"Software '{software}' but no camera"); score += 0.10

    info_str = " ".join(str(v) for v in (image.info or {}).values()).lower()
    for kw in _AI_KW:
        if kw in info_str:
            findings.append(f"AI keyword '{kw}' in metadata"); score += 0.40; break

    w, h = image.size
    if w == h and w >= 512 and w % 64 == 0:
        findings.append(f"Suspicious dims {w}×{h}"); score += 0.10
    elif w % 64 == 0 and h % 64 == 0 and w >= 512 and h >= 512:
        findings.append(f"Dims {w}×{h} multiples of 64"); score += 0.05

    if not findings:
        findings.append("Metadata appears normal")
    return {"metadata_score": round(max(0, min(1, score)), 4),
            "metadata_findings": findings}


# --- Frequency ---
def _run_frequency(image: Image.Image) -> dict:
    try:
        gray = np.array(image.convert("L"), dtype=np.float32)
        h, w = gray.shape
        if max(h, w) > 512:
            sc = 512 / max(h, w)
            gray = np.array(Image.fromarray(gray.astype(np.uint8)).resize(
                (int(w * sc), int(h * sc)), Image.LANCZOS), dtype=np.float32)
            h, w = gray.shape
        f = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log1p(np.abs(f))
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        d = np.sqrt((X - cx)**2 + (Y - cy)**2)
        r = min(cy, cx)
        lo = float(np.mean(mag[d < r*0.25]))
        mi = float(np.mean(mag[(d >= r*0.25) & (d < r*0.6)]))
        hi = float(np.mean(mag[d >= r*0.6]))
        total = lo + mi + hi + 1e-10
        hf = hi / total
        if hf < 0.18:   s = 0.85
        elif hf < 0.22: s = 0.65
        elif hf < 0.27: s = 0.45
        elif hf < 0.32: s = 0.25
        else:            s = 0.10
        return {"frequency_score": round(s, 4), "hf_ratio": round(hf, 4)}
    except Exception:
        return {"frequency_score": 0.5, "hf_ratio": 0.0}


# --- Texture / Smoothness (catches GAN faces) ---
def _run_texture(image: Image.Image) -> dict:
    try:
        gray = np.array(image.convert("L"), dtype=np.float32)
        h, w = gray.shape
        if max(h, w) > 384:
            sc = 384 / max(h, w)
            gray = np.array(Image.fromarray(gray.astype(np.uint8)).resize(
                (int(w * sc), int(h * sc)), Image.LANCZOS), dtype=np.float32)

        # Laplacian variance (sharpness)
        lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))

        # Local variance (smoothness map)
        k = 7
        g64 = gray.astype(np.float64)
        lm = cv2.blur(g64, (k, k))
        lv = cv2.blur(g64**2, (k, k)) - lm**2
        lv = np.maximum(lv, 0)
        smooth_ratio = float(np.mean(lv < 15.0))

        # Gradient uniformity
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gm = np.sqrt(gx**2 + gy**2)
        grad_cv = float(np.std(gm) / max(np.mean(gm), 1e-6))

        # Scores
        if lap_var < 100:     ls = 0.90
        elif lap_var < 300:   ls = 0.70
        elif lap_var < 600:   ls = 0.50
        elif lap_var < 1200:  ls = 0.30
        else:                 ls = 0.10

        if smooth_ratio > 0.70:   ss = 0.85
        elif smooth_ratio > 0.55: ss = 0.65
        elif smooth_ratio > 0.40: ss = 0.45
        elif smooth_ratio > 0.25: ss = 0.25
        else:                     ss = 0.10

        if grad_cv < 0.8:   gs = 0.75
        elif grad_cv < 1.1: gs = 0.55
        elif grad_cv < 1.5: gs = 0.35
        else:               gs = 0.15

        score = 0.40 * ls + 0.35 * ss + 0.25 * gs
        return {"texture_score": round(max(0, min(1, score)), 4),
                "laplacian_var": round(lap_var, 2),
                "smooth_ratio": round(smooth_ratio, 4),
                "gradient_cv": round(grad_cv, 4)}
    except Exception:
        return {"texture_score": 0.5, "laplacian_var": 0, "smooth_ratio": 0, "gradient_cv": 0}


# ===================================================================
#  MAJORITY-VOTE ENSEMBLE
# ===================================================================
# Each neural model gets equal vote.  Forensics provide support.
_W_NEURAL   = 0.70   # Total weight for neural signal
_W_ELA      = 0.06
_W_METADATA = 0.08
_W_FREQ     = 0.08
_W_TEXTURE  = 0.08


def _compute_ensemble(
    siglip_fake: float,
    vit_fake: float,
    sdxl_fake: float,
    ela_score: float,
    meta_score: float,
    freq_score: float,
    texture_score: float,
) -> tuple[str, float, float, float]:
    """
    MAJORITY-VOTE ensemble:
      - If 2+ of 3 neural models say fake (>50%), use their average as the
        neural signal.  This is a STRONG fake indicator.
      - If only 1 model says fake, use a dampened signal — one model alone
        cannot override the verdict.  Forensics must support it.
      - If 0 models say fake, use the average (likely all low → real).
    """
    fakes = [siglip_fake, vit_fake, sdxl_fake]
    avg_fake = sum(fakes) / len(fakes)
    votes_fake = sum(1 for f in fakes if f > 0.50)

    if votes_fake >= 2:
        # MAJORITY says fake — strong signal
        # Use average of the models that voted fake, boosted
        fake_scores = [f for f in fakes if f > 0.50]
        neural_fake = sum(fake_scores) / len(fake_scores)
        # Boost slightly for agreement
        neural_fake = min(1.0, neural_fake + 0.05 * votes_fake)
    elif votes_fake == 1:
        # Only ONE model says fake — uncertain
        # Use a damped version: avg of all three (dilutes the outlier)
        neural_fake = avg_fake
        # But if the single model is very confident AND forensics
        # show suspicious indicators, boost the signal
        max_fake = max(fakes)
        forensic_avg = (ela_score + meta_score + freq_score + texture_score) / 4
        # Count how many forensic signals are elevated (>0.30)
        elevated = sum(1 for s in [ela_score, meta_score, freq_score, texture_score] if s > 0.30)

        if max_fake > 0.90 and forensic_avg > 0.30:
            # Very confident model + some forensic support
            neural_fake = max(neural_fake, 0.50 + 0.05 * elevated)
        elif max_fake > 0.95 and elevated >= 2:
            # Extremely confident model + 2+ elevated forensics
            neural_fake = max(neural_fake, 0.60)
    else:
        # All models say real
        neural_fake = avg_fake

    neural_fake = max(0.0, min(1.0, neural_fake))

    # --- Weighted ensemble ---
    ensemble_fake = (
        _W_NEURAL   * neural_fake
        + _W_ELA    * ela_score
        + _W_METADATA * meta_score
        + _W_FREQ   * freq_score
        + _W_TEXTURE * texture_score
    )

    ensemble_fake = max(0.0, min(1.0, ensemble_fake))
    ensemble_real = 1.0 - ensemble_fake

    label = "Fake" if ensemble_fake >= 0.50 else "Real"
    confidence = ensemble_fake if label == "Fake" else ensemble_real
    return label, confidence, ensemble_real, ensemble_fake


# ===================================================================
#  PUBLIC API
# ===================================================================

def predict_pil_image(image: Image.Image) -> dict:
    """Run all detection signals and return comprehensive result."""
    # Neural classifiers
    siglip = _predict_multi("siglip", SIGLIP_ID, image)
    vit    = _predict_multi("vit",    VIT_ID,    image)
    sdxl   = _predict_multi("sdxl",   SDXL_ID,   image)

    # Forensic checks
    ela     = _run_ela(image)
    meta    = _run_metadata(image)
    freq    = _run_frequency(image)
    texture = _run_texture(image)

    # Ensemble
    label, confidence, prob_real, prob_fake = _compute_ensemble(
        siglip["prob_fake"], vit["prob_fake"], sdxl["prob_fake"],
        ela["ela_score"], meta["metadata_score"],
        freq["frequency_score"], texture["texture_score"],
    )

    return {
        "label": label,
        "confidence": round(confidence, 6),
        "prob_real": round(prob_real, 6),
        "prob_fake": round(prob_fake, 6),
        "backend": "triple-model-ensemble",
        "signals": {
            "siglip": {
                "name": "SigLIP-2 (Diffusion)",
                "weight": _W_NEURAL,
                "prob_fake": round(siglip["prob_fake"], 4),
                "prob_real": round(siglip["prob_real"], 4),
            },
            "vit": {
                "name": "ViT (GAN Faces)",
                "weight": _W_NEURAL,
                "prob_fake": round(vit["prob_fake"], 4),
                "prob_real": round(vit["prob_real"], 4),
            },
            "sdxl": {
                "name": "Swin (SDXL/General)",
                "weight": _W_NEURAL,
                "prob_fake": round(sdxl["prob_fake"], 4),
                "prob_real": round(sdxl["prob_real"], 4),
            },
            "ela": {
                "name": "Error Level Analysis",
                "weight": _W_ELA,
                "score": ela["ela_score"],
                "mean": ela["ela_mean"], "std": ela["ela_std"],
            },
            "metadata": {
                "name": "Metadata Analysis",
                "weight": _W_METADATA,
                "score": meta["metadata_score"],
                "findings": meta["metadata_findings"],
            },
            "frequency": {
                "name": "Frequency Analysis",
                "weight": _W_FREQ,
                "score": freq["frequency_score"],
                "hf_ratio": freq["hf_ratio"],
            },
            "texture": {
                "name": "Texture Analysis",
                "weight": _W_TEXTURE,
                "score": texture["texture_score"],
                "laplacian_var": texture["laplacian_var"],
                "smooth_ratio": texture["smooth_ratio"],
                "gradient_cv": texture["gradient_cv"],
            },
        },
    }


def predict_image(img_path: str | Path) -> dict:
    path = Path(img_path)
    image = Image.open(path).convert("RGB")
    return predict_pil_image(image)
