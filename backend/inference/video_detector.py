"""Sample frames from a video and aggregate deepfake scores."""
from __future__ import annotations

import math
import statistics
from pathlib import Path

import cv2
from PIL import Image

from inference.predict_image import predict_pil_image

MAX_FRAMES = 24
_LOGIT_EPS = 1e-12
_FACE_MIN_SIZE = 48
_FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)
# Heuristic bands to separate Deepfake vs AI-Edited from binary fake score.
_DEEPFAKE_MIN = 0.72
_EDITED_MIN = 0.52
_EDITED_MAX = 0.72
_EDITED_MAX_STD = 0.16


def _pool_frames_logit_avg(
    prob_reals: list[float], prob_fakes: list[float], weights: list[float] | None = None
) -> tuple[float, float]:
    """
    Combine per-frame (p_real, p_fake) by averaging log-odds log(p_fake/p_real),
    then mapping back to a single probability pair. This matches independent
    binary classifiers better than averaging probabilities directly.
    """
    logits: list[float] = []
    use_weights = weights is not None and len(weights) == len(prob_fakes)
    if use_weights:
        total_w = sum(max(w, _LOGIT_EPS) for w in weights)
    else:
        total_w = float(len(prob_fakes))

    for i, (pr, pf) in enumerate(zip(prob_reals, prob_fakes)):
        pr = max(pr, _LOGIT_EPS)
        pf = max(pf, _LOGIT_EPS)
        l = math.log(pf / pr)
        w = max(weights[i], _LOGIT_EPS) if use_weights else 1.0
        logits.append(l * w)
    mean_logit = sum(logits) / max(total_w, _LOGIT_EPS)
    prob_fake = 1.0 / (1.0 + math.exp(-mean_logit))
    prob_real = 1.0 - prob_fake
    return prob_real, prob_fake


def _center_crop(frame_bgr, ratio: float = 0.7) -> Image.Image:
    h, w = frame_bgr.shape[:2]
    ch = max(1, int(h * ratio))
    cw = max(1, int(w * ratio))
    y1 = max(0, (h - ch) // 2)
    x1 = max(0, (w - cw) // 2)
    crop = frame_bgr[y1 : y1 + ch, x1 : x1 + cw]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _largest_face_crop(frame_bgr) -> tuple[Image.Image, bool]:
    """Return largest detected face crop + whether a face was found."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(_FACE_MIN_SIZE, _FACE_MIN_SIZE),
    )
    h, w = frame_bgr.shape[:2]
    if len(faces) == 0:
        return _center_crop(frame_bgr), False

    x, y, fw, fh = max(faces, key=lambda v: v[2] * v[3])
    pad_w = int(fw * 0.2)
    pad_h = int(fh * 0.2)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + fw + pad_w)
    y2 = min(h, y + fh + pad_h)
    crop = frame_bgr[y1:y2, x1:x2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb), True


def _frame_count(cap: cv2.VideoCapture) -> int:
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n > 0:
        return n
    seen = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        seen += 1
    return seen


def _classify_video(
    pooled_fake: float,
    frame_spread: float,
    avg_frame_weight: float,
    edited_min: float = _EDITED_MIN,
) -> tuple[str, float]:
    """
    Convert pooled binary score into project labels:
      - Real
      - AI-Edited
      - Deepfake
    """
    if pooled_fake >= _DEEPFAKE_MIN:
        return "Deepfake", pooled_fake

    in_edited_band = edited_min <= pooled_fake < _EDITED_MAX
    sufficiently_consistent = frame_spread <= _EDITED_MAX_STD
    sufficiently_decisive = avg_frame_weight >= 0.06
    if in_edited_band and (sufficiently_consistent or sufficiently_decisive):
        return "AI-Edited", pooled_fake

    return "Real", 1.0 - pooled_fake


def _pool_frames_hybrid(prob_fakes: list[float], weights: list[float]) -> float:
    """
    Hybrid pooling:
    - weighted mean over all frames (stability)
    - mean of top-k fake scores (captures localized edits)
    """
    if not prob_fakes:
        return 0.0
    total_w = sum(max(w, _LOGIT_EPS) for w in weights)
    weighted_mean = (
        sum(p * max(w, _LOGIT_EPS) for p, w in zip(prob_fakes, weights))
        / max(total_w, _LOGIT_EPS)
    )
    k = max(1, int(round(len(prob_fakes) * 0.3)))
    top_mean = sum(sorted(prob_fakes, reverse=True)[:k]) / k
    # Lean more toward strong evidence so subtle edits are not drowned out.
    return 0.45 * weighted_mean + 0.55 * top_mean


def predict_video(video_path: str | Path) -> dict:
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "prob_real": 0.0,
            "prob_fake": 0.0,
            "frames_analyzed": 0,
            "error": "Could not open video file.",
        }

    total = _frame_count(cap)
    cap.release()

    if total < 1:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "prob_real": 0.0,
            "prob_fake": 0.0,
            "frames_analyzed": 0,
            "error": "Video has no readable frames.",
        }

    cap = cv2.VideoCapture(str(path))
    n_samples = min(MAX_FRAMES, total)
    if total == 1:
        indices = [0]
    else:
        step = (total - 1) / (n_samples - 1)
        indices = sorted({min(total - 1, int(round(i * step))) for i in range(n_samples)})

    prob_reals: list[float] = []
    prob_fakes: list[float] = []
    frame_weights: list[float] = []
    backend = None
    faces_found = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        face_crop, has_face = _largest_face_crop(frame)
        center_crop = _center_crop(frame, ratio=0.72)
        out_face = predict_pil_image(face_crop)
        out_center = predict_pil_image(center_crop)
        # Pick stronger fake evidence for this frame.
        out = out_face if out_face["prob_fake"] >= out_center["prob_fake"] else out_center

        prob_reals.append(out["prob_real"])
        prob_fakes.append(out["prob_fake"])
        # Give higher weight to decisive frames and lower weight to ambiguous ones.
        frame_weights.append(
            abs(out["prob_fake"] - out["prob_real"]) + (0.015 if has_face else 0.0) + 1e-3
        )
        if has_face:
            faces_found += 1
        backend = out.get("backend")

    cap.release()

    if not prob_fakes:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "prob_real": 0.0,
            "prob_fake": 0.0,
            "frames_analyzed": 0,
            "error": "No frames could be analyzed.",
        }

    pooled_real_logit, pooled_fake_logit = _pool_frames_logit_avg(
        prob_reals, prob_fakes, frame_weights
    )
    pooled_fake_hybrid = _pool_frames_hybrid(prob_fakes, frame_weights)
    pooled_fake = 0.45 * pooled_fake_logit + 0.55 * pooled_fake_hybrid
    pooled_real = 1.0 - pooled_fake
    frame_spread = (
        statistics.pstdev(prob_fakes) if len(prob_fakes) > 1 else 0.0
    )
    avg_frame_weight = sum(frame_weights) / len(frame_weights)
    face_detection_ratio = faces_found / len(prob_fakes)

    # Adjust threshold slightly when face detections are sparse.
    edited_min = _EDITED_MIN - 0.03 if face_detection_ratio < 0.35 else _EDITED_MIN
    label, confidence = _classify_video(
        pooled_fake, frame_spread, avg_frame_weight, edited_min=edited_min
    )

    return {
        "label": label,
        "confidence": round(confidence, 6),
        "prob_real": round(pooled_real, 6),
        "prob_fake": round(pooled_fake, 6),
        "frames_analyzed": len(prob_fakes),
        "frame_prob_fake_std": round(frame_spread, 6),
        "avg_frame_weight": round(avg_frame_weight, 6),
        "face_detection_ratio": round(face_detection_ratio, 6),
        "backend": backend,
    }


def detect_video(video_path: str | Path) -> dict:
    """Alias for API compatibility."""
    return predict_video(video_path)
