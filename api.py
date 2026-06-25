"""
api.py — REST API (FastAPI) pembungkus model klasifikasi sampah + asisten AI
============================================================================
Tujuan: memberi interface HTTP agar orkestrator non-Python (mis. **n8n** untuk bot
Telegram) bisa memakai model TensorFlow + LLM yang sama dengan aplikasi Streamlit.

Endpoint:
  GET  /health   → cek hidup + daftar kelas.
  POST /predict  → multipart `file` (gambar) → jenis sampah + keyakinan + Grad-CAM (base64).
  POST /chat     → {pred_class,label,confidence,messages[]} → balasan asisten AI (guardrail).

Jalankan NATIVE (bukan Docker) di Apple Silicon — TensorFlow x86 di container kena
emulasi AVX → "Illegal instruction". Contoh:
  OPENROUTER_API_KEY="sk-or-..." uvicorn api:app --host 0.0.0.0 --port 8000

Reuse penuh: src/preprocessing.py, src/gradcam.py, src/llm.py, src/labels.py.
TIDAK meng-import app.py (agar tidak mengeksekusi Streamlit).
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image

from src.preprocessing import preprocess_image
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.llm import build_system_prompt, chat as llm_chat, get_api_key, DEFAULT_MODEL
from src.labels import RECYCLE_INFO

# ----------------------------------------------------------------------------
# Muat model & label SEKALI saat startup (import-time).
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model_sampah.h5"
CLASS_PATH = BASE_DIR / "model" / "class_names.json"

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model tidak ditemukan di {MODEL_PATH}. Latih dulu lewat notebooks/train_model.ipynb."
    )

import tensorflow as tf  # import berat → lakukan setelah cek file model.

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

app = FastAPI(title="Klasifikasi Sampah API", version="1.0")


# ----------------------------------------------------------------------------
# Skema request
# ----------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    pred_class: str
    label: str
    confidence: float = 0.0
    messages: List[ChatMessage]
    model: Optional[str] = None


# ----------------------------------------------------------------------------
# Endpoint
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "classes": class_names}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File kosong.")
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")

    arr = preprocess_image(pil)                 # (1, 224, 224, 3)
    probs = model.predict(arr, verbose=0)[0]    # vektor 6 kelas
    idx = int(np.argmax(probs))
    pred_class = class_names[idx]
    info = RECYCLE_INFO.get(pred_class, {"label": pred_class, "status": "", "tip": ""})

    # Grad-CAM → PNG base64 (best-effort; kegagalan tak boleh menggagalkan prediksi).
    gradcam_b64 = None
    try:
        heatmap = make_gradcam_heatmap(arr, model, pred_index=idx)
        overlay = overlay_heatmap(heatmap, pil)  # PIL.Image RGB
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        gradcam_b64 = None

    return {
        "pred_class": pred_class,
        "label": info["label"],
        "confidence": float(probs[idx]),
        "status": info["status"],
        "tip": info["tip"],
        "probs": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "gradcam_b64": gradcam_b64,
    }


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    # Env diutamakan (server dijalankan dgn OPENROUTER_API_KEY) → hindari warning Streamlit.
    api_key = os.environ.get("OPENROUTER_API_KEY") or get_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY belum diset di server API.")

    system = build_system_prompt(req.pred_class, req.label, req.confidence)
    messages = [{"role": "system", "content": system}] + [
        {"role": m.role, "content": m.content} for m in req.messages
    ]
    try:
        reply = llm_chat(messages, api_key, model=req.model or DEFAULT_MODEL)
    except RuntimeError as exc:
        # Jangan kembalikan teks kosong ke Telegram (memicu "Bad Request"). Kirim pesan
        # ramah sebagai reply agar bot tetap membalas sesuatu yang informatif.
        reply = f"⚠️ {exc}"
    return {"reply": reply}
