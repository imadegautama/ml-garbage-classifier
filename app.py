"""
app.py — Aplikasi Streamlit "Klasifikasi Sampah"
================================================
KODE APLIKASI (Tahap 2). Tugasnya HANYA: memuat model yang sudah dilatih lalu
menerima input gambar dari pengguna dan menampilkan prediksi. Kode PELATIHAN
ada terpisah di notebooks/train_model.ipynb (syarat soal: pisahkan train & app).

Alur:
  unggah/foto  ->  preprocess_image() (SAMA seperti training)  ->  model.predict
               ->  tampilkan kelas + keyakinan + Grad-CAM
"""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Fungsi preprocessing BERSAMA → menjamin input app diproses sama seperti training.
from src.preprocessing import preprocess_image
# Klien LLM (OpenRouter) untuk fitur asisten AI (rekomendasi + chat).
from src.llm import DEFAULT_MODEL, build_system_prompt, chat as llm_chat, get_api_key
# Metadata label kelas — dipakai bersama dengan api.py (bot Telegram).
from src.labels import RECYCLE_INFO

# ----------------------------------------------------------------------------
# Konfigurasi & path
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model_sampah.h5"
CLASS_PATH = BASE_DIR / "model" / "class_names.json"

# RECYCLE_INFO (info edukatif per kelas) dipindah ke src/labels.py agar dipakai bersama
# oleh app.py dan api.py (bot Telegram). Lihat impor di atas.

st.set_page_config(page_title="Klasifikasi Sampah", page_icon="🗑️", layout="centered")


# ----------------------------------------------------------------------------
# Pemuatan model & label (di-cache agar tidak dimuat ulang tiap interaksi → hemat RAM)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model…")
def load_model():
    # import di dalam fungsi: app tetap bisa memberi pesan ramah meski TF belum siap.
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def predict(image_bytes):
    """Preprocess 1 gambar lalu jalankan model. Return (indeks, probabilitas).

    Di-cache berdasarkan ISI gambar (bytes) supaya interaksi chat — yang memicu
    rerun penuh Streamlit tiap pesan — tidak menjalankan ulang model.predict.
    """
    model = load_model()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = preprocess_image(pil_image)            # (1, 224, 224, 3)
    probs = model.predict(arr, verbose=0)[0]     # vektor probabilitas 6 kelas
    return int(np.argmax(probs)), probs


@st.cache_data(show_spinner=False)
def gradcam_overlay(image_bytes, pred_index):
    """Hasilkan overlay Grad-CAM, di-cache per (gambar, kelas) agar chat tetap responsif."""
    from src.gradcam import make_gradcam_heatmap, overlay_heatmap
    model = load_model()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = preprocess_image(pil_image)
    heatmap = make_gradcam_heatmap(arr, model, pred_index=pred_index)
    return overlay_heatmap(heatmap, pil_image)


# ----------------------------------------------------------------------------
# Sidebar (informasi)
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Tentang aplikasi")
    st.write(
        "Mengklasifikasikan gambar sampah ke **6 kategori** dengan **MobileNetV2** "
        "(transfer learning). Dibuat untuk UAS Pembelajaran Mesin."
    )
    st.subheader("Kategori sampah")
    for key, v in RECYCLE_INFO.items():
        st.markdown(f"- **{v['label']}** (`{key}`) — {v['status']}")

    st.divider()
    st.subheader("🤖 Asisten AI (OpenRouter)")
    st.text_input(
        "OpenRouter API Key",
        type="password",
        key="openrouter_api_key",
        placeholder="sk-or-...",
        help="Dapatkan gratis di openrouter.ai/keys. Bisa juga diset via .streamlit/secrets.toml.",
    )
    st.text_input("Model", value=DEFAULT_MODEL, key="openrouter_model")
    if get_api_key():
        st.caption("✅ API key terdeteksi — fitur chat aktif.")
    else:
        st.caption("⚠️ Belum ada API key — fitur chat nonaktif.")
    st.caption("[Buat API key →](https://openrouter.ai/keys)")

    st.caption("Primakara University • 2026")


# ----------------------------------------------------------------------------
# Halaman utama
# ----------------------------------------------------------------------------
st.title("🗑️ Klasifikasi Sampah")
st.write(
    "Unggah foto sampah → model menebak kategorinya, lengkap dengan tingkat "
    "keyakinan dan **Grad-CAM** (area yang diperhatikan model)."
)

# Bila model belum ada: tampilkan instruksi, JANGAN crash.
if not MODEL_PATH.exists():
    st.warning(
        "⚠️ **Model belum tersedia** (`model/model_sampah.h5`).\n\n"
        "Latih model dulu lewat `notebooks/train_model.ipynb` (Google Colab), "
        "lalu letakkan `model_sampah.h5` & `class_names.json` di folder `model/`."
    )
    st.stop()

class_names = load_class_names()
load_model()  # pra-muat model lebih awal (di-cache) → spinner ramah saat pertama kali.

# Dua cara input: unggah file atau kamera.
tab_upload, tab_camera = st.tabs(["📁 Unggah gambar", "📷 Kamera"])
with tab_upload:
    uploaded = st.file_uploader("Pilih gambar (jpg/png)", type=["jpg", "jpeg", "png"])
with tab_camera:
    captured = st.camera_input("Ambil foto")

image_source = uploaded or captured
if image_source is None:
    st.info("Silakan unggah gambar atau ambil foto untuk memulai.")
    st.stop()

image_bytes = image_source.getvalue()
pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

# Prediksi.
idx, probs = predict(image_bytes)
pred_class = class_names[idx]
confidence = float(probs[idx])
info = RECYCLE_INFO.get(pred_class, {"label": pred_class, "status": "", "tip": ""})

# Tampilkan gambar + Grad-CAM berdampingan.
col1, col2 = st.columns(2)
with col1:
    st.image(pil_image, caption="Gambar masukan", use_container_width=True)
with col2:
    try:
        st.image(gradcam_overlay(image_bytes, idx),
                 caption="Grad-CAM (area yang diperhatikan)", use_container_width=True)
    except Exception as exc:  # Grad-CAM gagal tidak boleh mematikan app.
        st.caption(f"Grad-CAM tidak tersedia: {exc}")

# Hasil utama.
st.subheader(f"Prediksi: {info['label']} (`{pred_class}`)")
st.metric("Tingkat keyakinan", f"{confidence * 100:.1f}%")
st.write(f"{info['status']} — {info['tip']}")
if confidence < 0.5:
    st.warning("Keyakinan rendah — coba foto lebih jelas atau objek lebih fokus.")

# Probabilitas seluruh kelas, urut dari tertinggi.
st.subheader("Probabilitas tiap kelas")
for i in np.argsort(probs)[::-1]:
    name = class_names[i]
    label = RECYCLE_INFO.get(name, {}).get("label", name)
    st.write(f"**{label}** (`{name}`) — {probs[i] * 100:.1f}%")
    st.progress(float(probs[i]))


# ----------------------------------------------------------------------------
# Asisten AI (OpenRouter): auto-rekomendasi pengolahan + chat bebas lanjutan.
# ----------------------------------------------------------------------------
st.divider()
st.subheader("💬 Tanya asisten AI tentang sampah ini")

# Key unik per (gambar, prediksi): ganti gambar → percakapan otomatis di-reset.
det_key = hashlib.md5(image_bytes).hexdigest() + ":" + pred_class
if st.session_state.get("chat_det_key") != det_key:
    st.session_state.chat_det_key = det_key
    st.session_state.chat_history = []
    st.session_state.auto_reco_needed = True

api_key = get_api_key()
model_name = st.session_state.get("openrouter_model") or DEFAULT_MODEL

if not api_key:
    st.info(
        "🔑 Masukkan **OpenRouter API Key** di sidebar (atau set di "
        "`.streamlit/secrets.toml`) untuk mengaktifkan rekomendasi & chat AI. "
        "Key gratis bisa dibuat di [openrouter.ai/keys](https://openrouter.ai/keys)."
    )
else:
    system_prompt = build_system_prompt(pred_class, info["label"], confidence)

    # Auto-rekomendasi: dijalankan sekali tiap gambar baru terdeteksi.
    if st.session_state.get("auto_reco_needed"):
        with st.spinner("Menyusun rekomendasi pengolahan…"):
            try:
                first_user = (
                    f"Berikan rekomendasi cara pengolahan dan daur ulang untuk "
                    f"sampah jenis {info['label']} ini."
                )
                reply = llm_chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": first_user},
                    ],
                    api_key,
                    model=model_name,
                )
                st.session_state.chat_history = [{"role": "assistant", "content": reply}]
            except RuntimeError as exc:
                st.error(f"Gagal memuat rekomendasi: {exc}")
        st.session_state.auto_reco_needed = False

    # Tampilkan riwayat percakapan.
    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input pertanyaan lanjutan (chat bebas).
    user_prompt = st.chat_input("Tanya apa saja, mis. cara membuang yang benar…")
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Menjawab…"):
                try:
                    messages = [{"role": "system", "content": system_prompt}]
                    messages += st.session_state.chat_history
                    reply = llm_chat(messages, api_key, model=model_name)
                    st.markdown(reply)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": reply}
                    )
                except RuntimeError as exc:
                    st.error(f"Gagal mendapat jawaban: {exc}")
