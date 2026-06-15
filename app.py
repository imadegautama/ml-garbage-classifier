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

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Fungsi preprocessing BERSAMA → menjamin input app diproses sama seperti training.
from src.preprocessing import preprocess_image

# ----------------------------------------------------------------------------
# Konfigurasi & path
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model_sampah.h5"
CLASS_PATH = BASE_DIR / "model" / "class_names.json"

# Info edukatif daur ulang per kelas (sentuhan "berdampak" + memudahkan pembacaan hasil).
RECYCLE_INFO = {
    "cardboard": {"label": "Kardus",  "status": "♻️ Dapat didaur ulang",          "tip": "Ratakan & jaga tetap kering."},
    "glass":     {"label": "Kaca",    "status": "♻️ Dapat didaur ulang",          "tip": "Bilas; hati-hati pecahan tajam."},
    "metal":     {"label": "Logam",   "status": "♻️ Dapat didaur ulang",          "tip": "Kaleng aluminium/baja bernilai tinggi."},
    "paper":     {"label": "Kertas",  "status": "♻️ Dapat didaur ulang",          "tip": "Jaga kering & bebas minyak."},
    "plastic":   {"label": "Plastik", "status": "♻️ Sebagian dapat didaur ulang", "tip": "Cek kode resin 1–7."},
    "trash":     {"label": "Residu",  "status": "🗑️ Sulit didaur ulang",          "tip": "Buang ke tempat sampah umum."},
}

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


def predict(model, pil_image):
    """Preprocess 1 gambar lalu jalankan model. Return (indeks, probabilitas, array)."""
    arr = preprocess_image(pil_image)            # (1, 224, 224, 3)
    probs = model.predict(arr, verbose=0)[0]     # vektor probabilitas 6 kelas
    return int(np.argmax(probs)), probs, arr


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
model = load_model()

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

pil_image = Image.open(image_source).convert("RGB")

# Prediksi.
idx, probs, arr = predict(model, pil_image)
pred_class = class_names[idx]
confidence = float(probs[idx])
info = RECYCLE_INFO.get(pred_class, {"label": pred_class, "status": "", "tip": ""})

# Tampilkan gambar + Grad-CAM berdampingan.
col1, col2 = st.columns(2)
with col1:
    st.image(pil_image, caption="Gambar masukan", use_container_width=True)
with col2:
    try:
        from src.gradcam import make_gradcam_heatmap, overlay_heatmap
        heatmap = make_gradcam_heatmap(arr, model, pred_index=idx)
        st.image(overlay_heatmap(heatmap, pil_image),
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
