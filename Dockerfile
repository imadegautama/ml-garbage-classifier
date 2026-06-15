# =============================================================================
# Dockerfile — aplikasi Streamlit "Klasifikasi Sampah"
# -----------------------------------------------------------------------------
# Image portabel agar app bisa di-deploy ke host container apa pun
# (Render, Railway, Google Cloud Run, Fly.io) ATAU dijalankan lokal tanpa pusing
# soal versi Python di mesin. Streamlit Community Cloud TIDAK memakai file ini
# (ia langsung baca requirements.txt), jadi Dockerfile murni sebagai opsi.
# =============================================================================

# Python 3.12 = kompatibel TensorFlow & ramping.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependency dulu (layer terpisah → cache build lebih cepat saat kode berubah).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa kode + model.
COPY . .

# Banyak host container memberi port lewat env $PORT; fallback 8501 untuk lokal.
EXPOSE 8501
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true
