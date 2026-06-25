# 🗑️ Klasifikasi Sampah — UAS Pembelajaran Mesin

Aplikasi machine learning untuk mengklasifikasikan **gambar sampah** ke dalam 6
kategori (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`) menggunakan
**transfer learning MobileNetV2**, lalu di-*deploy* sebagai aplikasi web
**Streamlit** yang menerima foto dari pengguna dan menampilkan prediksi + tingkat
keyakinan + **Grad-CAM** (area yang diperhatikan model).

> **UAS Pembelajaran Mesin — Primakara University, TA 2025/2026**
> Dosen: Ida Bagus Kresna Sudiatmika, S.Kom., M.T.
>
> **Kelompok:**
> - I Made Gautama — 2301020008
> - I Kadek Indra Satya Ananda — 2301020078

- 🔗 **Aplikasi (deploy):** <https://uas-ml-garbage-classifier.streamlit.app/>
- 💻 **Repository:** <https://github.com/imadegautama/uas-pembelajaran-mesin>

---

## ✨ Fitur
- Unggah gambar **atau** ambil foto langsung dari kamera.
- Prediksi kategori sampah + **tingkat keyakinan**.
- **Grad-CAM**: visualisasi area gambar yang paling memengaruhi keputusan model.
- Informasi edukatif daur ulang untuk tiap kategori.
- **🤖 Asisten AI (OpenRouter):** setelah sampah terdeteksi, otomatis muncul
  rekomendasi cara pengolahan/daur ulang, lalu bisa **tanya bebas** lewat chat.
- Preprocessing yang **identik** antara training dan aplikasi (lihat `src/preprocessing.py`).

## 🧱 Struktur Proyek
```
.
├── app.py                    # APLIKASI Streamlit (Tahap 2)
├── api.py                    # REST API (FastAPI) — dipakai bot Telegram via n8n
├── src/
│   ├── preprocessing.py      # preprocessing gambar — dipakai BERSAMA train & app
│   ├── gradcam.py            # util Grad-CAM
│   ├── labels.py             # metadata label kelas (dipakai app.py + api.py)
│   └── llm.py                # klien OpenRouter (asisten AI: rekomendasi + chat + guardrail)
├── n8n/
│   └── waste-bot.workflow.json  # workflow Telegram (import ke n8n)
├── docs/
│   └── TELEGRAM_N8N.md       # panduan bot Telegram (n8n + BotFather)
├── .streamlit/
│   └── secrets.toml.example  # template API key (salin → secrets.toml, lalu isi)
├── model/
│   ├── model_sampah.h5       # model terlatih (hasil notebook)
│   └── class_names.json      # urutan label kelas
├── notebooks/
│   └── train_model.ipynb     # PELATIHAN model (Tahap 1) — jalankan di Google Colab
├── data/README.md            # cara mengunduh dataset (data mentah tidak di-commit)
├── requirements.txt          # dependensi aplikasi
├── Dockerfile                # (opsional) deploy ke host container apa pun
└── README.md
```
Kode **pelatihan** (`notebooks/`) dan kode **aplikasi** (`app.py`) sengaja dipisah
sesuai ketentuan soal.

## 🧠 Dataset
- **Garbage Classification** (berbasis TrashNet, Stanford) — Kaggle:
  `asdasdasasdas/garbage-classification`.
- 6 kelas, ±2.500 gambar. Detail & cara unduh: [`data/README.md`](data/README.md).

## 🔬 Tahap 1 — Membangun Model (`notebooks/train_model.ipynb`)
Dijalankan di **Google Colab** (GPU gratis). Ringkasan:
1. **Persiapan data** — unduh, buang gambar rusak, split 70/15/15, augmentasi (flip/rotasi/zoom), normalisasi `x/127.5 − 1`.
2. **Optimasi #1 — Hyperparameter tuning** (Keras Tuner/Hyperband): cari `dropout`, `learning_rate`, unit dense terbaik.
3. **Pelatihan** — MobileNetV2 (ImageNet) + kepala dengan hyperparameter terbaik; base dibekukan.
4. **Evaluasi** — accuracy, **F1 macro**, classification report, confusion matrix.
5. **Optimasi #2 — Fine-tuning** 30 layer teratas (LR 1e-5); bandingkan sebelum/sesudah.
6. **Simpan** — `model_sampah.h5` + `class_names.json` (+ `model_sampah.tflite`).

### 📊 Hasil Evaluasi _(test set: 374 gambar)_
| Metrik | Baseline (base beku) | Setelah fine-tuning | Δ |
|---|---|---|---|
| Accuracy | 84,2% | **86,1%** | +1,9% |
| F1 macro | 80,5% | **84,0%** | +3,5% |

Kedua tahap optimasi terbukti menaikkan performa. Fine-tuning juga memperbaiki
kelas tersulit `trash` (F1 0,57 → 0,70). Kelas terbaik: `paper` & `cardboard`
(F1 ≈ 0,89). Kelas yang masih sering tertukar: `glass`/`metal`/`plastic` (bahan
mengkilap/transparan mirip secara visual).

## 🖥️ Tahap 2 — Aplikasi (`app.py`)
Antarmuka:
- **Input:** gambar `jpg/png` (unggah) atau foto kamera.
- **Proses:** `preprocess_image()` (resize 224×224 + normalisasi) — sama seperti training.
- **Output:** kategori prediksi, persentase keyakinan, bar probabilitas semua kelas, overlay Grad-CAM, info daur ulang, **rekomendasi & chat AI**.

## 🤖 Asisten AI (OpenRouter)
Setelah gambar diklasifikasikan, aplikasi otomatis meminta LLM (lewat
[OpenRouter](https://openrouter.ai)) memberi **rekomendasi cara pengolahan/daur ulang**
untuk jenis sampah tersebut. Pengguna lalu bisa **bertanya bebas** di kolom chat —
konteks jenis sampah yang terdeteksi otomatis disertakan. Logikanya ada di `src/llm.py`.

**Menyiapkan API key** (buat gratis di <https://openrouter.ai/keys>). Urutan prioritas
pembacaan key: input sidebar → `st.secrets` → environment variable.

- **Lokal (file secrets):**
  ```bash
  cp .streamlit/secrets.toml.example .streamlit/secrets.toml
  # lalu isi OPENROUTER_API_KEY = "sk-or-..."  (file ini sudah di-gitignore)
  ```
- **Saat demo cepat:** cukup tempel key di kolom **"OpenRouter API Key"** pada sidebar.
- **Streamlit Community Cloud:** menu **Settings → Secrets**, tambahkan
  `OPENROUTER_API_KEY = "sk-or-..."`.
- **Docker:** kirim sebagai env var → `docker run -p 8501:8501 -e OPENROUTER_API_KEY="sk-or-..." klasifikasi-sampah`.

> Model default `google/gemma-4-31b-it:free` (gratis). Bisa diganti ke model lain
> (mis. berbayar yang lebih kuat) lewat kolom **Model** di sidebar. Model `:free` kadang
> kena rate-limit (error 429) — cukup ganti ke model lain bila terjadi. Tanpa API key,
> fitur chat nonaktif tetapi klasifikasi gambar tetap berjalan normal.

## 🤖 Bot Telegram (n8n) — opsional
Selain web Streamlit, klasifikasi bisa diakses lewat **Telegram**: kirim foto → bot membalas
jenis sampah + Grad-CAM + rekomendasi → bisa tanya-jawab AI (khusus topik sampah). Model & LLM
diakses lewat REST API `api.py`; orkestrasi Telegram memakai **n8n** + bot dari **BotFather**.

```bash
# 1) Jalankan REST API (native — di Apple Silicon JANGAN via Docker karena isu TF/AVX)
OPENROUTER_API_KEY="sk-or-..." uvicorn api:app --host 0.0.0.0 --port 8800
# 2) Jalankan n8n (Docker) + tunnel, lalu import n8n/waste-bot.workflow.json
docker run -it --rm -p 5678:5678 -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n start --tunnel
```
Panduan lengkap (BotFather, import workflow, kredensial, troubleshooting):
[`docs/TELEGRAM_N8N.md`](docs/TELEGRAM_N8N.md).

## 🚀 Cara Menjalankan

### A. Lokal (virtual environment)
> Gunakan **Python 3.10–3.13** (TensorFlow belum mendukung 3.14).
```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Buka <http://localhost:8501>. Jika `model/model_sampah.h5` belum ada, app akan
menampilkan instruksi untuk melatih model lebih dulu.

> ℹ️ `requirements.txt` otomatis memilih paket TensorFlow sesuai OS lewat penanda
> platform: **`tensorflow`** di macOS, **`tensorflow-cpu`** di Linux/Windows
> (termasuk Streamlit Cloud). Jadi perintah di atas sama untuk semua OS.

### B. Docker (opsional, portabel)
```bash
docker build -t klasifikasi-sampah .
docker run -p 8501:8501 -e OPENROUTER_API_KEY="sk-or-..." klasifikasi-sampah
```
> `-e OPENROUTER_API_KEY=...` opsional — hanya untuk mengaktifkan asisten AI.

### C. Melatih ulang model
Buka `notebooks/train_model.ipynb` di Google Colab → *Runtime: GPU* → jalankan
semua sel → unduh `model_sampah.h5` & `class_names.json` → letakkan di `model/`.

## ☁️ Deployment
**Streamlit Community Cloud (disarankan):** push repo ke GitHub → buka
<https://share.streamlit.io> → pilih repo → main file `app.py` → set Python 3.12/3.13.
Alternatif container: Render / Railway / Cloud Run / Fly.io memakai `Dockerfile`.

**Coolify (CI/CD via GitHub Actions → Docker Hub):** push ke `main` otomatis mem-build 2 image
(API `Dockerfile.api` + Streamlit `Dockerfile`) → push ke Docker Hub → trigger redeploy Coolify.
Workflow: [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml). Panduan lengkap
(secrets, resource Coolify, n8n): [`docs/COOLIFY_DEPLOY.md`](docs/COOLIFY_DEPLOY.md).

## 🛠️ Teknologi
Python · TensorFlow/Keras (MobileNetV2) · scikit-learn · Streamlit · Pillow · NumPy · Matplotlib · OpenRouter (LLM) via `requests`

## 🗺️ Pemetaan ke Kriteria Penilaian
| Aspek (bobot) | Di mana |
|---|---|
| Pembangunan & evaluasi model (30%) | `notebooks/train_model.ipynb` |
| Optimasi model (10%) | **Hyperparameter tuning (Keras Tuner) + fine-tuning** (notebook) |
| Implementasi ke aplikasi (30%) | `app.py` + `src/` |
| Deployment & demonstrasi (15%) | Streamlit Cloud / Docker + video demo |
| Kualitas kode & dokumentasi (15%) | Struktur rapi, preprocessing bersama, README ini |
