# 🤖 Bot Telegram via n8n — Klasifikasi Sampah

Panduan menjalankan klasifikasi sampah **lewat Telegram**: kirim foto ke bot → bot membalas
jenis sampah + Grad-CAM + rekomendasi pengolahan → bisa tanya-jawab AI lanjutan (khusus
topik sampah). Orkestrasi memakai **n8n**; model + LLM diakses lewat **REST API (`api.py`)**.

```
Telegram ──foto/teks──▶ n8n (Docker, token BotFather) ──HTTP──▶ FastAPI (native di Mac)
   ▲                          │  host.docker.internal:8800        /predict   /chat
   └──── hasil + Grad-CAM ─────┘                                  (model + OpenRouter)
            + chat AI
```

> **Kenapa API jalan native (bukan Docker)?** Di Mac Apple Silicon, TensorFlow x86 di dalam
> container kena emulasi instruksi AVX → proses mati (`Illegal instruction`). n8n di Docker
> aman karena tidak memuat TensorFlow; bagian yang butuh TF (`api.py`) dijalankan native.

---

## Prasyarat
- Sudah bisa menjalankan project ini (lihat README utama), `model/model_sampah.h5` ada.
- **Docker Desktop** aktif (untuk n8n).
- **API key OpenRouter** (lihat README → bagian Asisten AI).

---

## Langkah 1 — Buat bot di BotFather
1. Di Telegram, chat **@BotFather** → kirim `/newbot`.
2. Beri nama & username bot → BotFather memberi **token** (mis. `123456:ABC-DEF...`). Simpan.

## Langkah 2 — Jalankan REST API (native)
```bash
cd /Users/imadegautama/Documents/codes/uas-pembelajaran-mesin
source .venv/bin/activate
pip install -r requirements.txt          # sekali, untuk fastapi/uvicorn/python-multipart
OPENROUTER_API_KEY="sk-or-..." uvicorn api:app --host 0.0.0.0 --port 8800
```
Cek: `curl http://localhost:8800/health` → `{"status":"ok",...}`.

> **Port 8800?** Port 8000 sering dipakai server lain (mis. PHP/Laravel). Jika ingin port
> lain, ganti `--port` DAN URL di node n8n (lihat Langkah 4). API key boleh juga lewat
> `.streamlit/secrets.toml` (di-uncomment) — `api.py` membacanya sebagai fallback.

## Langkah 3 — Jalankan n8n (Docker) dengan tunnel
Telegram Trigger butuh webhook publik. Untuk dev, pakai tunnel bawaan n8n:
```bash
docker run -it --rm \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n start --tunnel
```
- Buka editor n8n di **http://localhost:5678**.
- n8n di dalam container menjangkau API di host lewat **`host.docker.internal:8800`**
  (sudah dipakai di workflow).
- Alternatif tunnel: jalankan tanpa `--tunnel`, lalu `ngrok http 5678` dan set
  `WEBHOOK_URL` ke URL ngrok (`-e WEBHOOK_URL=https://xxxx.ngrok-free.app`).

## Langkah 4 — Import workflow & aktifkan
1. n8n → menu **⋮ / Import from File** → pilih `n8n/waste-bot.workflow.json`.
2. **Buat kredensial Telegram:** salah satu node Telegram → field *Credential* → *Create New*
   → tempel **token BotFather**. Lalu set kredensial yang sama di SEMUA node Telegram
   (Telegram Trigger, Get File, Send Result, Send Gradcam, Send Reco, Send Reply, Send Welcome).
3. (Opsional) kalau port/URL API beda, ubah field **URL** di node `Predict`, `Auto Reco`,
   dan `Chat`.
4. Klik **Save** lalu toggle **Active** (kanan atas).
5. Buka bot di Telegram → kirim `/start` → kirim **foto sampah**.

---

## Cara pakai
- **Kirim foto** → bot membalas: (1) jenis + keyakinan, (2) gambar **Grad-CAM**,
  (3) **rekomendasi** pengolahan/daur ulang.
- **Balas dengan pertanyaan** (mis. *"berapa lama terurai?"*) → dijawab kontekstual.
  Bot **mengingat** jenis sampah terakhir + beberapa giliran percakapan (per pengguna).
- **Tanya di luar topik** (mis. minta kodingan) → bot **menolak sopan** (guardrail di `/chat`).
- Kirim **foto baru** → konteks ter-reset ke jenis sampah yang baru.

---

## Struktur workflow (untuk verifikasi/perbaikan manual)
Bila ada node yang tidak ter-import sempurna (beda versi n8n), konfigurasikan manual:

| Node | Tipe | Setelan kunci |
|------|------|---------------|
| **Telegram Trigger** | Telegram Trigger | Updates: `message` |
| **Is Photo?** | IF | `{{ $json.message.photo }}` **exists** (array) → true ke jalur foto |
| **Get File** | Telegram | Resource **File**, File ID `={{ $json.message.photo[$json.message.photo.length-1].file_id }}`, **Download = ON** (hasil binary di property `data`) |
| **Predict** | HTTP Request | POST `http://host.docker.internal:8800/predict`; Body **multipart-form-data**; parameter tipe **n8n Binary File**, name `file`, input field `data` |
| **Send Result** | Telegram | sendMessage; Chat ID `={{ $('Telegram Trigger').item.json.message.chat.id }}`; teks dari `$('Predict')` |
| **Gradcam To Binary** | Code | ubah `gradcam_b64` → binary property `gradcam` (lihat kode di node) |
| **Send Gradcam** | Telegram | sendPhoto; **Binary Data = ON**, property `gradcam` |
| **Auto Reco** | HTTP Request | POST `.../chat`; Body **JSON**; minta rekomendasi untuk label terdeteksi |
| **Send Reco** | Telegram | sendMessage; teks `={{ $('Auto Reco').item.json.reply }}` |
| **Save Context** | Code | simpan `{pred_class,label,confidence,history}` ke static data per `chatId` |
| **Build Messages** | Code | baca static data; susun `messages = history + [user]`; `/start` / belum ada konteks → `hasContext=false` |
| **Has Context?** | IF | `{{ $json.hasContext }}` **true** → Chat; **false** → Send Welcome |
| **Chat** | HTTP Request | POST `.../chat`; Body **JSON** dari `$('Build Messages')` |
| **Send Reply** | Telegram | sendMessage; teks `={{ $('Chat').item.json.reply }}` |
| **Update Context** | Code | tambahkan giliran user+assistant ke history (maks 10 pesan) |
| **Send Welcome** | Telegram | sendMessage; instruksi kirim foto |

> Memori percakapan disimpan di **workflow static data** (`$getWorkflowStaticData('global')`)
> keyed by `chat.id`. Cukup untuk satu instance n8n; riwayat dibatasi 10 pesan agar hemat token.

---

## Troubleshooting
- **Bot tidak merespons** → pastikan workflow **Active**, token benar, dan tunnel jalan
  (Telegram Trigger menampilkan URL webhook).
- **Node Predict/Chat error "ECONNREFUSED"** → API belum jalan / URL salah. Pastikan uvicorn
  hidup dan URL pakai `host.docker.internal` (bukan `localhost`) karena n8n di dalam Docker.
- **`/chat` 503 "OPENROUTER_API_KEY belum diset"** → jalankan uvicorn dengan env
  `OPENROUTER_API_KEY=...` (atau uncomment di `.streamlit/secrets.toml`).
- **Grad-CAM tidak terkirim** → cek `gradcam_b64` ada di respons `/predict` (`curl ... | jq`).
- **Model gratis 429 (rate-limit)** → ganti model di `src/llm.py` (`DEFAULT_MODEL`) atau kirim
  field `model` dari node HTTP. Lihat README → Asisten AI.
