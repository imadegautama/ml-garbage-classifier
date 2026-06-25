# 🚀 Deploy ke Coolify (GitHub Actions → Docker Hub → Coolify)

Pipeline CI/CD: **push ke `main`** → GitHub Actions **build 2 image** (API + Streamlit) →
push ke **Docker Hub** → **trigger redeploy** 2 resource Coolify lewat webhook.

```
git push main ─▶ GitHub Actions ─┬─ build Dockerfile.api  ─▶ Docker Hub: uas-sampah-api
                                 └─ build Dockerfile       ─▶ Docker Hub: uas-sampah-streamlit
                                          │
                                          └─ curl webhook ─▶ Coolify pull image + redeploy
```

Target: 3 service di satu Coolify → **API** (8800), **Streamlit** (8501), **n8n** (5678).

> Coolify jalan di Linux **x86_64** → `tensorflow-cpu` jalan normal (bebas isu AVX Apple Silicon).

---

## Langkah 1 — Docker Hub
1. Punya akun Docker Hub.
2. **Account Settings → Security → New Access Token** → simpan (jadi `DOCKERHUB_TOKEN`).
3. (Repo akan dibuat otomatis saat push pertama: `uas-sampah-api` & `uas-sampah-streamlit`.)

## Langkah 2 — GitHub Secrets
Repo GitHub → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Isi |
|--------|-----|
| `DOCKERHUB_USERNAME` | username Docker Hub |
| `DOCKERHUB_TOKEN` | access token Docker Hub (Langkah 1) |
| `COOLIFY_TOKEN` | API token Coolify (Langkah 3) |
| `COOLIFY_WEBHOOK_API` | deploy-webhook resource **API** (Langkah 4) |
| `COOLIFY_WEBHOOK_STREAMLIT` | deploy-webhook resource **Streamlit** (Langkah 4) |

## Langkah 3 — Token Coolify
Coolify → **Keys & Tokens → API tokens → Create** → simpan (jadi `COOLIFY_TOKEN`).

## Langkah 4 — Buat resource di Coolify

### a) App API
- **+ New → Resource → Docker Image** → image: `<DOCKERHUB_USERNAME>/uas-sampah-api:latest`.
- **Ports Exposes:** `8800`.
- **Environment Variables:** `OPENROUTER_API_KEY=sk-or-...` (untuk endpoint `/chat`).
- Set **Domain** (mis. `https://api-sampah.domain.com`) → Coolify urus HTTPS otomatis.
- Buka tab **Webhooks / Deploy** → salin **Deploy Webhook URL** → simpan ke GitHub secret
  `COOLIFY_WEBHOOK_API`.

### b) App Streamlit
- **Docker Image** → `<DOCKERHUB_USERNAME>/uas-sampah-streamlit:latest`.
- **Ports Exposes:** `8501`.
- **Environment Variables:** `OPENROUTER_API_KEY=sk-or-...`.
- Set **Domain** (mis. `https://sampah.domain.com`).
- Salin **Deploy Webhook URL** → GitHub secret `COOLIFY_WEBHOOK_STREAMLIT`.
- Jika UI macet di balik proxy (jarang), tambahkan **Custom Start Command**:
  `streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false`

### c) n8n
- **+ New → Resource → Service → n8n** (one-click).
- Pasang **persistent volume** (agar workflow & kredensial tidak hilang saat redeploy).
- Env: `WEBHOOK_URL=https://<domain-n8n>` (wajib, agar webhook Telegram HTTPS).
- Set **Domain** untuk n8n.

## Langkah 5 — Deploy pertama
```bash
git add Dockerfile.api .github/workflows/deploy.yml docs/COOLIFY_DEPLOY.md
git commit -m "Tambah CI/CD deploy ke Coolify (API + Streamlit)"
git push origin main
```
- Tab **Actions** GitHub: job **build-and-push** (2 image) harus hijau → cek image muncul di
  Docker Hub → job **deploy** memicu Coolify menarik image & redeploy.

## Langkah 6 — Hubungkan bot Telegram ke API di Coolify
Di **n8n (Coolify)**: import `n8n/waste-bot.workflow.json`, set kredensial Telegram, lalu
**ubah URL** di node `Predict`, `Auto Reco`, `Chat`:
- dari `http://host.docker.internal:8800/...`
- menjadi **domain publik API**: `https://api-sampah.domain.com/predict` & `/chat`
  (alternatif lebih cepat/privat: hostname internal service API di jaringan Coolify).

Lalu **Activate** workflow → kirim foto ke bot.

---

## Verifikasi
```bash
curl https://<domain-api>/health          # → {"status":"ok",...}
curl -F file=@foto.jpg https://<domain-api>/predict   # → JSON + gradcam_b64
```
- Buka `https://<domain-streamlit>` → klasifikasi + chat jalan.
- Kirim foto ke bot Telegram → jenis + Grad-CAM + rekomendasi.

## Catatan
- **Jangan** uji image API ini di Mac (arm64): `tensorflow-cpu` hanya x86; via emulasi kena
  `Illegal instruction`. Verifikasi di Coolify (x86) / GitHub Actions.
- `OPENROUTER_API_KEY` di-set sebagai **env Coolify**, bukan di-bake ke image. Untuk balasan
  andal (bukan "model sibuk 429"), pakai **BYOK** — lihat README → Asisten AI.
- Tag image yang dipush: `latest` (main), `sha-<commit>`, nama branch, dan `vX.Y.Z` (saat tag).
- Update berikutnya: cukup `git push` → Actions build & trigger redeploy otomatis.
