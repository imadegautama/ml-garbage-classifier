"""
src/labels.py — Metadata label kelas sampah (dipakai BERSAMA)
============================================================
`RECYCLE_INFO` semula didefinisikan inline di `app.py`, tetapi `app.py` tidak bisa
di-import (mengeksekusi Streamlit saat di-import). Modul kecil ini menampung dict
tersebut agar BISA dipakai bersama oleh:
  - `app.py`        (UI Streamlit)
  - `api.py`        (REST API untuk bot Telegram / integrasi lain)

Kunci dict = nama kelas bahasa Inggris (sesuai `model/class_names.json`).
"""

# Info edukatif daur ulang per kelas (label Indonesia + status + tips singkat).
RECYCLE_INFO = {
    "cardboard": {"label": "Kardus",  "status": "♻️ Dapat didaur ulang",          "tip": "Ratakan & jaga tetap kering."},
    "glass":     {"label": "Kaca",    "status": "♻️ Dapat didaur ulang",          "tip": "Bilas; hati-hati pecahan tajam."},
    "metal":     {"label": "Logam",   "status": "♻️ Dapat didaur ulang",          "tip": "Kaleng aluminium/baja bernilai tinggi."},
    "paper":     {"label": "Kertas",  "status": "♻️ Dapat didaur ulang",          "tip": "Jaga kering & bebas minyak."},
    "plastic":   {"label": "Plastik", "status": "♻️ Sebagian dapat didaur ulang", "tip": "Cek kode resin 1–7."},
    "trash":     {"label": "Residu",  "status": "🗑️ Sulit didaur ulang",          "tip": "Buang ke tempat sampah umum."},
}
