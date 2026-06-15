"""
preprocessing.py
================
Modul pra-pemrosesan gambar yang DIPAKAI BERSAMA oleh:
  - notebook pelatihan : notebooks/train_model.ipynb
  - aplikasi Streamlit : app.py

KENAPA modul terpisah?
  Soal UAS (Tahap 2.3) mewajibkan "input pengguna melewati pra-pemrosesan yang
  SAMA seperti saat pelatihan". Dengan meletakkan logika preprocessing di SATU
  tempat lalu meng-import-nya di dua sisi (training & app), kita dijamin
  konsisten -- tidak ada risiko beda resize/normalisasi yang membuat prediksi
  app melenceng dari hasil training. Ini juga poin nilai "kualitas kode".

Catatan: modul ini sengaja hanya bergantung pada numpy + Pillow (TANPA
TensorFlow) supaya ringan dan mudah diuji.
"""

import numpy as np
from PIL import Image

# MobileNetV2 menerima gambar berukuran 224x224 piksel, 3 channel (RGB).
IMG_SIZE = (224, 224)


def preprocess_image(img):
    """Ubah SATU gambar menjadi array siap-prediksi untuk MobileNetV2.

    Parameters
    ----------
    img : PIL.Image.Image | str
        Objek gambar Pillow, atau path string menuju file gambar.

    Returns
    -------
    np.ndarray
        Array berbentuk (1, 224, 224, 3) bertipe float32, sudah dinormalisasi
        ke rentang [-1, 1]. Dimensi pertama (1) adalah "batch" -- model Keras
        selalu mengharapkan input berbentuk batch walau hanya 1 gambar.
    """
    # Jika yang diberikan adalah path file, buka dulu sebagai gambar.
    if isinstance(img, str):
        img = Image.open(img)

    # 1) Paksa jadi 3 channel RGB.
    #    Foto bisa saja grayscale (1 channel) atau PNG dengan alpha (4 channel);
    #    konversi ke RGB menyeragamkan bentuknya menjadi (H, W, 3).
    img = img.convert("RGB")

    # 2) Resize ke ukuran input model (224x224).
    img = img.resize(IMG_SIZE)

    # 3) Ubah ke array angka float.
    arr = np.asarray(img, dtype=np.float32)

    # 4) Normalisasi ala MobileNetV2 (mode 'tf'): petakan piksel [0, 255] -> [-1, 1].
    #    Rumus ini PERSIS sama dengan
    #    keras.applications.mobilenet_v2.preprocess_input, hanya ditulis manual
    #    agar transparan dan tidak menambah dependency berat.
    arr = arr / 127.5 - 1.0

    # 5) Tambahkan dimensi batch di depan: (224,224,3) -> (1,224,224,3).
    arr = np.expand_dims(arr, axis=0)

    return arr
