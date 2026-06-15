"""
gradcam.py
==========
Grad-CAM = Gradient-weighted Class Activation Mapping.

IDE DASAR (penting untuk dijelaskan saat ujian):
  Saat model CNN memutuskan sebuah gambar adalah "plastik", sebenarnya ada
  bagian gambar tertentu yang paling memengaruhi keputusan itu. Grad-CAM
  menghitung seberapa besar pengaruh tiap area pada feature map konvolusi
  terakhir terhadap skor kelas yang diprediksi, lalu menggambarnya sebagai
  "peta panas" (merah = paling berpengaruh).

  Manfaatnya: membuktikan model "melihat" objek sampahnya, bukan menebak dari
  latar belakang -- ini bukti pemahaman, bukan asal jalan.

Fungsi publik:
  - make_gradcam_heatmap(img_array, model) -> heatmap (H', W') nilai 0..1
  - overlay_heatmap(heatmap, original_img) -> PIL.Image gabungan
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib


def _find_last_conv_layer(model):
    """Cari nama layer konvolusi terakhir secara otomatis.

    Ciri feature map konvolusi: output-nya 4 dimensi (batch, tinggi, lebar,
    channel). Kita telusuri layer dari belakang dan ambil yang pertama
    ber-output 4D.
    """
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except AttributeError:
            continue
    raise ValueError("Tidak menemukan layer konvolusi (output 4D) pada model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """Hitung heatmap Grad-CAM untuk satu gambar yang sudah dipreprocess.

    Parameters
    ----------
    img_array : np.ndarray berbentuk (1, 224, 224, 3) -- output preprocess_image().
    model : tf.keras.Model -- model terlatih (harus model fungsional agar layer
        konvolusinya bisa diakses; lihat catatan di notebook training).
    last_conv_layer_name : str, opsional -- nama layer conv terakhir. Bila None,
        dideteksi otomatis.
    pred_index : int, opsional -- indeks kelas yang ingin dijelaskan. Bila None,
        memakai kelas dengan probabilitas tertinggi.

    Returns
    -------
    np.ndarray 2D bernilai 0..1 (peta panas berukuran feature map, mis. 7x7).
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(model)

    # Model bantu yang mengeluarkan DUA hal sekaligus:
    #   (1) output feature map konvolusi terakhir, dan
    #   (2) prediksi akhir.
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    # Rekam gradien skor kelas terhadap feature map konvolusi.
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    # Gradien = seberapa sensitif skor kelas terhadap tiap piksel feature map.
    grads = tape.gradient(class_channel, conv_output)

    # Rata-ratakan gradien tiap channel -> bobot pentingnya channel itu.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Jumlahkan feature map berbobot -> peta panas mentah.
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Buang nilai negatif (ReLU) lalu skala ke 0..1 agar mudah divisualisasi.
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """Tempelkan heatmap di atas gambar asli dengan colormap 'jet'.

    Parameters
    ----------
    heatmap : np.ndarray 2D 0..1 (output make_gradcam_heatmap).
    original_img : PIL.Image.Image | str -- gambar asli (atau path).
    alpha : float -- transparansi heatmap (0 = gambar asli, 1 = heatmap penuh).

    Returns
    -------
    PIL.Image.Image RGB hasil penggabungan.
    """
    if isinstance(original_img, str):
        original_img = Image.open(original_img)
    original_img = original_img.convert("RGB")
    width, height = original_img.size

    # Perbesar heatmap (mis. 7x7) ke ukuran gambar asli.
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (width, height), Image.BILINEAR
    )
    heatmap_arr = np.asarray(heatmap_img) / 255.0

    # Warnai pakai colormap 'jet' (biru=rendah, merah=tinggi); buang channel alpha.
    colormap = matplotlib.colormaps["jet"]
    colored = colormap(heatmap_arr)[..., :3]
    colored_img = Image.fromarray(np.uint8(255 * colored))

    # Campur gambar asli + heatmap berwarna.
    return Image.blend(original_img, colored_img, alpha=alpha)
