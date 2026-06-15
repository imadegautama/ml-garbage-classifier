# Dataset — Klasifikasi Sampah (TrashNet / Garbage Classification)

Data mentah **tidak di-commit** ke repository (ukurannya besar dan bukan kode).
Bagian ini menjelaskan cara mendapatkannya kembali.

## Sumber
- **Kaggle:** `asdasdasasdas/garbage-classification`
  <https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification>
- Berasal dari **TrashNet** (Stanford, Gary Thung & Mindy Yang).

## Kelas (6)
`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
(±2.500 gambar; kelas `trash` paling sedikit → catat soal *class imbalance*).

## Struktur yang diharapkan setelah diekstrak
```
data/dataset/
├── cardboard/   *.jpg
├── glass/       *.jpg
├── metal/       *.jpg
├── paper/       *.jpg
├── plastic/     *.jpg
└── trash/       *.jpg
```

## Cara unduh

### Opsi A — di Google Colab (disarankan, dipakai notebook training)
```python
import kagglehub
path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
print("Dataset tersimpan di:", path)
```

### Opsi B — manual
1. Buka tautan Kaggle di atas, klik **Download**.
2. Ekstrak isinya ke `data/dataset/` mengikuti struktur folder di atas.

> Notebook `notebooks/train_model.ipynb` akan otomatis mencari folder kelas ini.
