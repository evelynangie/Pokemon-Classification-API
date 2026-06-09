# Product Requirements Document (PRD) & Technical Documentation

## Pokemon Classification API

Sistem ini adalah Web Service komersial berkinerja tinggi untuk mengklasifikasikan tipe Pokemon. Model yang awalnya berbasis eksperimen monolitik telah direstrukturisasi menjadi arsitektur modular menggunakan FastAPI, PyTorch, dan MobileNetV2. Sistem ini siap untuk diintegrasikan dengan aplikasi klien (Frontend/Mobile).

## 1. Struktur Direktori

```
PokemonClassification/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── predictor.py
├── data/
│   ├── images/
│   └── pokemon.csv
├── models/
│   └── model.pkl
├── build_model.py
├── convert_base64.py
├── test_api.py
├── requirements.txt
└── README.md
```

## 2. Komponen Utama

| **Komponen**          | **Deskripsi**                                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **app/main.py**       | Titik masuk utama aplikasi FastAPI yang menangani perutean HTTP dan komunikasi data.                       |
| **app/predictor.py**  | Kelas modular terisolasi untuk memuat model, melakukan pra-pemrosesan gambar, dan mengeksekusi inferensi.  |
| **build_model.py**    | Skrip pelatihan model menggunakan MobileNetV2 dengan teknik transfer learning (freezing/unfreezing layer). |
| **convert_base64.py** | Utilitas CLI untuk mengonversi file gambar lokal menjadi representasi string base64.                       |
| **test_api.py**       | Utilitas CLI untuk melakukan simulasi pengiriman HTTP POST request secara terotomatisasi.                  |

## 3. Panduan Instalasi Sistem

Pastikan Python 3.9 atau versi lebih baru telah terpasang di sistem operasi. Disarankan menggunakan lingkungan virtual.

**A. Membuat dan Mengaktifkan Virtual Environment**

**Bash**

```
python -m venv .venv
```

*Untuk Windows:*

**Bash**

```
.venv\Scripts\activate
```

*Untuk Linux / macOS:*

**Bash**

```
source .venv/bin/activate
```

**B. Instalasi Dependensi**

**Bash**

```
pip install -r requirements.txt
```

## 4. Pelatihan Model (Opsional)

Jika Anda ingin melatih ulang model atau memperbarui bobot klasifikasi, jalankan skrip `build_model.py`. Skrip ini akan membaca dataset dari `data/pokemon.csv` dan memproses gambar dari `data/images/`.

**Bash**

```
python build_model.py
```

> Catatan: Output pelatihan berupa file biner akan otomatis disimpan dan menimpa file `models/model.pkl`.

## 5. Menjalankan Web Service API

Jalankan server menggunakan Uvicorn agar aplikasi dapat menerima lalu lintas permintaan.

**Bash**

```
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* **API berjalan pada:** `http://localhost:8000/`
* **Dokumentasi Interaktif (Swagger UI):** `http://localhost:8000/docs`

## 6. Pengujian API

Sistem menyediakan skrip khusus untuk menghindari kegagalan peramban akibat menyalin teks base64 yang terlalu panjang secara manual.

**Simulasi API dengan gambar lokal:**

**Bash**

```
python test_api.py data/images/Pikachu.png
```

**Melihat representasi base64 dari sebuah gambar:**

**Bash**

```
python convert_base64.py data/images/Pikachu.png
```

## 7. Spesifikasi Endpoint dan Protokol Komunikasi

**Rute Utama:** `POST /predict`

**Headers:** `Content-Type: application/json`

Sistem menerima HTTP POST dengan *payload* JSON berisi representasi teks Base64 dari gambar yang ingin diprediksi.

### Skenario Respons

| **Skenario** | **Kode Status** | **Keterangan**                                                                              |
| ------------------ | --------------------- | ------------------------------------------------------------------------------------------------- |
| **Valid**    | 200 OK                | Prediksi berhasil diproses dan dikembalikan.                                                      |
| **Gagal**    | 400 Bad Request       | Teks bukan base64, tipe data kosong, atau string terkorupsi.                                      |
| **Gagal**    | 500 Internal Error    | Model `.pkl`tidak ditemukan atau terjadi kegagalan perangkat keras (contoh: memori CUDA penuh). |

### Detail Payload JSON

**Request JSON:**

**JSON**

```
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Response Valid (200 OK):**

**JSON**

```
{
  "status": "success",
  "data": {
    "type": "Electric",
    "confidence": 0.9845
  }
}
```

**Response Gagal (400 Bad Request):**

**JSON**

```
{
  "detail": "Format base64 tidak valid atau data korup"
}
```

**Response Gagal (500 Internal Server Error):**

**JSON**

```
{
  "detail": "Terjadi kesalahan pemrosesan gambar: [Pesan teknis spesifik]"
}
```
