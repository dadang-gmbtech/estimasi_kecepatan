

#Deskripsi singkat proyek ini: estimasi kecepatan kendaraan dari video atau gambar.


### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python app.py --input video.mp4
```

### Referensi Kecepatan Relatif

Jika ingin menggunakan satu kendaraan sebagai referensi kecepatan, tersedia dua cara:

1. Command-line:
```bash
python speed_ref.py --input video.mp4 --ref-id 3 --ref-speed 60.0
```
2. GUI Streamlit:
```bash
streamlit run app_gui.py
```

- `--ref-id` adalah ID kendaraan referensi dari hasil pelacakan.
- `--ref-speed` adalah kecepatan nyata kendaraan referensi dalam `km/h`.
- `app_gui.py` menyediakan antarmuka untuk memilih kendaraan referensi, memasukkan kecepatan nyata, dan melihat hasil di browser.

### Model

Letakkan `yolo11n.pt` di folder proyek atau ubah path di konfigurasi.

---

Jika mau, saya bisa menyesuaikan README ini dengan deskripsi proyek Anda, menambahkan contoh perintah lengkap, atau menaruh badge dan `LICENSE`.

## System requirements & Preparation

- **Operating system:** Windows 10/11, Ubuntu 18.04+ or macOS 10.15+. Linux is recommended for GPU workflows.
- **Python:** 3.8 — 3.11 (3.10 recommended). Gunakan virtual environment (`venv`/`conda`).
- **Hardware:**
  - CPU-only: 4+ cores, 8+ GB RAM (untuk pengujian kecil).
  - GPU (opsional, untuk inference cepat): NVIDIA GPU dengan CUDA 11+ jika memakai GPU-accelerated model.
- **Disk / files:** Pastikan ruang cukup untuk model dan dataset. Model `yolo11n.pt` biasanya beberapa puluhan MB — cek ukurannya.

### Dependencies

1. Pastikan `requirements.txt` berisi semua dependensi (contoh: `torch`, `opencv-python`, `numpy`, `ultralytics` / `yolov5` bila dipakai).
2. Instal di virtualenv:

```bash
python -m venv .venv
.\.venv\Scripts\activate    # Windows PowerShell
source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
```

atau dengan `conda`:

```bash
conda create -n estimasi python=3.10
conda activate estimasi
pip install -r requirements.txt
```

### Environment / Config

- Jika ada variabel konfigurasi (mis. path model, device), letakkan di file `.env` atau argumen baris perintah. Contoh env:

```
MODEL_PATH=yolo11n.pt
DEVICE=cuda
```

- Di `app.py` pastikan parameter input/output mudah diubah lewat argumen.

### Apa yang perlu di-upload ke GitHub

- `app.py`, `requirements.txt`, `README.md`, dan skrip/utility yang diperlukan.
- File model besar: jangan commit file model besar langsung ke Git jika ukurannya besar (>50 MB). Opsi yang direkomendasikan:
  - Gunakan GitHub Releases untuk menyertakan file model.
  - Atau gunakan Git LFS (`git lfs install` lalu `git lfs track "*.pt"`).
  - Atau simpan di cloud storage (Google Drive / S3) dan berikan link di README.
- Contoh file yang boleh di-commit: `yolo11n.pt` hanya jika ukurannya kecil; jika besar, beri instruksi download di README.
- Tambahkan `.gitignore` untuk menghindari commit file environment dan dataset besar, contohnya:

```
.venv/
__pycache__/
*.pyc
data/
*.pt  # gunakan LFS jika ingin track model
```

### Cara mengecek setelah persiapan

1. Buat virtualenv dan instal dependensi.
2. Jalankan contoh singkat:

```bash
python app.py --input sample_video.mp4
```

3. Pastikan output muncul dan tidak ada error import.

---

Jika Anda ingin, saya bisa:
- Menambahkan contoh environment `.env` dan file `.gitignore` ke repo.
- Menambahkan instruksi untuk meng-upload model menggunakan Releases atau Git LFS.

