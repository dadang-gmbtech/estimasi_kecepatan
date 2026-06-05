# 🚗 Interactive Vehicle Speed Estimator (Dynamic Depth-Scaling)

Aplikasi ini adalah sistem estimasi kecepatan kendaraan berbasis video menggunakan deteksi objek (YOLO) dan pelacakan (ByteTrack). Aplikasi ini menggunakan profil pergerakan kendaraan referensi untuk mengatasi masalah distorsi perspektif. Kecepatan dihitung menggunakan *Dynamic Depth-Scaling* sehingga hasilnya konsisten dari jauh maupun dekat kamera.

## File Presentasi, Video Ground Truth dan Laporan ada di Link Berikut  https://s.id/ck8IU

## 📋 Prasyarat (Prerequisites)

Sebelum menjalankan aplikasi, pastikan sistem Anda memenuhi persyaratan berikut:
- **Sistem Operasi**: Windows 10/11, macOS, atau Linux.
- **Python**: Versi 3.8 hingga 3.11 (disarankan Python 3.10).
- **Hardware**: 
  - CPU (bisa berjalan namun pemrosesan video mungkin lebih lambat).
  - GPU NVIDIA dengan dukungan CUDA sangat disarankan untuk inference YOLO yang jauh lebih cepat.

## 🛠 Instalasi

1. **Clone repository ini** (atau unduh source code ke komputer Anda):
   ```bash
   git clone https://github.com/dadang-gmbtech/estimasi_kecepatan.git
   cd estimasi_kecepatan
   ```

2. **Buat Virtual Environment** (Opsional tapi sangat disarankan agar library tidak bentrok):
   ```bash
   # Pengguna Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Pengguna macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependensi**:
   Instal semua library yang dibutuhkan menggunakan `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   *Catatan: Proses ini akan mengunduh library seperti `ultralytics`, `opencv-python`, dan `streamlit`.*
   *Saat pertama kali dijalankan, aplikasi juga mungkin akan mengunduh model bobot `yolo11s.pt` secara otomatis.*

## 🚀 Cara Menjalankan Aplikasi

Aplikasi ini dibangun menggunakan antarmuka grafis interaktif berbasis **Streamlit**.

1. Buka Terminal / Command Prompt di folder project (pastikan virtual environment sudah aktif).
2. Jalankan perintah berikut:
   ```bash
   streamlit run app_interactive.py
   ```
3. Browser Anda akan secara otomatis membuka halaman aplikasi (biasanya di `http://localhost:8501`). Jika tidak terbuka otomatis, Anda bisa menyalin link yang muncul di terminal dan membukanya di browser.

## 💡 Panduan Penggunaan Aplikasi (UI)

1. **Upload Video**: Unggah file video (.mp4, .mov, atau .avi) yang berisi rekaman pergerakan kendaraan.
2. **Kalibrasi Referensi Dinamis**:
   - Pilih **satu kendaraan** di video yang terlihat bergerak secara stabil dari jarak jauh hingga dekat kamera dengan durasi waktu yang cukup.
   - Gunakan alat *drawing* (berbentuk kotak) yang ada di layar untuk menggambar *bounding box* (kotak) pada kendaraan tersebut.
   - Masukkan **Kecepatan Aktual** dari kendaraan referensi tersebut dalam satuan km/jam (misalnya: 60 km/jam).
3. **Mulai Proses**: Klik tombol `▶ Mulai Proses`. Aplikasi akan:
   - Menjalankan pelacakan (tracking) menggunakan model YOLO.
   - Membangun model matematika *Depth-Scaling* berbasis pergerakan kendaraan referensi yang Anda pilih untuk mengkalibrasi jarak piksel terhadap jarak nyata.
   - Mengestimasi kecepatan seluruh kendaraan lain secara dinamis.
4. **Hasil Estimasi**: Setelah proses selesai, Anda dapat memutar video hasil yang sudah dilengkapi anotasi (kotak pelacakan & kecepatan), melihat grafik statistik kecepatan, distribusi histogram, dan mengunduh data mentah riwayat kecepatan semua kendaraan dalam format CSV.
