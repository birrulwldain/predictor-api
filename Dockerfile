# Dockerfile

# --- Tahap 1: Base Image ---
# Memulai dari image Python resmi yang ringan dan efisien.
FROM python:3.11-slim

# --- Tahap 2: Menyiapkan Lingkungan di Dalam Kontainer ---
# Menetapkan direktori kerja utama di dalam kontainer.
# Semua perintah selanjutnya akan dijalankan dari direktori ini.
WORKDIR /app

# --- Tahap 3: Instalasi Dependensi ---
# Salin file requirements.txt terlebih dahulu.
# Ini adalah optimasi: Docker akan menyimpan lapisan ini dalam cache.
# Jika Anda tidak mengubah requirements.txt, langkah ini tidak akan dijalankan ulang saat build berikutnya.
COPY requirements.txt .

# Jalankan pip untuk menginstal semua pustaka yang dibutuhkan.
# --no-cache-dir membuat ukuran image akhir lebih kecil.
RUN pip install --no-cache-dir -r requirements.txt

# --- Tahap 4: Salin Kode dan Aset Aplikasi ---
# Salin direktori 'assets' yang berisi model dan file json Anda.
COPY assets/ ./assets/

# Salin direktori 'app' yang berisi semua kode Python Anda.
COPY app/ ./app/

# --- Tahap 5: Konfigurasi Jaringan dan Perintah Eksekusi ---
# Memberitahu Docker bahwa kontainer akan mendengarkan permintaan di port 8080.
EXPOSE 8080

# Perintah yang akan dijalankan saat kontainer dimulai.
# Ini akan menjalankan server web FastAPI menggunakan Uvicorn.
# --host 0.0.0.0 membuatnya bisa diakses dari luar kontainer.
CMD exec uvicorn app.main:api --host 0.0.0.0 --port ${PORT:-8080}