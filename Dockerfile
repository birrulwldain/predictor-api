# Dockerfile

# Memulai dari image Python resmi yang ringan
FROM python:3.11-slim

# Menetapkan direktori kerja utama di dalam kontainer
WORKDIR /app

# Salin file requirements.txt terlebih dahulu untuk optimasi cache
COPY requirements.txt .

# Instal semua dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin direktori assets dan direktori aplikasi Anda
COPY assets/ ./assets/
COPY app/ ./app/

# Memberitahu Docker bahwa kontainer akan berjalan di port 8080
EXPOSE 8080

# Perintah untuk menjalankan server API FastAPI saat kontainer dimulai
CMD ["uvicorn", "app.main:api", "--host", "0.0.0.0", "--port", "8080"]