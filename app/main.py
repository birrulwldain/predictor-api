# app/main.py

# Impor pustaka yang dibutuhkan
import torch
import numpy as np
import json
from scipy.signal import find_peaks
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Impor dari file lain dalam package 'app' yang telah kita buat
from .model import load_prediction_assets
from .processing import prepare_asc_data

# =============================================================================
# INISIALISASI APLIKASI DAN MODEL (Dijalankan sekali saat startup)
# =============================================================================

# Inisialisasi aplikasi FastAPI
api = FastAPI(
    title="Spectral Predictor API",
    description="API untuk memprediksi komposisi elemen dari data spektrum.",
    version="1.0.0"
)

# Tambahkan Middleware untuk CORS agar frontend bisa mengakses API ini
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua sumber, bisa diperketat untuk produksi
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Muat model dan semua aset penting sekali saja saat server pertama kali dijalankan.
# Ini sangat efisien karena tidak perlu dimuat ulang untuk setiap permintaan.
try:
    print("--- Memuat model dan aset saat startup... ---")
    model, element_map, target_wavelengths = load_prediction_assets()
    class_names = list(element_map.keys())
    print("--- Model dan aset berhasil dimuat. API siap menerima permintaan. ---")
except Exception as e:
    print(f"--- FATAL ERROR SAAT STARTUP: Gagal memuat aset: {e} ---")
    # Jika aset gagal dimuat, aplikasi tidak bisa berjalan.
    model = None 

# =============================================================================
# DEFINISI ENDPOINT API
# =============================================================================

@api.get("/", summary="Endpoint Cek Status")
def read_root():
    """Endpoint sederhana untuk memeriksa apakah API berjalan."""
    return {"status": "ok", "message": "Welcome to the Spectral Predictor API"}


@api.post("/predict", summary="Prediksi Spektrum Elemen")
async def handle_prediction(request: Request):
    """
    Menerima data spektrum .asc mentah, menjalankan prediksi, dan mengembalikan hasilnya.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model tidak berhasil dimuat saat startup. Periksa log server.")

    try:
        # 1. Dapatkan konten file .asc dari body request
        asc_content = await request.body()
        
        # 2. Proses dan siapkan data menggunakan fungsi dari processing.py
        spectrum_data = prepare_asc_data(asc_content.decode('utf-8'), target_wavelengths)
        
        # 3. Deteksi Puncak
        peak_indices, _ = find_peaks(spectrum_data, height=0.03, distance=8)
        
        # 4. Lakukan Prediksi
        input_tensor = torch.from_numpy(spectrum_data[np.newaxis, :, np.newaxis])
        with torch.no_grad():
            output_logits = model(input_tensor)
            full_probabilities = torch.sigmoid(output_logits).squeeze(0).cpu().numpy()
        full_predictions = (full_probabilities > 0.5).astype(int)

        # 5. Format Anotasi untuk dikirim ke Plotly di frontend
        annotations = []
        for i, peak_idx in enumerate(peak_indices):
            prediction_at_peak = full_predictions[peak_idx]
            detected_indices = np.where(prediction_at_peak == 1)[0]
            if detected_indices.size > 0:
                label_lines = [f"{class_names[j]} ({full_probabilities[peak_idx, j]*100:.1f}%)" for j in detected_indices if class_names[j] != 'background']
                if label_lines:
                    annotations.append(dict(
                        x=float(target_wavelengths[peak_idx]), 
                        y=float(spectrum_data[peak_idx]),
                        text="<br>".join(label_lines), showarrow=True, arrowhead=2, ax=0, ay=-(40 + (i % 5) * 20),
                        font=dict(color="#ffffff"), bgcolor="#003366", opacity=0.8, borderpad=4
                    ))
        
        # 6. Kemas semua hasil ke dalam dictionary. FastAPI akan otomatis mengubahnya menjadi JSON.
        results = {
            "wavelengths": target_wavelengths.tolist(),
            "spectrum_data": spectrum_data.tolist(),
            "peak_wavelengths": target_wavelengths[peak_indices].tolist(),
            "peak_intensities": spectrum_data[peak_indices].tolist(),
            "annotations": annotations,
        }
        
        return results

    except Exception as e:
        # Kirim respons error yang jelas jika terjadi masalah saat pemrosesan
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {str(e)}")