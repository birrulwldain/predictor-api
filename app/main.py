# app/main.py

import torch
import numpy as np
import json
from scipy.signal import find_peaks
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Impor dari file lain dalam package 'app'
from .model import load_assets
from .processing import prepare_asc_data

# Inisialisasi API
api = FastAPI(title="Spectral Predictor API")

# Atur CORS Middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Muat model dan aset sekali saja saat server pertama kali dijalankan
model, element_map, target_wavelengths = load_assets()
class_names = list(element_map.keys())

# Definisikan endpoint API untuk prediksi
@api.post("/predict", summary="Prediksi Spektrum Elemen")
async def handle_prediction(request: Request):
    try:
        # Dapatkan konten file .asc dari body request
        asc_content = await request.body()
        
        # Proses data mentah
        spectrum_data = prepare_asc_data(asc_content.decode('utf-8'), target_wavelengths)
        
        # Deteksi Puncak
        peak_indices, _ = find_peaks(spectrum_data, height=0.03, distance=8)
        
        # Lakukan Prediksi
        input_tensor = torch.from_numpy(spectrum_data[np.newaxis, :, np.newaxis])
        with torch.no_grad():
            output_logits = model(input_tensor)
            full_probabilities = torch.sigmoid(output_logits).squeeze(0).cpu().numpy()
        full_predictions = (full_probabilities > 0.5).astype(int)

        # Format anotasi untuk Plotly
        annotations = []
        for i, peak_idx in enumerate(peak_indices):
            prediction_at_peak = full_predictions[peak_idx]
            detected_indices = np.where(prediction_at_peak == 1)[0]
            if detected_indices.size > 0:
                label_lines = [f"{class_names[j]} ({full_probabilities[peak_idx, j]*100:.1f}%)" for j in detected_indices if class_names[j] != 'background']
                if label_lines:
                    annotations.append(dict(
                        x=float(target_wavelengths[peak_idx]), y=float(spectrum_data[peak_idx]),
                        text="<br>".join(label_lines), showarrow=True, arrowhead=2, ax=0, ay=-(40 + (i % 5) * 20),
                        font=dict(color="#ffffff"), bgcolor="#003366", opacity=0.8, borderpad=4
                    ))
        
        # Kemas semua hasil ke dalam JSON untuk dikirim kembali ke frontend
        results = {
            "wavelengths": target_wavelengths.tolist(),
            "spectrum_data": spectrum_data.tolist(),
            "peak_wavelengths": target_wavelengths[peak_indices].tolist(),
            "peak_intensities": spectrum_data[peak_indices].tolist(),
            "annotations": annotations,
        }
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi error di server: {str(e)}")

@api.get("/", summary="Endpoint Cek Status")
def read_root():
    return {"status": "ok", "message": "Welcome to the Spectral Predictor API"}