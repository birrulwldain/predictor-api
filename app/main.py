# app/main.py

import torch
import numpy as np
import json
from scipy.signal import find_peaks, savgol_filter
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import traceback # Add this import

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

from pydantic import BaseModel

class PredictWithProminenceInput(BaseModel):
    asc_content: str
    prominence: float | None = None
    apply_baseline_correction: bool = False
    apply_auc_normalization: bool = False

@api.post("/predict_with_prominence", summary="Predict Spectral Elements with Prominence Parameter")
async def handle_prediction_with_prominence(input_data: PredictWithProminenceInput):
    try:
        asc_content_str = input_data.asc_content

        # Proses data mentah
        spectrum_data = prepare_asc_data(asc_content_str, target_wavelengths)

        # Apply baseline correction if requested
        if input_data.apply_baseline_correction:
            # A simple baseline correction using Savitzky-Golay filter to estimate baseline
            # and then subtract it. Parameters might need tuning.
            window_length = min(len(spectrum_data) - 1 if len(spectrum_data) % 2 == 0 else len(spectrum_data), 51) # Ensure odd and within bounds
            polyorder = min(window_length - 1, 3) # Ensure polyorder < window_length
            if window_length > 1 and polyorder >= 0:
                baseline_estimated = savgol_filter(spectrum_data, window_length, polyorder)
                spectrum_data = spectrum_data - baseline_estimated
            else:
                print("Warning: Skipping Savitzky-Golay filter due to insufficient data points or invalid parameters.")

        # Apply AUC normalization if requested
        if input_data.apply_auc_normalization:
            # Calculate AUC using trapezoidal rule
            auc = np.trapz(spectrum_data, target_wavelengths)
            if auc != 0:
                spectrum_data = spectrum_data / auc
            else:
                spectrum_data = np.zeros_like(spectrum_data) # Avoid division by zero

        # Tentukan prominence yang akan digunakan
        # Gunakan prominence yang disediakan, jika tidak ada, default ke 0.03
        peak_prominence = input_data.prominence if input_data.prominence is not None else 0.03

        # Deteksi Puncak
        # Gunakan prominence yang disediakan, pertahankan distance hardcoded untuk saat ini
        peak_indices, _ = find_peaks(spectrum_data, prominence=peak_prominence, distance=8)

        # Lakukan Prediksi (logika prediksi lainnya tetap sama)
        input_tensor = torch.from_numpy(spectrum_data[np.newaxis, :, np.newaxis])
        with torch.no_grad():
            output_logits = model(input_tensor)
            full_probabilities = torch.sigmoid(output_logits).squeeze(0).cpu().numpy()
        full_predictions = (full_probabilities > 0.7).astype(int) # Pertahankan ambang batas 0.7

        # Format anotasi untuk Plotly
        annotations = []
        for i, peak_idx in enumerate(peak_indices):
            prediction_at_peak = full_predictions[peak_idx]
            detected_indices = np.where(prediction_at_peak == 1)[0]
            if detected_indices.size > 0:
                label_lines = [f"{class_names[j]} ({full_probabilities[peak_idx, j]*100:.1f}%)" for j in detected_indices if class_names[j] != 'background' and full_probabilities[peak_idx, j] > 0.7]
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
        full_predictions = (full_probabilities > 0.7).astype(int)

        # Format anotasi untuk Plotly
        annotations = []
        for i, peak_idx in enumerate(peak_indices):
            prediction_at_peak = full_predictions[peak_idx]
            detected_indices = np.where(prediction_at_peak == 1)[0]
            if detected_indices.size > 0:
                label_lines = [f"{class_names[j]} ({full_probabilities[peak_idx, j]*100:.1f}%)" for j in detected_indices if class_names[j] != 'background' and full_probabilities[peak_idx, j] > 0.7]
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