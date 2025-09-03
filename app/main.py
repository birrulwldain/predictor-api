# app/main.py

import torch
import numpy as np
import io
import pandas as pd
import traceback
from typing import Optional, List
from scipy.signal import find_peaks
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Impor dari file lain dalam package 'app'
from .model import load_assets, als_baseline_correction
from .processing import prepare_asc_data

# Inisialisasi API
app = FastAPI(title="Machine Learning-Powered Spectroscopic Data Interpreter API")

# Atur CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ================================================================
# Muat Aset Model (dijalankan sekali saat startup)
# ================================================================
model, element_map, target_wavelengths = load_assets()
class_names = list(element_map.keys())

# ================================================================
# Pydantic Models (Definisi Input)
# ================================================================
class PredictWithProminenceInput(BaseModel):
    asc_content: str
    prominence: Optional[float] = None
    apply_baseline_correction: bool = False
    lam: Optional[float] = None
    p: Optional[float] = None
    niter: Optional[int] = None
    distance: Optional[float] = None
    height: Optional[float] = None
    width: Optional[float] = None
    threshold: float = 0.6

class ValidateInput(PredictWithProminenceInput):
    ground_truth_elements: List[str]

# ================================================================
# ENDPOINTS API
# ================================================================

@app.get("/", summary="Status Check Endpoint")
def read_root():
    return {"status": "ok", "message": "Welcome to the Spectral Predictor API"}

@app.post("/predict_with_prominence", summary="Predict Spectral Elements (Original Endpoint)")
async def handle_prediction_with_prominence(input_data: PredictWithProminenceInput):
    try:
        # --- Logika Prediksi Lengkap ---
        spectrum_data = prepare_asc_data(input_data.asc_content, target_wavelengths)
        original_spectrum = spectrum_data.copy()

        baseline_data = None
        if input_data.apply_baseline_correction and input_data.lam is not None and input_data.p is not None:
            niter = input_data.niter if input_data.niter is not None else 10
            baseline_data = als_baseline_correction(spectrum_data, input_data.lam, input_data.p, niter)
            spectrum_data = spectrum_data - baseline_data

        peak_indices, _ = find_peaks(
            spectrum_data, prominence=input_data.prominence,
            distance=input_data.distance if input_data.distance is not None else 1,
            height=input_data.height, width=input_data.width
        )

        input_tensor = torch.from_numpy(spectrum_data[np.newaxis, :, np.newaxis]).float()
        with torch.no_grad():
            output_logits = model(input_tensor)
            full_probabilities = torch.sigmoid(output_logits).squeeze(0).cpu().numpy()

        # <<< BAGIAN PENTING: LOGIKA ANOTASI DIMASUKKAN DI SINI >>>
        annotations = []
        for i, peak_idx in enumerate(peak_indices):
            prediction_at_peak = (full_probabilities[peak_idx] > input_data.threshold).astype(int)
            detected_indices = np.where(prediction_at_peak == 1)[0]
            if detected_indices.size > 0:
                label_lines = [f"{class_names[j]} ({full_probabilities[peak_idx, j]*100:.1f}%)" for j in detected_indices if class_names[j] != 'background' and full_probabilities[peak_idx, j] > 0.55]
                if label_lines:
                    annotations.append(dict(
                        x=float(target_wavelengths[peak_idx]), y=float(spectrum_data[peak_idx]),
                        text="<br>".join(label_lines), showarrow=True, arrowhead=2, ax=0, ay=-(40 + (i % 5) * 20)
                    ))

        return {
            "wavelengths": target_wavelengths.tolist(), "original_spectrum": original_spectrum.tolist(),
            "spectrum_data": spectrum_data.tolist(), "baseline": baseline_data.tolist() if baseline_data is not None else None,
            "peak_wavelengths": target_wavelengths[peak_indices].tolist(), "peak_intensities": spectrum_data[peak_indices].tolist(),
            "annotations": annotations
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.post("/validate", summary="Validate and get JSON results including metrics")
async def validate_and_get_json(validation_data: ValidateInput):
    try:
        # --- 1. Logika Prediksi Lengkap (Diulang agar mandiri) ---
        spectrum_data = prepare_asc_data(validation_data.asc_content, target_wavelengths)
        original_spectrum = spectrum_data.copy()
        baseline_data = None
        if validation_data.apply_baseline_correction and validation_data.lam is not None and validation_data.p is not None:
            niter = validation_data.niter if validation_data.niter is not None else 10
            baseline_data = als_baseline_correction(spectrum_data, validation_data.lam, validation_data.p, niter)
            spectrum_data = spectrum_data - baseline_data
        
        peak_indices, _ = find_peaks(
            spectrum_data, prominence=validation_data.prominence,
            distance=validation_data.distance if validation_data.distance is not None else 1,
            height=validation_data.height, width=validation_data.width
        )
        
        input_tensor = torch.from_numpy(spectrum_data[np.newaxis, :, np.newaxis]).float()
        with torch.no_grad():
            output_logits = model(input_tensor)
            full_probabilities = torch.sigmoid(output_logits).squeeze(0).cpu().numpy()

        # --- 2. Logika Validasi Lengkap ---
        predicted_elements_with_locations = {}
        for peak_idx in peak_indices:
            prediction_at_peak = (full_probabilities[peak_idx] > validation_data.threshold).astype(int)
            detected_indices = np.where(prediction_at_peak == 1)[0]
            for idx in detected_indices:
                element_name = class_names[idx]
                if element_name != 'background':
                    if element_name not in predicted_elements_with_locations:
                        predicted_elements_with_locations[element_name] = []
                    peak_wavelength = round(target_wavelengths[peak_idx], 2)
                    predicted_elements_with_locations[element_name].append(peak_wavelength)
        
        predicted_elements_set = set(predicted_elements_with_locations.keys())
        ground_truth_set = set(validation_data.ground_truth_elements)
        true_positives_set = predicted_elements_set.intersection(ground_truth_set)
        false_positives_set = predicted_elements_set.difference(ground_truth_set)
        false_negatives_set = ground_truth_set.difference(predicted_elements_set)
        
        tp = len(true_positives_set); fp = len(false_positives_set); fn = len(false_negatives_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        summary_metrics = {
            "True Positives (TP)": tp, "False Positives (FP)": fp, "False Negatives (FN)": fn,
            "Precision": f"{precision:.2%}", "Recall": f"{recall:.2%}", "F1-Score": f"{f1_score:.2%}"
        }

        validation_table = []
        def format_locations(locations): return '; '.join(map(str, locations))
        for el in sorted(list(true_positives_set)):
            locs = predicted_elements_with_locations.get(el, [])
            validation_table.append({"Elemen": el, "Status Prediksi": "True Positive", "Jumlah Puncak": len(locs), "Lokasi Puncak (nm)": format_locations(locs)})
        for el in sorted(list(false_positives_set)):
            locs = predicted_elements_with_locations.get(el, [])
            validation_table.append({"Elemen": el, "Status Prediksi": "False Positive", "Jumlah Puncak": len(locs), "Lokasi Puncak (nm)": format_locations(locs)})
        for el in sorted(list(false_negatives_set)):
            validation_table.append({"Elemen": el, "Status Prediksi": "False Negative", "Jumlah Puncak": 0, "Lokasi Puncak (nm)": "-"})

        # <<< BAGIAN PENTING: LOGIKA ANOTASI SEKARANG DIMASUKKAN JUGA DI SINI >>>
        annotations = []
        for i, peak_idx in enumerate(peak_indices):
            prediction_at_peak = (full_probabilities[peak_idx] > validation_data.threshold).astype(int)
            detected_indices = np.where(prediction_at_peak == 1)[0]
            if detected_indices.size > 0:
                label_lines = [f"{class_names[j]} ({full_probabilities[peak_idx, j]*100:.1f}%)" for j in detected_indices if class_names[j] != 'background' and full_probabilities[peak_idx, j] > 0.55]
                if label_lines:
                    annotations.append(dict(
                        x=float(target_wavelengths[peak_idx]), y=float(spectrum_data[peak_idx]),
                        text="<br>".join(label_lines), showarrow=True, arrowhead=2, ax=0, ay=-(40 + (i % 5) * 20)
                    ))

        return {
            "wavelengths": target_wavelengths.tolist(), "original_spectrum": original_spectrum.tolist(),
            "spectrum_data": spectrum_data.tolist(), "baseline": baseline_data.tolist() if baseline_data is not None else None,
            "peak_wavelengths": target_wavelengths[peak_indices].tolist(), "peak_intensities": spectrum_data[peak_indices].tolist(),
            "annotations": annotations, # <-- Anotasi sekarang ada di sini
            "validation_table": validation_table, 
            "summary_metrics": summary_metrics
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/download_excel", summary="Generate a 2-sheet Excel report with quantitative summary")
async def download_excel(validation_data: ValidateInput):
    try:
        # Panggil endpoint /validate untuk mendapatkan data yang sudah lengkap
        full_results = await validate_and_get_json(validation_data)
        
        # Siapkan data untuk kedua sheet
        df_details = pd.DataFrame(full_results.get("validation_table", []))
        df_summary = pd.DataFrame(list(full_results.get("summary_metrics", {}).items()), columns=['Metrik', 'Nilai'])
        
        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            df_details.to_excel(writer, sheet_name='Detail Validasi', index=False)
            df_summary.to_excel(writer, sheet_name='Ringkasan Kuantitatif', index=False)
        
        output_buffer.seek(0)
        
        return StreamingResponse(
            output_buffer,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={"Content-Disposition": "attachment; filename=laporan_validasi_kuantitatif.xlsx"}
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")