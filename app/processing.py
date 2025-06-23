# app/processing.py

import pandas as pd
import numpy as np
from io import StringIO

TARGET_MAX_INTENSITY = 0.8

def prepare_asc_data(
    asc_content_string: str,
    target_wavelengths: np.ndarray,
    baseline_poly_order: int = 5
) -> np.ndarray:
    """Memproses string konten file .asc menjadi spektrum yang siap diprediksi."""
    try:
        df = pd.read_csv(StringIO(asc_content_string), sep=r'\s+', names=['wavelength', 'intensity'], comment='#')
        original_wavelengths, original_spectrum = df['wavelength'].values, df['intensity'].values
    except Exception as e:
        raise ValueError(f"Gagal mem-parsing data ASC. Pastikan formatnya benar. Detail: {e}")

    # Koreksi Baseline
    anchor_points = np.percentile(original_spectrum, 10)
    baseline_points = original_spectrum < anchor_points
    if np.sum(baseline_points) > baseline_poly_order:
        coeffs = np.polyfit(original_wavelengths[baseline_points], original_spectrum[baseline_points], deg=baseline_poly_order)
        baseline = np.polyval(coeffs, original_wavelengths)
        spectrum_corrected = original_spectrum - baseline
        spectrum_corrected[spectrum_corrected < 0] = 0
    else:
        spectrum_corrected = original_spectrum - np.min(original_spectrum)

    # Normalisasi
    max_val = np.max(spectrum_corrected)
    processed_spectrum = (spectrum_corrected / max_val) * TARGET_MAX_INTENSITY if max_val > 0 else spectrum_corrected
    
    # Resampling
    resampled_spectrum = np.interp(target_wavelengths, original_wavelengths, processed_spectrum)
    return resampled_spectrum.astype(np.float32)